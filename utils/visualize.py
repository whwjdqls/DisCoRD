import os
from os.path import join as pjoin
from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter,FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3


from common.quaternion import *
from .paramUtil import *

# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_rot(data, joints_num, skeleton):
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions

# radius = 10*offsets
def plot_3d_motion_kit(save_path, joints, title, figsize=(5, 5), interval=100, radius=246 * 12):
    matplotlib.use('Agg')
    
    if joints.shape[1] == 251:
        joints = recover_from_ric(torch.from_numpy(joints).unsqueeze(0).float(), 21).squeeze().cpu().numpy()

    assert joints.shape[1] == 21 and (joints.shape[2] == 3), f"joints shape is {joints.shape}"
    
    kinematic_tree = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]]

    title_sp = title.split(' ')
    if len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])
    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        # print(title)
        fig.suptitle(title)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.reshape(len(joints), -1, 3)
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'magenta', 'black', 'green', 'blue', 'red', 'magenta', 'black', 'green', 'blue']
    frame_number = data.shape[0]
    #     print(data.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    #     print(trajec.shape)

    def update(index):
        #         print(index)
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=110, azim=-90)
        ax.dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0], MAXS[0], 0, MINS[2], MAXS[2])
        ax.scatter(data[index, :, 0], data[index, :, 1], data[index, :, 2], color='black')
        for chain, color in zip(kinematic_tree, colors):
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=2.0, color=color)
        #         print(trajec[:index, 0].shape)
        if index > 1:
            ax.plot3D(trajec[:index, 0], np.zeros_like(trajec[:index, 0]), trajec[:index, 1], linewidth=1.0,
                      color='blue')
        #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=interval, repeat=True, repeat_delay=50)

    ani.save(save_path, writer='pillow')
    plt.close()
    
def plot_3d_motion(save_path, joints, title, figsize=(10, 10), fps=120, radius=4, fontsize=10):
    """
    NOTE: Only support Humanml3d dataset
    joints (numpy array): (seq_len, 22, 3), if (seq_len, 263), then recover from RIC
    title (str): title of the animation

    """
    if joints.shape[1] == 263:
        joints = recover_from_ric(torch.from_numpy(joints).unsqueeze(0).float(), 22).squeeze().cpu().numpy()

    # assert and if not, raise error
    assert joints.shape[1] == 22 and (joints.shape[2] == 3), f"joints shape is {joints.shape}"
    
    kinematic_tree = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
    title_sp = title.split(' ')
    if len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])
    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        # print(title)
        fig.suptitle(title, fontsize=fontsize)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    data = joints.copy().reshape(len(joints), -1, 3)
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',  
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
             'darkred', 'darkred','darkred','darkred','darkred']
    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]
    
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90) 
        ax.dist = 7.5
        plot_xzPlane(MINS[0]-trajec[index, 0], MAXS[0]-trajec[index, 0], 0, MINS[2]-trajec[index, 1], MAXS[2]-trajec[index, 1])
        
        if index > 1:
            ax.plot3D(trajec[:index, 0]-trajec[index, 0], np.zeros_like(trajec[:index, 0]), trajec[:index, 1]-trajec[index, 1], linewidth=1.0,
                      color='blue')
        
        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth, color=color)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000/fps, repeat=False)
    ani.save(save_path, writer=PillowWriter(fps=fps))
    plt.close()

def plot_3d_motion_osx(save_path, joints, title, figsize=(10, 10), fps=120, radius=4, fontsize=10):
    """
    NOTE: Only support Humanml3d dataset
    joints (numpy array): (seq_len, 22, 3), if (seq_len, 263), then recover from RIC
    title (str): title of the animation

    """
    if joints.shape[1] == 263:
        joints = recover_from_ric(torch.from_numpy(joints).unsqueeze(0).float(), 22).squeeze().cpu().numpy()
    
    # assert and if not, raise error
    assert (joints.shape[1] == 22) and (joints.shape[2] == 3), f"joints shape is {joints.shape}"

    kinematic_tree = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
    title_sp = title.split(' ')
    if len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])
    
    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        fig.suptitle(title, fontsize=fontsize)
        ax.grid(visible=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        # Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    data = joints.copy().reshape(len(joints), -1, 3)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')  # Updated to work with modern matplotlib
    init()

    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',  
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred','darkred','darkred','darkred']
    
    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]
    
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):
        # Instead of using ax.lines or ax.collections, we clear the axes using ax.cla()
        ax.cla()  # Clears the current axes
        
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        ax.view_init(elev=120, azim=-90) # Reset view
        ax.dist = 7.5
        
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1], MAXS[2] - trajec[index, 1])

        if index > 1:
            ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]), trajec[:index, 1] - trajec[index, 1], linewidth=1.0, color='blue')
        
        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            linewidth = 4.0 if i < 5 else 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth, color=color)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000/fps, repeat=False)

    try:
        ani.save(save_path, writer=PillowWriter(fps=fps))
    except:
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=fps)
        ani.save(save_path, writer=writer)
    
    plt.close()

def get_visualize_data(data_root, sample_num=10, batch=False):
    '''
    load visualize data from npz file
    '''
    visualize_data_pth = data_root + '/visualize_sample_data.npz'
    data = np.load(visualize_data_pth, allow_pickle=True)
    sample_names = data.files[:sample_num]
    sample_data = {name: data[name].item() for name in sample_names}
    
    if batch:
        max_len = max([data[name].item()['m_length'] for name in sample_names])
        caption_batch = []
        motion_batch = []
        lengths_batch = []
        for name in sample_names:
            caption_batch.append(data[name].item()['caption'])
            padded_motion = np.pad(data[name].item()['motion'], ((0, max_len - data[name].item()['m_length']), (0, 0)), mode='constant')
            motion_batch.append(padded_motion)
            lengths_batch.append(data[name].item()['m_length'])
        motion_batch = np.stack(motion_batch)
        lengths_batch = np.array(lengths_batch)
            
        return caption_batch, motion_batch, lengths_batch
    
    return sample_data


import torch.nn.functional as F
import wandb
import clip
import torch

def log_samples_wandb(model, vae, std, mean, animation_dir, data_root_path, quant_factor, cfg=5.0, cfg_schedule='linear'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    vae = vae.to(device)
    vae.eval()
    visualize_dict = get_visualize_data(data_root_path)
    gif_paths = []
    for name, data in visualize_dict.items():
        save_path = os.path.join(animation_dir, name + '.gif')
        texts = clip.tokenize(data['caption'], truncate=True).to(device) # (1, len)
        # generate 1 token at a time
        predicted_latents = model.generate(torch.tensor([data['m_length']//(2**quant_factor)]).to(device), texts = texts, num_iter=data['m_length']//((2**quant_factor)), cfg=cfg, cfg_schedule=cfg_schedule, temperature=1.0, progress=False)
        predicted_latents = F.pad(predicted_latents, ((0,0,0, (2**quant_factor) - predicted_latents.shape[1] %(2**quant_factor),0,0)), mode='constant', value=0)
        predicted_motions = vae.decode(predicted_latents) # bs, seq_len, embdim
        predicted_motions = predicted_motions.cpu().detach().numpy() * std + mean
        plot_3d_motion(save_path, predicted_motions[0, :data['m_length']], data['caption'], figsize=(4,4), fps=20)
        gif_paths.append(save_path)
    wandb.log({"videos": [wandb.Video(data_or_path=path, caption=name, fps=20) for path, name in zip(gif_paths, visualize_dict.keys())]})
    
# def visualize_and_log(data_root, animation_dir, sample_nusm=10)
#     sample_data_dict = get_visualize_data(data_root)
#     gif_paths = []

#     # Generate and save GIFs
#     for name, data in sample_data_dict.items():
#         save_path = os.path.join(animation_dir, name + '.gif')
#         visualize.plot_3d_motion(save_path, data['motion'], data['caption'], figsize=(4,4), fps=20)
#         gif_paths.append(save_path)

#     wandb.log({"videos": [wandb.Video(data_or_path=path, caption=name, fps=20) for path, name in zip(gif_paths, sample_data_dict.keys())]})

        # video_save_dir = os.path.join(f"{output_dir}/samples", f"sample_epoch_{epoch}_step_{step}.gif")
        # export_to_gif(output, video_save_dir)
        # if use_wandb :
        #     wandb.log({"video": wandb.Video(data_or_path=video_save_dir, caption=f'step_{step}', fps=10)})
