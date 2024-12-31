import argparse
import os
import sys
from os.path import join as pjoin

import numpy as np
import torch
from configs import config_utils
from datasets.t2m_dataset import Text2MotionDatasetEval, collate_fn
from evaluation import eval_t2m
from evaluation.get_opt import get_opt
from evaluation.t2m_eval_wrapper import EvaluatorModelWrapper
from MotionPriors import MotionPrior
from MotionPriors.models import TemporalVAE, TransformerVAE
from MotionPriors.models.vq.model import RVQVAE
from utils import visualize
from utils.utils import generate_date_time, seed_everything
from utils.word_vectorizer import WordVectorizer


def main(args):
    # # Load Configs
    data_cfg = config_utils.get_yaml_config(args.data_cfg_path)
    model_cfg = config_utils.get_yaml_config(args.model_cfg_path)
    # # setup seed, device, wandb
    # seed_everything(model_cfg.utils.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Data
    if args.train_data == "t2m":
        data_cfg = data_cfg.t2m
    elif args.train_data == "kit":
        data_cfg = data_cfg.kit 
    
    # load the mean an std of the dataset that was used to train the model

    meta_dir = os.path.dirname(os.path.dirname(args.model_ckpt_path)) + '/meta' 
    mean = np.load(meta_dir +'/mean.npy') 
    std = np.load(meta_dir +'/std.npy')

    print(f"Loading model from {args.model_ckpt_path}")
    net = MotionPrior.MotionPriorWrapper(model_cfg, args.model_ckpt_path, device)
    if net.model.__class__.__name__ in ["Flow", "RectifiedFlowDecoder"]:
        net.set_vqvae()
        net.model.noise_std = args.decoder_noise
    # net = MotionPrior.load_MotionPrior(args.model_ckpt_path, model_cfg)
    net.eval()
    net.to(device)
    quant_factor = net.quant_factor
    sample_data_dict = visualize.get_visualize_data(data_cfg.root_dir)
    model_root_dir = os.path.dirname(os.path.dirname(args.model_ckpt_path))
    model_epoch = os.path.basename(args.model_ckpt_path).split("_")[1]
    animation_dir = os.path.join(model_root_dir, f"animations_{model_epoch}_{generate_date_time()}")
    vae_animation_dir = os.path.join(model_root_dir, "vae_animations")
    save_dir = os.path.join(model_root_dir, f"motion_params_{model_epoch}_{generate_date_time()}")
        
    if net.model.__class__.__name__ == "Flow":
        if args.decoder_noise == 0:
            net.deterministic = True
        else:
            net.deterministic = False
        save_dir = save_dir + "_deterministic" if net.deterministic else save_dir + "_stochastic"
        animation_dir = animation_dir + "_deterministic" if net.deterministic else animation_dir + "_stochastic"
        save_dir = save_dir + "_noise" + str(args.decoder_noise)
        animation_dir = animation_dir + "_noise" + str(args.decoder_noise)
        
    if net.model.__class__.__name__ == "MMM_DiT":
        animation_dir = animation_dir + "_cfg" + str(args.cfg)
    if net.model.__class__.__name__ in ["Flow", "RectifiedFlowDecoder"] and args.num_sample_steps:
        net.num_sample_steps = args.num_sample_steps
        print(f"Setting num_sample_steps to {args.num_sample_steps}")
        animation_dir = animation_dir + f"_num_sample_steps_{args.num_sample_steps}"
        save_dir = save_dir + f"_num_sample_steps_{args.num_sample_steps}"
        
    os.makedirs(animation_dir, exist_ok=True)
    os.makedirs(vae_animation_dir, exist_ok=True)
    if args.save_motion:
        os.makedirs(save_dir, exist_ok=True)
        
    for name, data in sample_data_dict.items():
        save_path = os.path.join(animation_dir, name + '.gif')
        vae_save_path = os.path.join(vae_animation_dir, name + '.gif')
        model_param_save_path = os.path.join(save_dir, name + '.npy')
        # texts = clip.tokenize(data['caption'], truncate=True).to(device)
        # predicted_latents = model.generate(torch.tensor([data['m_length']//4]).to(device), texts = texts, num_iter=10, cfg=1.0, cfg_schedule="linear", temperature=1.0, progress=True)
        # predicted_motions = vae.decode(predicted_latents)
        # predicted_motions = predicted_motions.cpu().detach().numpy()
        data['motion'] = (data['motion'] - mean) / std
        # pad data['motion'] to be divisible by 4
        if "text_condition" in model_cfg.model.keys() and model_cfg.model.text_condition:
            text_embedding = net.model.net.encode_text(data['caption']) # wrapper.flow.net
        else:
            text_embedding = None
        data['motion'] = np.pad(data['motion'], ((0, (2**quant_factor) - len(data['motion']) % (2**quant_factor)), (0, 0)), mode='constant')
        data['motion']  = data['motion'][:196,:]
        input_motion = torch.tensor(data['motion']).unsqueeze(0).to(device).float()
        # if "encoder_noise" in model_cfg.model.keys() and model_cfg.model.encoder_noise:
        #     # this is for encoder_noise in flow models
        #     input_motion = input_motion + torch.randn_like(input_motion) * model_cfg.model.encoder_noise
        if args.encoder_noise: # this is to just add noise to the input motion so that we can see the effect of the noise
            input_motion = input_motion + torch.randn_like(input_motion) * args.encoder_noise
            
        pred_pose_eval, others= net(input_motion, cfg_scale=args.cfg, text_embedding=text_embedding)
        pred_pose_eval = pred_pose_eval.cpu().detach().numpy()
        if args.save_motion:
            np.save(model_param_save_path, pred_pose_eval[0]* std + mean)
        # if vae_animation dir does not exist
        if not os.path.exists(vae_animation_dir) and net.model.__class__.__name__ == "MMM_DiT" or net.model.__class__.__name__ == "Flow":
            if isinstance(others, tuple):
                others = others[0]
            others = others.cpu().detach().numpy()
            others =   others* std + mean
            visualize.plot_3d_motion(vae_save_path, others[0] , data['caption'], figsize=(4,4), fps=20)
        pred_pose_eval = pred_pose_eval * std + mean
        visualize.plot_3d_motion(save_path, pred_pose_eval[0] , data['caption'], figsize=(4,4), fps=20)
               
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # use the paths used to train the model
    parser.add_argument("--data_cfg_path", type=str, default="./configs/t2mdataset.yaml")
    parser.add_argument("--model_cfg_path", type=str, default="./configs/temporalVAE.yaml")
    parser.add_argument("--model_ckpt_path", type=str)
    parser.add_argument("--train_data", type=str, default="t2m", choices=["t2m", "kit"])
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--num_sample_steps", type=int, default=16)
    parser.add_argument("--decoder_noise", type=float, default=0) # if 0, deterministic
    parser.add_argument("--encoder_noise", type=float, default=0)
    parser.add_argument("--save_motion", action="store_true")
    args = parser.parse_args()
    main(args)

