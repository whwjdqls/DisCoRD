from __future__ import annotations

import math
from copy import deepcopy
from functools import partial
from math import sqrt
from typing import Literal

import einx
import numpy as np
import torch
import torch.nn.functional as F
from denoising_diffusion_pytorch import Unet1D
from einops import pack, rearrange, reduce, repeat, unpack
from einops.layers.torch import Rearrange

# from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncodingPermute1D
from torch import nn, pi
from torch.nn import Module, ModuleList
from torchdiffeq import odeint
from tqdm import tqdm

from .helpers import *


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device, dtype=tensor.dtype)
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc
    
class Flow(Module):
    def __init__(
        self,
        dim: int,
        net: MLPAdaLN,
        *,
        atol = 1e-5,
        rtol = 1e-5,
        noise_std = 0.1,
        method = 'midpoint',
        use_diffusion_pos_embed = True
    ):
        super().__init__()
        self.net = net
        self.dim = dim
        self.noise_std = noise_std
        self.use_diffusion_pos_embed = use_diffusion_pos_embed
        self.mlp = net.__class__.__name__ == 'MLPAdaLN'
        # if set to True, we have to use diffusion_pos_embed, and (BS*seq_len, dim) input
        # if set to False, we don't use diffusion_pos_embed, and (BS, seq_len, dim) input
        # assert self.mlp == use_diffusion_pos_embed, "If using MLP, must use diffusion_pos_embed"
        if use_diffusion_pos_embed:
            self.pos_embed_diffusion = PositionalEncoding1D(dim)

        self.odeint_kwargs = dict(
            atol = atol,
            rtol = rtol,
            method = method
        )

    @property
    def device(self):
        return next(self.net.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        cond = None,
        bsz = 1,
        length = 196,
        num_sample_steps = 16,
        deterministic = False,
        z=None,
        padding_mask = None,
        text_embedding = None
    ):
        if cond is not None:
            bsz= cond.shape[0]
            if deterministic:
                noise = cond
            else:
                noise = cond + torch.randn_like(cond) * self.noise_std
            if self.mlp:
                noise = rearrange(noise, 'b n c -> (b n) c')
        else:
            if self.mlp:
                sampled_data_shape = (bsz*length, self.dim)
            else:
                sampled_data_shape = (bsz, length, self.dim) 
            noise = torch.randn(sampled_data_shape, device = self.device)
        # time steps

        times = torch.linspace(0., 1., num_sample_steps, device = self.device)
        
            
        if self.use_diffusion_pos_embed: 
            pos = self.pos_embed_diffusion(torch.zeros(bsz, length, self.dim, device = self.device))
            pos = rearrange(pos, 'b n c -> (b n) c')
            noise = noise + pos
            
            
        def ode_fn(t, x):
            if self.mlp:
                t = repeat(t, '-> b', b = bsz*length)
            else:
                t = repeat(t, '-> b', b = bsz)
            # if (padding_mask is not None) and (not self.mlp): # for DiT
            #     flow = self.net(x, times = t, padding_mask = padding_mask, text_embedding=text_embedding)
            if not self.mlp: # for Dit
                flow = self.net(x, times = t, z=z, padding_mask = padding_mask, text_embedding=text_embedding)
            else: # for MLP_
                flow = self.net(x, times = t) # do we need padding makes here? how to implement?
            return flow

        trajectory = odeint(ode_fn, noise, times, **self.odeint_kwargs)
        sampled = trajectory[-1]
        if self.mlp:
            sampled = rearrange(sampled, '(b n) c -> b n c', b = bsz)
        return sampled

    def forward(self, *args, type="forward", **kwargs):
        if type == "forward":
            return self.forward_forward(*args, **kwargs)
        elif type == "sample":
            return self.sample(*args, **kwargs)
        else:
            raise ValueError(f"Invalid type {type}")
    
    # training
    def forward_forward(self, gt, noisy_input, z=None,padding_mask=None, text_embedding=None):
        batch_size, seq_len, dim = gt.shape
        device = self.device
        assert dim == self.dim, f'dimension of sequence being passed in must be {self.dim} but received {dim}'
        
        if self.mlp:
            gt = rearrange(gt, 'b n c -> (b n) c')
            noisy_input = rearrange(noisy_input, 'b n c -> (b n) c')
            
        times = torch.rand(gt.shape[0], device = device)
        padded_times = right_pad_dims_to(gt, times)
        
        source_dist_samples = noisy_input + torch.randn_like(noisy_input) * self.noise_std
        
        x_t = (1.- padded_times) * source_dist_samples + padded_times * gt
        
        if self.use_diffusion_pos_embed: 
            pos = self.pos_embed_diffusion(torch.zeros(batch_size, seq_len, dim, device = device))
            pos = pos.reshape(-1, dim)
            x_t = x_t + pos
    

        if (padding_mask is not None) and (not self.mlp):
            v_t  = self.net(x_t, times = times, z=z,padding_mask = padding_mask, text_embedding=text_embedding)
        else:
            v_t  = self.net(x_t, times = times, z=z)
        
        # print("v_t",v_t)
        # print("v_t",v_t)
        flow = gt - source_dist_samples

        if (padding_mask is not None) and (not self.mlp):
            # print(padding_mask)
            # print("v_t.mean()",v_t.mean())
            # print("flow.mean()",flow.mean())
            loss = F.mse_loss(v_t, flow, reduction = 'none')
            # print("loss",loss.shape)
            # print("loss.sum()",loss.sum())
            mask = ~padding_mask
            mask = mask.unsqueeze(-1)
            # print("mask",mask.shape)
            mask = mask.float()
            masked_loss = (loss * mask).mean(dim=-1).sum()/ (mask.sum() + 1e-8)
            return masked_loss
        elif (padding_mask is not None) and (self.mlp):
            loss = F.mse_loss(v_t, flow, reduction = 'none')
            padding_mask = rearrange(padding_mask, 'b n -> (b n)')
            mask = ~padding_mask
            mask = mask.unsqueeze(-1).repeat(1, self.dim)
            loss = (loss * mask).sum() / mask.sum()
            return loss
        else:
            return F.mse_loss(v_t, flow)
        
class Reflow(Module):
    def __init__(
        self,
        rectified_flow: Flow,
        frozen_model: Flow | None = None,
        *,
        batch_size = 16,
        length = 196
    ):
        super().__init__()
        self.model = rectified_flow
        if not exists(frozen_model):
            # make a frozen copy of the model and set requires grad to be False for all parameters for safe measure
            frozen_model = deepcopy(rectified_flow)

            for p in frozen_model.parameters():
                p.detach_()

        self.frozen_model = frozen_model

    def device(self):
        return next(self.parameters()).device

    def parameters(self):
        return self.model.parameters() # omit frozen model

    def sample(self, *args, **kwargs):
        return self.model.sample(*args, **kwargs)

    def forward(self, noisy_input, padding_mask = None, num_sample_steps = 16):

        sampled_output = self.frozen_model.sample(cond=noisy_input, num_sample_steps = num_sample_steps)

        # the coupling in the paper is (noise, sampled_output)
        loss = self.model(gt=sampled_output, noisy_input = noisy_input, padding_mask=padding_mask)

        return loss


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

# if __name__ == '__main__':
#     import copy
#     import os

#     import numpy as np
#     import torch.nn.functional as F
#     from configs import config_utils
#     from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
#     from datasets import t2m_dataset
#     from utils import visualize
#     from utils.utils import generate_date_time, seed_everything

#     from MotionPriors import MotionPrior
    
#     data_cfg = config_utils.get_yaml_config("./configs/t2mdataset.yaml")
#     vae_root = "/scratch2/jbc/MotionPriors/checkpoints/t2m_STVQVAE_qf2_sqf2_hd2d128_hd1d64_ebd32_nemb8192_fd256_nl0_nh4l2_jw0.5_cw0.02_bsz1024_1009125852"
#     vae_ckpt = "/scratch2/jbc/MotionPriors/checkpoints/t2m_STVQVAE_qf2_sqf2_hd2d128_hd1d64_ebd32_nemb8192_fd256_nl0_nh4l2_jw0.5_cw0.02_bsz1024_1009125852/checkpoints/STVQVAE_154_0.020333_fid0.01185_mpjpe0.03378.pth"
#     vae_cfg_path = os.path.join(vae_root, "configs/config_model.yaml")
#     vae_meta_path = os.path.join(vae_root, "meta")
#     vae_cfg = config_utils.get_yaml_config(vae_cfg_path)
#     seed_everything(42)
#     data_cfg = data_cfg.t2m
#     mean = np.load(os.path.join(vae_meta_path, "mean.npy"))
#     std = np.load(os.path.join(vae_meta_path, "std.npy"))
    
#     #####################################
    
#     debug = False
#     full_motion = True
#     use_diffusion_pos_embed = False
#     net = "mlp"
#     stage = "reflow"
    
#     #######################################
#     if debug:
#         import pdb
#         pdb.set_trace()
#     # simple test
#     mlp_depth = 6
#     mlp_width = 1024
#     dim_input = 263
    
#     mlp = MLPAdaLN(
#         dim_input = dim_input,
#         depth = mlp_depth,
#         width = mlp_width,
#     )
    
#     unet = Unet1D( # need to handle paddings. so currently use with full motion = False
#         dim = 384,
#         dim_mults = (1, 1),
#         channels = dim_input
#     )
    
#     denoiser = unet if net == "unet" else mlp
    
#     flow = Flow(
#         dim  = dim_input,
#         net = denoiser,
#         use_diffusion_pos_embed=use_diffusion_pos_embed
#     )
        
#     vqvae = MotionPrior.load_MotionPrior(vae_ckpt, vae_cfg)
#     vqvae.eval()
#     vqvae.cuda()
    
#     for param in vqvae.parameters():
#         param.requires_grad = False
    
#     flow.load_state_dict(torch.load("/home/jw1510/MARM/checkpoints/RF_297_8.501201216131449.pth")) # MLP + NO Pos
#     flow = flow.cuda()
    
#     if debug:
#         B = 2
#         S = 196
        
#         random_input = torch.randn(B, S, dim_input)
#         random_gt = torch.randn(B, S, dim_input)
        
#         if net == "unet":
#             random_input = rearrange(random_input, 'b n c -> b c n').cuda()
#             random_gt = rearrange(random_gt, 'b n c -> b c n').cuda()
#         else:
#             random_input = random_input.reshape(-1, dim_input).cuda()
#             random_gt = random_gt.reshape(-1, dim_input).cuda()
        
#         loss = flow(random_gt, noisy_input = random_input)
#         sample = flow.sample(cond=random_input, num_sample_steps = 16)
#         # sample_from_scratch = flow.sample(bsz=B, length=S, num_sample_steps = 16)
        
#         print(loss)
    
    #     sample_data_dict = visualize.get_visualize_data("/datasets2/human_motion/HumanML3D")
    #     animation_dir = './gifs/rectified_flow'
    #     os.makedirs(animation_dir, exist_ok=True)
        
    #     flow.eval()
    #     with torch.no_grad():
    #         for name, data in tqdm(sample_data_dict.items()):
    #             motion =(data['motion'] - mean)/ std
    #             m_length = min(data['m_length'], S)
    #             motion = np.pad(motion, ((0, 4 - len(motion) % 4), (0, 0)), mode='constant')
    #             motion = torch.tensor(motion).unsqueeze(0).float().cuda()
    #             motion = motion[:, :S]
    #             pred_motion, _, _ = vqvae(motion)
                
    #             flow_refined = flow.sample(cond=pred_motion.reshape(-1, dim_input), num_sample_steps = 16)
    #             flow_refined = rearrange(flow_refined, '(b n) c -> b n c', b=1)
    #             pred_motion = pred_motion.cpu().detach().numpy()
    #             flow_refined = flow_refined.cpu().detach().numpy()
    #             pred_motion = pred_motion * std + mean
    #             flow_refined = flow_refined * std + mean
    #             save_path_vae = os.path.join(animation_dir, f'{name}_vae.gif')
    #             save_path_flow = os.path.join(animation_dir, f'{name}_flow.gif')
    #             visualize.plot_3d_motion(save_path_vae, pred_motion[0][:m_length], data['caption'], figsize=(4,4), fps=20)
    #             visualize.plot_3d_motion(save_path_flow, flow_refined[0][:m_length], data['caption'], figsize=(4,4), fps=20)
        
    #     pdb.set_trace()
    #     train_dataset = t2m_dataset.MotionDataset(data_cfg, mean, std, split='val')
    #     train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, drop_last=True, shuffle=False, pin_memory=True)
    # else:
    #     if not full_motion:
    #         train_dataset = t2m_dataset.MotionDataset(data_cfg, mean, std, split='train')
    #         train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, drop_last=True, shuffle=True, pin_memory=True)
    #     else:
    #         train_dataset = t2m_dataset.Text2MotionDataset(data_cfg, mean, std, split='train')
    #         train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, drop_last=True, shuffle=True, pin_memory=True)
    
    #     import wandb
    #     exp_name = f"RF_{net}_diffpos_{use_diffusion_pos_embed}_fulllength_{full_motion}_{stage}"
    #     wandb.init(project=f'RF', name=exp_name)
        
    
    # if stage != "reflow":
    #     flow.train()
    #     optimizer = torch.optim.AdamW(flow.parameters(), lr=0.0002, weight_decay=1e-2, betas=(0.9, 0.95))
    #     scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=300, max_lr=0.0002, min_lr=0.00001, warmup_steps=50)
    #     optimizer.zero_grad()
    #     best_loss = float('inf')
    #     previous_best_model_path = None
    #     os.makedirs(f"./checkpoints/{exp_name}", exist_ok=True)
    #     model_params = list(flow.parameters())
    #     ema_params = copy.deepcopy(model_params)
    #     for epoch in range(300):
    #         print(f'Current Epoch: {epoch}/300')
    #         loss_epoch = 0
    #         total_steps = len(train_dataloader)
            
    #         for idx, data in enumerate(tqdm(train_dataloader)):
    #             if full_motion:
    #                 _, gt, _ = data
    #                 gt = gt.float().cuda()
    #             else:
    #                 gt = data.float().cuda()
                
    #             noisy_input, _, _ = vqvae(gt)
    #             if net  == "unet":
    #                 noisy_input = rearrange(noisy_input, 'b n c -> b c n')
    #                 gt = rearrange(gt, 'b n c -> b c n')
    #                 padding_mask = None # must not use full motion
    #             else:
    #                 padding_mask = (gt.sum(dim=-1) == 0).to(gt.dtype).cuda().bool()
    #                 padding_mask = padding_mask.reshape(-1)
    #                 gt = gt.reshape(-1, dim_input)
    #                 noisy_input = noisy_input.reshape(-1, dim_input)
                    
                
    #             loss = flow(gt, noisy_input = noisy_input, padding_mask=padding_mask)
    #             loss.backward()
    #             optimizer.step()
    #             optimizer.zero_grad()
    #             update_ema(ema_params, model_params, 0.9999)
                
    #             if not debug:
    #                 wandb.log({"loss": loss.item()})
    #             # print(f'Current Loss: {loss.item()}')
    #             loss_epoch += loss.item()
            
    #         if not debug:
    #             wandb.log({"loss_epoch": loss_epoch / total_steps})
    #             wandb.log({"epoch": epoch})
    #             wandb.log({"lr": optimizer.param_groups[0]['lr']})
                
    #         print(f'Epoch Loss: {loss_epoch / total_steps}')
    #         scheduler.step()
                
    #         if (loss_epoch/total_steps) < best_loss:
    #             best_loss = (loss_epoch/total_steps)
    #             if previous_best_model_path is not None:
    #                 os.remove(previous_best_model_path)
    #             previous_best_model_path = f"./checkpoints/{exp_name}/RF_{epoch}_{best_loss}.pth"
    #             torch.save(flow.state_dict(), previous_best_model_path)
    # else:
    #     flow.train()
    #     reflow = Reflow(flow)
    #     optimizer = torch.optim.AdamW(flow.parameters(), lr=0.0002, weight_decay=1e-2, betas=(0.9, 0.95))
    #     scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=300, max_lr=0.0002, min_lr=0.00001, warmup_steps=50)
    #     optimizer.zero_grad()
    #     best_loss = float('inf')
    #     previous_best_model_path = None
    #     os.makedirs(f"./checkpoints/{exp_name}_reflow", exist_ok=True)
        
    #     model_params = list(flow.parameters())
    #     ema_params = copy.deepcopy(model_params)
    #     for epoch in range(300):
    #         print(f'Current Epoch: {epoch}/300')
    #         loss_epoch = 0
    #         total_steps = len(train_dataloader)
            
    #         for idx, data in enumerate(tqdm(train_dataloader)):
    #             if full_motion:
    #                 _, gt, _ = data
    #                 gt = gt.float().cuda()
    #             else:
    #                 gt = data.float().cuda()
                
    #             noisy_input, _, _ = vqvae(gt)
    #             if net  == "unet":
    #                 noisy_input = rearrange(noisy_input, 'b n c -> b c n')
    #                 gt = rearrange(gt, 'b n c -> b c n')
    #                 padding_mask = None # must not use full motion
    #             else:
    #                 padding_mask = (gt.sum(dim=-1) == 0).to(gt.dtype).cuda().bool()
    #                 padding_mask = padding_mask.reshape(-1)
    #                 gt = gt.reshape(-1, dim_input)
    #                 noisy_input = noisy_input.reshape(-1, dim_input)
                
    #             loss = reflow(noisy_input = noisy_input, padding_mask=padding_mask)
    #             loss.backward()
    #             optimizer.step()
    #             optimizer.zero_grad()
    #             update_ema(ema_params, model_params, 0.9999)
                
    #             if not debug:
    #                 wandb.log({"loss": loss.item()})
    #             # print(f'Current Loss: {loss.item()}')
    #             loss_epoch += loss.item()
            
    #         if not debug:
    #             wandb.log({"loss_epoch": loss_epoch / total_steps})
    #             wandb.log({"epoch": epoch})
    #             wandb.log({"lr": optimizer.param_groups[0]['lr']})
                
    #         print(f'Epoch Loss: {loss_epoch / total_steps}')
    #         scheduler.step()
                
    #         if (loss_epoch/total_steps) < best_loss:
    #             best_loss = (loss_epoch/total_steps)
    #             if previous_best_model_path is not None:
    #                 os.remove(previous_best_model_path)
    #             previous_best_model_path = f"./checkpoints/{exp_name}_reflow/RF_{epoch}_{best_loss}.pth"
    #             torch.save(flow.state_dict(), previous_best_model_path)
    