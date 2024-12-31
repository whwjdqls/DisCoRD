from __future__ import annotations

import argparse
import copy
import math
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from configs import config_utils
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from datasets import t2m_dataset
from datasets.t2m_dataset import Text2MotionDatasetEval, collate_fn, make_rf_decoder_dataset
from denoising_diffusion_pytorch import Unet1D
from einops import pack, rearrange, reduce, repeat, unpack
from einops.layers.torch import Rearrange
from evaluation import eval_t2m
from MotionPriors import MotionPrior
from MotionPriors.models.rf_decoder import get_flow_backbone
from MotionPriors.models.rf_decoder import rectified_flow
from omegaconf import OmegaConf

# from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncodingPermute1D
from torch import nn, pi
from torch.nn import Module, ModuleList
from torchdiffeq import odeint
from tqdm import tqdm
from utils import visualize
from utils.utils import generate_date_time, seed_everything
from utils.word_vectorizer import WordVectorizer

import wandb


def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask  # (b, len)

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
        
def train_one_epoch(config, epoch, flow, vqvae, optimizer, data_loader, device,model_params=None, ema_params=None, use_wandb=False):
    """
    Train the model for one epoch
    """
    flow.train()
    flow.to(device)
    loss_epoch = 0
    total_steps = len(data_loader)
    for i, data in enumerate(data_loader):
        if config.train.full_motion:
            gt, z, m_length, caption, padding_mask = data
            padding_mask = padding_mask.to(device)
            if "text_condition" in config.model.keys() and config.model.text_condition:
                text_embedding = flow.net.encode_text(caption)  # text embedding already on device
            else:
                text_embedding = None
        else:
            gt, z = data
            padding_mask = None
            text_embedding = None
        z = z.to(device) #(b, n, 512
        gt = gt.float().to(device)
        optimizer.zero_grad()
        loss = flow(gt, z, padding_mask=padding_mask, text_embedding=text_embedding)
        loss.backward()
        if config.train.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=config.train.max_grad_norm)
        optimizer.step()

        if ema_params is not None and model_params is not None:
            update_ema(ema_params, model_params, rate=config.train.ema_rate)
            
        if i % config.train.log_step == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    i * len(data),
                    len(data_loader.dataset),
                    100.0 * i / len(data_loader),
                    loss.item(),
                )
            )
        if use_wandb:
            try:
                wandb.log({"train loss (step)": loss.detach().item()})
            except:
                print("W&B logging failed. Continuing training.")
        sys.stdout.flush()
    if use_wandb:
        try:
                wandb.log({"loss_epoch": loss_epoch / total_steps})
                wandb.log({"epoch": epoch})
                wandb.log({"lr": optimizer.param_groups[0]['lr']})
        except:
            print("W&B logging failed. Continuing training.")
    print("Train Epoch: {}\tAverage Loss: {:.6f}".format(epoch, loss_epoch / total_steps))

@torch.no_grad()
def val_one_epoch(config, epoch, flow, vqvae, data_loader, device, use_wandb=False):
    flow.eval()
    flow.to(device)
    loss_epoch = 0
    total_steps = len(data_loader)
    
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if config.train.full_motion:
                gt, z, m_length, caption, padding_mask = data
                padding_mask = padding_mask.to(device)
                if "text_condition" in config.model.keys() and config.model.text_condition:
                    text_embedding = flow.net.encode_text(caption)  # text embedding already on device
                else:
                    text_embedding = None
            else:
                gt, z = data
                padding_mask = None
                text_embedding = None
            z = z.to(device)
            gt = gt.float().to(device)
            loss = flow(gt, z, padding_mask=padding_mask, text_embedding=text_embedding)
            loss_epoch += loss.detach().item()
        avg_loss = loss_epoch / total_steps

        if use_wandb:
            try:
                wandb.log({"val loss": avg_loss})
            except:
                print("W&B logging failed. Continuing training.")
        print("Val Epoch: {}\tAverage Loss: {:.6f}".format(epoch, avg_loss))
    return avg_loss

        
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ############################# Configs #############################
    data_cfg = config_utils.get_yaml_config(args.data_cfg_path)
    model_cfg = config_utils.get_yaml_config(args.model_cfg_path)
    # backbone_name = "MLPAdaLN" if model_cfg.model.MLPAdaLN.use elif "DiT" in model_cfg.model.DiT.use elif "Unet1D" in model_cfg.model.Unet1D.use else 'Transformer'
    # Determine backbone_name based on model configuration
    if model_cfg.model.DiT.use:
        backbone_name = "DiT"
        key_values = model_cfg.model.DiT
    elif model_cfg.model.Unet1D.use:
        backbone_name = "Unet1D"
        key_values = model_cfg.model.Unet1D
    else:
        raise NotImplementedError(f"Model {model_cfg.model.name} not implemented")
    
    exp_name = (
        f"{args.train_data}_newdecoder_{model_cfg.model.name}_fm{model_cfg.train.full_motion}"
        f"_{backbone_name}"
        f"_{generate_date_time()}"
    )
    if "text_condition" in model_cfg.model.keys() and model_cfg.model.text_condition:
        exp_name += "_text_condition"
    # use the first 5 key values and extent exp_name
# Extend exp_name with the first 5 key-value pairs from key_values
    for key, value in key_values.items():
        exp_name += f"_{key[:2]}{value}"

        
    vae_ckpt = model_cfg.model.vqvae_weight_path # pretrained VQVAE
    vae_root = os.path.dirname(os.path.dirname(vae_ckpt))
    vae_cfg_path = os.path.join(vae_root, "configs/config_model.yaml")
    vae_meta_path = os.path.join(vae_root, "meta")
    vae_mean = np.load(os.path.join(vae_meta_path, "mean.npy"))
    vae_std = np.load(os.path.join(vae_meta_path, "std.npy"))
    
    vae_cfg = config_utils.get_yaml_config(vae_cfg_path)
    vqvae = MotionPrior.MotionPriorWrapper(vae_cfg, vae_ckpt, device)
    vqvae.eval()
    vqvae.to(device)
    
    for param in vqvae.parameters():
        param.requires_grad = False
        
    exp_name = vae_cfg.model.name + "_" + exp_name
    if args.wandb:
        wandb.init(project=f"decoderRF", name=exp_name, config=dict(model_cfg)) 
        
    save_path = Path(model_cfg.train.save_dir) / exp_name
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(f'{save_path}/checkpoints', exist_ok=True) # for saving checkpoints
    os.makedirs(f'{save_path}/configs', exist_ok=True) # for saving configs
    os.makedirs(f'{save_path}/meta', exist_ok=True) # for metadata
    with open(f'{save_path}/configs/config_model.yaml','w') as fp:
        OmegaConf.save(config=model_cfg, f=fp.name)
    with open(f'{save_path}/configs/config_data.yaml','w') as fp:
        OmegaConf.save(config=data_cfg, f=fp.name)
        
    ############################# Loading Flow #############################d
    denoiser = get_flow_backbone(model_cfg)
    flow = rectified_flow.RectifiedFlowDecoder(model = denoiser)
    flow.to(device)
    ############################# Dataset #############################
    if args.train_data == "t2m":
        data_cfg = data_cfg.t2m
    elif args.train_data == "kit":
        data_cfg = data_cfg.kit
        
    if not model_cfg.train.full_motion:
        data_cfg.feat_bias = 1.0 # as we want to use the same mean, std as the vae_version
        train_dataset = t2m_dataset.MotionDataset(data_cfg, vae_mean, vae_std, split='train', debug=False)
        val_dataset = t2m_dataset.MotionDataset(data_cfg, vae_mean,vae_std, split='val',debug=False)
    else:
        train_dataset = t2m_dataset.Text2MotionDataset(data_cfg, vae_mean, vae_std, split='train')
        val_dataset = t2m_dataset.Text2MotionDataset(data_cfg, vae_mean, vae_std,  split='val')
        
    refinement_batch_size = 256 if model_cfg.train.full_motion else 2048
    refine_train_dataset = make_rf_decoder_dataset(train_dataset, vqvae, refinement_batch_size)
    refine_val_dataset = make_rf_decoder_dataset(val_dataset, vqvae, refinement_batch_size)
    del train_dataset
    del val_dataset
    assert torch.allclose(torch.tensor(refine_train_dataset.mean), torch.tensor(vae_mean), atol=1e-5), "mean not equal"
    assert torch.allclose(torch.tensor(refine_val_dataset.std), torch.tensor(vae_std), atol=1e-5), "std not equal"
    
    train_dataloader = torch.utils.data.DataLoader(refine_train_dataset, batch_size=model_cfg.train.batch_size, drop_last=True, shuffle=True, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(refine_val_dataset, batch_size=model_cfg.val.batch_size, drop_last=True, shuffle=False, pin_memory=True)
    
    np.save(f'{save_path}/meta/mean.npy', vae_mean) # 
    np.save(f'{save_path}/meta/std.npy', vae_std)
    
    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    test_dataset = Text2MotionDatasetEval(data_cfg,vae_mean, vae_std, w_vectorizer, split='val', debug=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, drop_last=True, collate_fn=collate_fn,shuffle=True, pin_memory=True)
    
    ############################# Train Setup #############################
    optimizer = torch.optim.AdamW(flow.parameters(), lr=model_cfg.train.lr)
    if model_cfg.train.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=model_cfg.train.num_epochs)
    elif model_cfg.train.scheduler == "cosine_warmup":
        scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=model_cfg.train.num_epochs, max_lr=model_cfg.train.lr, min_lr=model_cfg.train.min_lr, warmup_steps=min(model_cfg.train.num_epochs//10, 50))
    else:
        scheduler = None
        
    if model_cfg.train.ema_rate:
        model_params = list(flow.parameters())
        ema_params = copy.deepcopy(model_params)
    else:
        model_params = None
        ema_params = None
        
    best_loss = float("inf")
    previous_best_model_path = None
    top_models = []  # List to keep track of top 5 models based on FID
    ############################# Train #############################
    for epoch in range(0, model_cfg.train.num_epochs):
        print("epoch", epoch, "num_epochs", model_cfg.train.num_epochs)
        train_one_epoch(model_cfg, epoch, flow,vqvae, optimizer, train_dataloader, device=device, model_params=model_params, ema_params=ema_params, use_wandb=args.wandb)
        val_loss = val_one_epoch(model_cfg, epoch, flow,vqvae, val_dataloader,device, use_wandb=args.wandb)
        if scheduler:
            scheduler.step()
        # saving models
        print("-" * 50)
        if val_loss < best_loss:
            best_loss = val_loss
            # Delete the previous best model if it exists
            if previous_best_model_path and os.path.exists(previous_best_model_path):
                os.remove(previous_best_model_path)
            # Save the new best model with loss in the filename
            best_model_path = f"{save_path}/checkpoints/{denoiser.__class__.__name__}_best_{epoch}_{val_loss:.6f}.pth"
            if model_cfg.model.Reflow: # we need to save the flow part only
                torch.save(flow.model.state_dict(), best_model_path)
            else:
                torch.save(flow.state_dict(), best_model_path)
            previous_best_model_path = best_model_path
            print(f"Saved best model at epoch {epoch} with loss {val_loss:.6f}\n")

        if (epoch != 0) and (epoch % model_cfg.train.save_every == 0):
            metric_dict = eval_t2m.evaluate_motion_prior(test_dataloader, flow, model_cfg, device,vqvae=vqvae,train_data=args.train_data, repeat_time=1)
            print("Evaluation results: ", metric_dict)
            if args.wandb:
                try:
                    wandb.log(metric_dict)
                except:
                    print("W&B logging failed. Continuing training.")
            FID = metric_dict["FID"]
            MPJPE = metric_dict["MAE"]  
            if len(top_models) < 20 or FID < max(top_models, key=lambda x: x['FID'])['FID']:
                # Save the current model
                save_model_path = f"{save_path}/checkpoints/{model_cfg.model.name}_{epoch}_{val_loss:.6f}_fid{FID:.5f}_mpjpe{MPJPE:.5f}.pth"
                if model_cfg.model.Reflow:  # we need to save the flow part only
                    torch.save(flow.model.state_dict(), save_model_path)
                else:
                    torch.save(flow.state_dict(), save_model_path)
                print(f"Saved model at epoch {epoch} with FID {FID:.5f}\n")

                # Add the current model to the top models list
                top_models.append({'FID': FID, 'path': save_model_path})

                # If we have more than 5 models, remove the one with the worst FID
                if len(top_models) > 40:# Find the model with the worst FID
                    worst_model = max(top_models, key=lambda x: x['FID'])# Remove it from the list
                    top_models.remove(worst_model)# Delete the model file
                    if os.path.exists(worst_model['path']):
                        os.remove(worst_model['path'])
                        print(f"Removed model with worst FID: {worst_model['path']}")
            else:
                print(f"Model at epoch {epoch} not in top 5 FID ({FID:.5f}), not saving.\n")   
                 

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg_path", type=str, default="./configs/motiondataset_w_vqvae.yaml")
    parser.add_argument("--model_cfg_path", type=str, default="./configs/MMM_DiT.yaml")
    parser.add_argument("--train_data", type=str, default="t2m", choices=["t2m", "kit"])
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    main(args)
