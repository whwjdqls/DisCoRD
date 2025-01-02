import os
from os.path import join as pjoin
import torch
import numpy as np
import argparse
from MotionGen.momask_transformer.transformer import MaskTransformer, ResidualTransformer
from MotionPriors import MotionPrior
from evaluation.get_opt import get_opt
from utils.utils import generate_date_time, seed_everything
from MotionPriors.models.vq.model import RVQVAE
from configs import config_utils
from typing import Literal
from utils import visualize
from evaluation.metrics import *

# python eval_Momask.py --time_steps 10 --ext evaluation
def load_vq_model(vq_opt):
    vq_model = RVQVAE(vq_opt,
            263,
            vq_opt.nb_code,
            vq_opt.code_dim,
            vq_opt.output_emb_width,
            vq_opt.down_t,
            vq_opt.stride_t,
            vq_opt.width,
            vq_opt.depth,
            vq_opt.dilation_growth_rate,
            vq_opt.vq_act,
            vq_opt.vq_norm
    )
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'net_best_fid.tar'),
                            map_location=opt.device)
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {vq_opt.name} Completed!')
    return vq_model, vq_opt

def load_trans_model(model_opt, which_model):
    t2m_transformer = MaskTransformer(code_dim=model_opt.code_dim,
            cond_mode='text',
            latent_dim=model_opt.latent_dim,
            ff_size=model_opt.ff_size,
            num_layers=model_opt.n_layers,
            num_heads=model_opt.n_heads,
            dropout=model_opt.dropout,
            clip_dim=512,
            cond_drop_prob=model_opt.cond_drop_prob,
            clip_version=clip_version,
            opt=model_opt,
    )
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model),
                        map_location=device)
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Mask Transformer {opt.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer

def load_res_model(res_opt):
    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens = vq_opt.nb_code
    res_transformer = ResidualTransformer(code_dim=vq_opt.code_dim,
            cond_mode='text',
            latent_dim=res_opt.latent_dim,
            ff_size=res_opt.ff_size,
            num_layers=res_opt.n_layers,
            num_heads=res_opt.n_heads,
            dropout=res_opt.dropout,
            clip_dim=512,
            shared_codebook=vq_opt.shared_codebook,
            cond_drop_prob=res_opt.cond_drop_prob,
            share_weight=res_opt.share_weight,
            clip_version=clip_version,
            opt=res_opt
    )

    ckpt = torch.load(pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model', 'net_best_fid.tar'),
                        map_location=device)
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer'], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Residual Transformer {res_opt.name} from epoch {ckpt["ep"]}!')
    return res_transformer

def visualize_motion(
                    clip_text,
                    m_length,
                    vq_model,
                    res_model,
                    trans,
                    time_steps,
                    cond_scale,
                    temperature,
                    topkr,
                    mean, 
                    std,
                    rf_model=None,
                    force_mask=False,
                    res_cond_scale=5,
                    save_dir='./gifs',
):
    
    trans.eval()
    vq_model.eval()
    res_model.eval()
    
    save_path = os.path.join(save_dir, f'{clip_text}.gif')
    clip_text = [clip_text]
    m_length = torch.tensor([m_length,]).cuda()
    bs = m_length.shape[0]

    mids = trans.generate(
        clip_text,
        m_length // 4,
        time_steps,
        cond_scale,
        temperature=temperature,
        topk_filter_thres=topkr,
        force_mask=force_mask,
    )
    
    pred_ids = res_model.generate(mids, clip_text, m_length // 4, temperature=1, cond_scale=res_cond_scale)
    pred_motions = vq_model.forward_decoder(pred_ids)
    if pred_motions.shape[1] != 196:
        print('pred_motions shape', pred_motions.shape)
        print('pred_ids shape', pred_ids.shape)
        print('m_length', m_length)
        pred_ids = torch.cat([pred_ids, -torch.ones(pred_ids.shape[0], (196//4) - pred_ids.shape[1], pred_ids.shape[2], dtype=pred_ids.dtype).cuda()], dim=1)
        pred_motions = torch.cat([pred_motions, torch.zeros(pred_motions.shape[0], 196 - pred_motions.shape[1], pred_motions.shape[2]).cuda()], dim=1)

    text_embedding = None
    pred_motions = rf_model.decode_with_RF(pred_ids, text_embedding=text_embedding, m_length=None)
    pred_motions = pred_motions.detach().cpu().numpy()
    pred_recon = pred_motions * std + mean # inverse transform
    visualize.plot_3d_motion(save_path, pred_recon[0, :m_length[0].item()], clip_text[0], figsize=(4, 4), fps=30)
    
    print(f"Saved gif to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_cfg_pth', type=str, default='./configs/momask_trans_eval_config_t2m.yaml')
    parser.add_argument("--data_cfg_path", type=str, default="./configs/config_data.yaml")
    parser.add_argument("--model_cfg_path", type=str, default="./configs/config_model.yaml")
    parser.add_argument("--train_data", type=str, default="t2m", choices=["t2m", "kit"])
    parser.add_argument("--num_sample_steps", type=int, default=16)
    parser.add_argument("--model_ckpt_path", type=str, default=None, help="if None, we are evaluating the original momask model, else we are evaluating refinement model")
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--input_text", type=str, default="A person is walking")
    parser.add_argument("--m_length", type=int, default=196)
    args = parser.parse_args()
    
    eval_cfg = config_utils.get_yaml_config(args.eval_cfg_pth)
    data_cfg = config_utils.get_yaml_config(args.data_cfg_path)
    model_cfg = config_utils.get_yaml_config(args.model_cfg_path)
    
    model_cfg.model.vqvae_weight_path = model_cfg.model.vqvae_weight_path.replace('/home/whwjdqls99/MARM/', './')
    
    opt = eval_cfg
    seed_everything(args.seed)
    opt.device = 'cuda'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.train_data == "t2m":
        data_cfg = data_cfg.t2m
    elif args.train_data == "kit":
        data_cfg = data_cfg.kit 
        
    model_opt_path = pjoin('./checkpoints/t2m/t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns', 'opt.txt')
    model_opt = get_opt(model_opt_path, device=device)
    clip_version = 'ViT-B/32'
    
    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=device)
    vq_model, vq_opt = load_vq_model(vq_opt)
    
    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, 'opt.txt')
    res_opt = get_opt(res_opt_path, device=device)
    res_model = load_res_model(res_opt)

    assert res_opt.vq_name == model_opt.vq_name
    
    if args.model_ckpt_path:
        meta_dir = os.path.dirname(os.path.dirname(args.model_ckpt_path)) + '/meta'
        # as we are evaluating with momask mean, std we don't need test_mean, test_std
        mean = np.load(meta_dir +'/mean.npy') 
        std = np.load(meta_dir +'/std.npy')
        rf_model = MotionPrior.MotionPriorWrapper(model_cfg, args.model_ckpt_path, device)
        rf_model.eval()
        rf_model.num_sample_steps = args.num_sample_steps
        rf_model.set_vqvae()
    else: # we are evaluating the original momask model
        rf_model = None
        mean = np.load("./datasets/t2m-mean.npy")
        std = np.load("./datasets/t2m-std.npy")
    
    t2m_transformer = load_trans_model(model_opt, 'latest.tar')
    t2m_transformer.eval()
    vq_model.eval()
    res_model.eval()

    t2m_transformer.to(device)
    vq_model.to(device)
    res_model.to(device)
    
    save_dir = f'./gifs/'
    os.makedirs(save_dir, exist_ok=True)
    
    visualize_motion(
        clip_text=args.input_text,
        m_length=args.m_length,
        vq_model=vq_model,
        res_model=res_model,
        rf_model=rf_model,
        trans=t2m_transformer,
        time_steps=opt.time_steps,
        cond_scale=opt.cond_scale,
        temperature=opt.temperature,
        topkr=opt.topkr,
        force_mask=opt.force_mask,
        mean=mean,
        std=std,
    )
    
    
    
    
    
    