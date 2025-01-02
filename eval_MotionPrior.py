import argparse
import os
import sys
from os.path import join as pjoin
from tqdm import tqdm

import numpy as np
import torch
from configs import config_utils
from datasets.t2m_dataset import Text2MotionDatasetEval, collate_fn
from evaluation import eval_t2m
from evaluation.get_opt import get_opt
from evaluation.t2m_eval_wrapper import EvaluatorModelWrapper
from MotionPriors import MotionPrior
from MotionPriors.models.vq.model import RVQVAE
from utils.utils import generate_date_time, seed_everything
from utils.word_vectorizer import WordVectorizer


def load_vq_model(vq_opt,rvq_model_dir, which_epoch):
    vq_model = RVQVAE(vq_opt,
                263, # THIS IS HARD CODED!!
                vq_opt.nb_code,
                vq_opt.code_dim,
                vq_opt.code_dim,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)
    ckpt = torch.load(pjoin(rvq_model_dir, 'model', which_epoch),
                            map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    vq_epoch = ckpt['ep'] if 'ep' in ckpt else -1
    print(f'Loading VQ Model {vq_opt.name} Completed!, Epoch {vq_epoch}')
    return vq_model, vq_epoch

def main(args):
    # # Load Configs
    data_cfg = config_utils.get_yaml_config(args.data_cfg_path)
    model_cfg = config_utils.get_yaml_config(args.model_cfg_path)
    # # setup seed, device, wandb
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Data
    if args.train_data == "t2m":
        data_cfg = data_cfg.t2m
    elif args.train_data == "kit":
        data_cfg = data_cfg.kit 

    meta_dir = os.path.dirname(os.path.dirname(args.model_ckpt_path)) + '/meta'

    if args.train_data == 'kit':
        test_mean = np.load("./datasets/kit_mean.npy")
        test_std = np.load("./datasets/kit_std.npy")
    else:
        test_mean = np.load("./datasets/t2m-mean.npy")
        test_std = np.load("./datasets/t2m-std.npy")

    mean = np.load(meta_dir +'/mean.npy') 
    std = np.load(meta_dir +'/std.npy')
    
    # this part is from momask-codes
    dataset_opt_path = f'evaluation/models/{args.train_data}/Comp_v6_KLD005/opt.txt'
    wrapper_opt = get_opt(dataset_opt_path,device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    EvaluatorModelWrapper.device = device
    
    ##### ---- Dataloader ---- #####
    nb_joints = 21 if args.train_data == 'kit' else 22
    dim_pose = 251 if args.train_data == 'kit' else 263

    print(f"Loading model from {args.model_ckpt_path}")
    
    if args.rvq_model_dir:
        vq_opt_path = pjoin(args.rvq_model_dir, 'opt.txt')
        vq_opt = get_opt(vq_opt_path, device=device)
        net, vq_epoch = load_vq_model(vq_opt, args.rvq_model_dir, 'net_best_fid.tar')
    else:
        net = MotionPrior.MotionPriorWrapper(model_cfg, args.model_ckpt_path, device)
    
    if model_cfg.model.name == "RFDecoder":
        vae_ckpt = model_cfg.model.vqvae_weight_path # pretrained VQVAE
        vae_root = os.path.dirname(os.path.dirname(vae_ckpt))
        vae_cfg_path = os.path.join(vae_root, "configs/config_model.yaml")
        vae_meta_path = os.path.join(vae_root, "meta")
        vae_mean = np.load(os.path.join(vae_meta_path, "mean.npy"))
        vae_std = np.load(os.path.join(vae_meta_path, "std.npy"))
        vae_cfg = config_utils.get_yaml_config(vae_cfg_path)
        vqvae = MotionPrior.MotionPriorWrapper(vae_cfg, vae_ckpt, device)
        net.vqvae = vqvae  
        assert torch.allclose(torch.tensor(mean), torch.tensor(vae_mean), atol=1e-5), "mean not equal"
        assert torch.allclose(torch.tensor(std), torch.tensor(vae_std), atol=1e-5), "std not equal"
        net.num_sample_steps = args.num_sample_steps
    
    net.eval()
    net.to(device)
    
    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    test_dataset = Text2MotionDatasetEval(data_cfg, mean, std, w_vectorizer, split='test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, drop_last=True, collate_fn=collate_fn,shuffle=True, pin_memory=True)
    
    fid, div, top1, top2, top3, matching, mae, gt_jerk_list, pred_jerk_list = [], [], [], [], [], [], [], [], []
    pred_static_areas, pred_noise_areas, pred_static_areas_v2, pred_noise_areas_v2 = [], [], [], []
    repeat_time = args.repeat_time
    normalize_with_test = not args.deactivate_norm_at_test
    for i in tqdm(range(repeat_time), desc="Repeat Time"):
        best_fid, best_div, Rprecision, best_matching, l1_dist, gt_jerk, pred_jerk,  pred_static_area, pred_noise_area, pred_static_area_v2, pred_noise_area_v2= \
            eval_t2m.evaluation_vqvae_plus_mpjpe_plus_jerk(test_dataloader, net, i, eval_wrapper=eval_wrapper, num_joint=nb_joints,
                                                 test_mean=test_mean, test_std=test_std, normalize_with_test=normalize_with_test,
                                                 smooth_sigma=args.smooth_sigma)
        fid.append(best_fid)
        div.append(best_div)
        top1.append(Rprecision[0])
        top2.append(Rprecision[1])
        top3.append(Rprecision[2])
        matching.append(best_matching)
        mae.append(l1_dist)
        gt_jerk_list.append(gt_jerk)
        pred_jerk_list.append(pred_jerk)
        pred_static_areas.append(pred_static_area)
        pred_noise_areas.append(pred_noise_area)
        pred_static_areas_v2.append(pred_static_area_v2)
        pred_noise_areas_v2.append(pred_noise_area_v2)
        

    fid = np.array(fid)
    div = np.array(div)
    top1 = np.array(top1)
    top2 = np.array(top2)
    top3 = np.array(top3)
    matching = np.array(matching)
    mae = np.array(mae)
    gt_jerk_list = np.array(gt_jerk_list)
    pred_jerk_list = np.array(pred_jerk_list)
    pred_noise_areas = np.array(pred_noise_areas)
    pred_static_areas = np.array(pred_static_areas)
    pred_noise_areas_v2 = np.array(pred_noise_areas_v2)
    pred_static_areas_v2 = np.array(pred_static_areas_v2)
    

    base_dir = os.path.dirname(os.path.dirname(args.model_ckpt_path)) # base model directory
    model_name = os.path.basename(args.model_ckpt_path).split('.')[0] # model name
    eval_dir = pjoin(base_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)

    file = pjoin(eval_dir, f'{model_name}_eval_smooth{args.smooth_sigma}_seed{args.seed}.txt')
    f = open(file, 'w')
    
    print(f'{file} final result')
    print(f'{file} final result', file=f, flush=True)

    msg_final = (
        f"\tFID: {np.mean(fid):.5f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}\n"
        f"\tDiversity: {np.mean(div):.5f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}\n"
        f"\tTOP1: {np.mean(top1):.5f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, "
        f"TOP2: {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, "
        f"TOP3: {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}\n"
        f"\tMatching: {np.mean(matching):.5f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}\n"
        f"\tMAE: {np.mean(mae):.5f}, conf. {np.std(mae)*1.96/np.sqrt(repeat_time):.3f}\n"
        f"\tStatic Area: {np.mean(pred_static_areas):.5f}, conf. {np.std(pred_static_areas)*1.96/np.sqrt(repeat_time):.3f}\n"
        f"\tNoise Area: {np.mean(pred_noise_areas):.5f}, conf. {np.std(pred_noise_areas)*1.96/np.sqrt(repeat_time):.3f}\n"
        f"\tStatic Area V2: {np.mean(pred_static_areas_v2):.5f}, conf. {np.std(pred_static_areas_v2)*1.96/np.sqrt(repeat_time):.3f}\n"
        f"\tNoise Area V2: {np.mean(pred_noise_areas_v2):.5f}, conf. {np.std(pred_noise_areas_v2)*1.96/np.sqrt(repeat_time):.3f}\n"
    )
    # logger.info(msg_final)
    print(msg_final)
    print(msg_final, file=f, flush=True)
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # use the paths used to train the model
    parser.add_argument("--data_cfg_path", type=str, default="./configs/config_data.yaml")
    parser.add_argument("--model_cfg_path", type=str, default="./configs/config_model.yaml")
    parser.add_argument("--model_ckpt_path", type=str)
    parser.add_argument("--train_data", type=str, default="t2m", choices=["t2m", "kit"])
    parser.add_argument("--num_sample_steps", type=int, default=16)
    parser.add_argument("--rvq_model_dir" , type=str, default=None)
    parser.add_argument("--smooth_sigma", type=float, default=0.0)
    parser.add_argument("--deactivate_norm_at_test", action="store_true", help="Deactivate normalization using test_mean and test_std")
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--repeat_time", type=int, default=20)
    args = parser.parse_args()
    main(args)
