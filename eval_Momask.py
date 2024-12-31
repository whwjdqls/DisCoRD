import os
from os.path import join as pjoin
import torch
import numpy as np
import argparse
from MotionGen.momask_transformer.transformer import MaskTransformer, ResidualTransformer
from MotionPriors import MotionPrior
from evaluation.get_opt import get_opt
from evaluation.t2m_eval_wrapper import EvaluatorModelWrapper
from evaluation import eval_t2m
from evaluation.eval_option_momask_trans import EvalT2MOptions, get_momask_default_options
from datasets.t2m_dataset import Text2MotionDatasetEval, collate_fn
from utils.word_vectorizer import WordVectorizer
from utils.utils import generate_date_time, seed_everything
from MotionPriors.models.vq.model import RVQVAE
from configs import config_utils
# python eval_Momask.py --time_steps 10 --ext evaluation
def load_vq_model(vq_opt, train_data='t2m'):
    # opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    if train_data == 'kit':
        motion_dim = 251
    else:
        motion_dim = 263
        
    vq_model = RVQVAE(vq_opt,
                motion_dim,
                vq_opt.nb_code,
                vq_opt.code_dim,
                vq_opt.output_emb_width,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)
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
                                      opt=model_opt)
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model),
                      map_location=device)
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    # print(ckpt.keys())
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
                                            # codebook=vq_model.quantizer.codebooks[0] if opt.fix_token_emb else None,
                                            share_weight=res_opt.share_weight,
                                            clip_version=clip_version,
                                            opt=res_opt)

    ckpt = torch.load(pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model', 'net_best_fid.tar'),
                      map_location=device)
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer'], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Residual Transformer {res_opt.name} from epoch {ckpt["ep"]}!')
    return res_transformer

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_cfg_pth', type=str, default='./configs/momask_trans_eval_config.yaml')
    parser.add_argument("--data_cfg_path", type=str, default="./configs/t2mdataset.yaml")
    parser.add_argument("--model_cfg_path", type=str, default="./configs/temporalVAE.yaml")
    parser.add_argument("--train_data", type=str, default="t2m", choices=["t2m", "kit"])
    parser.add_argument("--num_sample_steps", type=int, default=16)
    # parser.add_argument("--deterministic", action='store_true')
    parser.add_argument("--decoder_noise", type=float, default=0) # if 0, deterministic
    parser.add_argument("--model_ckpt_path", type=str, default=None, help="if None, we are evaluating the original momask model, else we are evaluating refinement model")
    parser.add_argument("--seed", type=int, default=24)
    args = parser.parse_args()
    
    eval_cfg = config_utils.get_yaml_config(args.eval_cfg_pth)
    data_cfg = config_utils.get_yaml_config(args.data_cfg_path)
    model_cfg = config_utils.get_yaml_config(args.model_cfg_path)

    opt = eval_cfg
    seed_everything(args.seed)
    opt.device = 'cuda'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.autograd.set_detect_anomaly(True)
    
    if args.train_data == "t2m":
        data_cfg = data_cfg.t2m
    elif args.train_data == "kit":
        data_cfg = data_cfg.kit 
    
    if args.train_data == 'kit':
        model_opt_path = './checkpoints/kit/t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns_k/opt.txt'
    elif args.train_data == 't2m':
        model_opt_path = './checkpoints/t2m/t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns/opt.txt'
    model_opt = get_opt(model_opt_path, device=device)
    clip_version = 'ViT-B/32'

    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=device)
    vq_model, vq_opt = load_vq_model(vq_opt,train_data=args.train_data)

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, 'opt.txt')
    res_opt = get_opt(res_opt_path, device=device)
    res_model = load_res_model(res_opt)

    assert res_opt.vq_name == model_opt.vq_name

    dataset_opt_path = f'evaluation/models/{args.train_data}/Comp_v6_KLD005/opt.txt'
    wrapper_opt = get_opt(dataset_opt_path,device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    EvaluatorModelWrapper.device = device

    ##### ---- Dataloader ---- #####
    opt.nb_joints = 21 if opt.dataset_name == 'kit' else 22
    if args.train_data == 'kit':
        test_mean = np.load("./datasets/kit_mean.npy")
        test_std = np.load("./datasets/kit_std.npy")
    elif args.train_data == 't2m':
        test_mean = np.load("./datasets/t2m-mean.npy")
        test_std = np.load("./datasets/t2m-std.npy")
        
        
    if args.model_ckpt_path:
        meta_dir = os.path.dirname(os.path.dirname(args.model_ckpt_path)) + '/meta'
        # as we are evaluating with momask mean, std we don't need test_mean, test_std
        mean = np.load(meta_dir +'/mean.npy') 
        std = np.load(meta_dir +'/std.npy')
        rf_model = MotionPrior.MotionPriorWrapper(model_cfg, args.model_ckpt_path, device)
        rf_model.eval()
        rf_model.num_sample_steps = args.num_sample_steps
        if model_cfg.model.name == "RectifiedFlow":
            if args.decoder_noise == 0:
                rf_model.deterministic = True
            else:
                rf_model.deterministic = False
            rf_model.model.noise_std = args.decoder_noise

        rf_model.set_vqvae() # as we have z condition, we always load vqvae
    else: # we are evaluating the original momask model
        # raise NotImplementedError
        rf_model = None
        mean = test_mean
        std = test_std

        
    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    test_dataset = Text2MotionDatasetEval(data_cfg, mean, std, w_vectorizer, split='test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, drop_last=True, collate_fn=collate_fn,shuffle=True, pin_memory=True)
    
    t2m_transformer = load_trans_model(model_opt, 'latest.tar')
    t2m_transformer.eval()
    vq_model.eval()
    res_model.eval()

    t2m_transformer.to(device)
    vq_model.to(device)
    res_model.to(device)

    fid, div, top1, top2, top3, matching, mm = [], [], [], [], [], [], []
    repeat_time = 20
    for i in range(repeat_time):
        with torch.no_grad():
            best_fid, best_div, Rprecision, best_matching, best_mm = \
                eval_t2m.evaluation_mask_transformer_test_plus_res(test_dataloader, vq_model, res_model, t2m_transformer,
                                                                    i, eval_wrapper=eval_wrapper, 
                                                                    rf_model=rf_model,
                                                        time_steps=opt.time_steps, cond_scale=opt.cond_scale,
                                                        temperature=opt.temperature, topkr=opt.topkr,
                                                                    force_mask=opt.force_mask, cal_mm=True)
        fid.append(best_fid)
        div.append(best_div)
        top1.append(Rprecision[0])
        top2.append(Rprecision[1])
        top3.append(Rprecision[2])
        matching.append(best_matching)
        mm.append(best_mm)

    fid = np.array(fid)
    div = np.array(div)
    top1 = np.array(top1)
    top2 = np.array(top2)
    top3 = np.array(top3)
    matching = np.array(matching)
    mm = np.array(mm)
    
    # noise_std = rf_model.model.get('noise_std', "No_decoder_noise")
    if model_cfg.model.name == "RectifiedFlow":
        stochasticity = 'deterministic' if rf_model.deterministic else f'stochastic_{rf_model.model.noise_std}'
    else:
        stochasticity = 'thisisDecoder'
    if args.model_ckpt_path:
        base_dir = os.path.dirname(os.path.dirname(args.model_ckpt_path)) # base model directory
        model_name = os.path.basename(args.model_ckpt_path).split('.')[0] # model name
        eval_dir = pjoin(base_dir, 'eval')
        os.makedirs(eval_dir, exist_ok=True)
        file = pjoin(eval_dir, f'{model_name}_eval_momask_step{args.num_sample_steps}_{stochasticity}_seed{args.seed}_date{generate_date_time()}.txt')
    else:
        file = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, f'eval_seed{args.seed}_date{generate_date_time()}.txt')
    f = open(file, 'w')
    
    print(f'{file} final result:')
    print(f'{file} final result:', file=f, flush=True)

    msg_final = f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1) * 1.96 / np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2) * 1.96 / np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tMatching: {np.mean(matching):.3f}, conf. {np.std(matching) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tMultimodality:{np.mean(mm):.3f}, conf.{np.std(mm) * 1.96 / np.sqrt(repeat_time):.3f}\n\n"
    # logger.info(msg_final)
    print(msg_final)
    print(msg_final, file=f, flush=True)

    f.close()


# python eval_t2m_trans.py --name t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_vq --dataset_name t2m --gpu_id 3 --cond_scale 4 --time_steps 18 --temperature 1 --topkr 0.9 --gumbel_sample --ext cs4_ts18_tau1_topkr0.9_gs