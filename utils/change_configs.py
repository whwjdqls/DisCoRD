from omegaconf import OmegaConf
from configs import config_utils
import os

def main(args):
    model_cfg = config_utils.get_yaml_config(args.model_cfg_path)
    
    model_cfg.model.encoder_noise_std = args.encoder_noise_std
    model_cfg.model.noise_std = args.noise_std
    model_cfg.model.cb_replace_prob = args.cb_replace_prob
    model_cfg.model.cb_replace_topk = args.cb_replace_topk

    os.makedirs(os.path.dirname(args.model_cfg_path) + '/debug', exist_ok=True)
    debug_model_cfg_path = os.path.dirname(args.model_cfg_path) + f'/debug/{args.gpu_id}.yaml'
    
    with open(debug_model_cfg_path, 'w') as fp:
        OmegaConf.save(config=model_cfg, f=fp.name)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cfg_path", type=str, required=True)
    parser.add_argument("--encoder_noise_std", type=float, required=True)
    parser.add_argument("--noise_std", type=float, required=True)
    parser.add_argument("--cb_replace_prob", type=float, required=True)
    parser.add_argument("--cb_replace_topk", type=int, required=True)
    parser.add_argument("--gpu_id", type=str, required=True)
    args = parser.parse_args()
    main(args)