
from . import Unet1Drefinefromnoise

def get_flow_backbone(model_cfg):  
    model_cfg = model_cfg.model
    if model_cfg.DiT.use:
        raise NotImplementedError(f"Model {model_cfg.name} not implemented")
    elif model_cfg.Unet1D.use:
        return Unet1Drefinefromnoise.UnetRFfromNoise(
            dim=model_cfg.Unet1D.dim,
            channels=model_cfg.output_dim,
            dim_mults=model_cfg.Unet1D.dim_mults,
            c_in_dim=model_cfg.Unet1D.c_in_dim,
            c_proj_dim=model_cfg.Unet1D.c_proj_dim,
            resnet_per_block=model_cfg.Unet1D.resnet_per_block,
            dropout=model_cfg.Unet1D.dropout,
            learned_variance=model_cfg.Unet1D.learned_variance,
            learned_sinusoidal_cond = model_cfg.Unet1D.learned_sinusoidal_cond,
            random_fourier_features = model_cfg.Unet1D.random_fourier_features,
            learned_sinusoidal_dim = model_cfg.Unet1D.learned_sinusoidal_dim,
            sinusoidal_pos_emb_theta = model_cfg.Unet1D.sinusoidal_pos_emb_theta,
            use_attention=model_cfg.Unet1D.use_attention,
            attn_dim_head=model_cfg.Unet1D.attn_dim_head,
            attn_heads=model_cfg.Unet1D.attn_heads)
        
        raise NotImplementedError(f"Model {model_cfg.name} not implemented")