from . import DiTforflow_decoder
from . import Unet1Dflow_decoder

def get_flow_backbone(model_cfg):  
    model_cfg = model_cfg.model
    if model_cfg.DiT.use:
        return DiTforflow_decoder.DiT1DforFlowDecoder(
            out_dim=model_cfg.output_dim,
            embed_dim=model_cfg.DiT.hidden_size,
            c_in_dim=model_cfg.DiT.c_in_dim,
            c_proj_dim=model_cfg.DiT.c_proj_dim,
            num_heads=model_cfg.DiT.num_heads, 
            mlp_ratio=model_cfg.DiT.mlp_ratio, 
            t_embedder=model_cfg.DiT.t_embedder,
            depth=model_cfg.DiT.num_layers, 
            max_seq_len=model_cfg.DiT.max_seq_len, 
            drop_out_prob=model_cfg.DiT.drop_out_prob,
            text_condition=model_cfg.text_condition if "text_condition" in model_cfg.keys() else None,
            pos_encoding=model_cfg.DiT.pos_encoding,
            temporal_bias=model_cfg.DiT.temporal_bias)
    elif model_cfg.Unet1D.use:
        if "up_conv_c" in model_cfg.Unet1D.keys():
            up_conv_c = model_cfg.Unet1D.up_conv_c
        else:
            up_conv_c = False
        if "resnet_per_block" in model_cfg.Unet1D.keys():
            resnet_per_block = model_cfg.Unet1D.resnet_per_block
        else:
            resnet_per_block = 1
            
        return Unet1Dflow_decoder.Unet1DforFlowDecoder(
            dim=model_cfg.Unet1D.dim,
            channels=model_cfg.output_dim,
            dim_mults=model_cfg.Unet1D.dim_mults,
            c_in_dim=model_cfg.Unet1D.c_in_dim,
            c_proj_dim=model_cfg.Unet1D.c_proj_dim,
            resnet_per_block=resnet_per_block,
            up_conv_c=up_conv_c,
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