from . import DiTforflow, MLPAdaLN, Unet1D, simple_rectified_flow


def get_flow_backbone(model_cfg):  
    model_cfg = model_cfg.model
    if model_cfg.MLPAdaLN.use:
        return MLPAdaLN.MLPAdaLN(
            model_cfg.output_dim,
            dim_cond = model_cfg.MLPAdaLN.dim_cond,
            depth = model_cfg.MLPAdaLN.depth,   
            width = model_cfg.MLPAdaLN.width,
            dropout = model_cfg.MLPAdaLN.dropout
        )
    elif model_cfg.Unet1D.use:
        return Unet1D.Unet1DforFlow(
            dim=model_cfg.Unet1D.dim,
            channels=model_cfg.output_dim,
            dim_mults=model_cfg.Unet1D.dim_mults,
            z_condition=model_cfg.z_condition,
            z_in_dim=model_cfg.Unet1D.z_in_dim,
            z_proj_dim=model_cfg.Unet1D.z_proj_dim,
            z_proj_type=model_cfg.Unet1D.z_proj_type,
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
        
    elif model_cfg.DiT.use:
        t_embedder = model_cfg.DiT.get('t_embedder', 'dit')
        z_condition = model_cfg.get('z_condition', False)
        z_in_dim = model_cfg.get('z_in_dim', 512)
        z_proj_dim = model_cfg.get('z_proj_dim', 256)
        z_proj_type = model_cfg.get('z_proj_type', 'linear')
            
        return DiTforflow.DiT1DforFlow(
            out_dim=model_cfg.output_dim,
            embed_dim=model_cfg.DiT.hidden_size,
            num_heads=model_cfg.DiT.num_heads, 
            mlp_ratio=model_cfg.DiT.mlp_ratio, 
            z_condition=z_condition,
            z_proj_type=z_proj_type,
            z_in_dim=z_in_dim,
            z_proj_dim=z_proj_dim,
            t_embedder=t_embedder,
            depth=model_cfg.DiT.num_layers, 
            max_seq_len=model_cfg.DiT.max_seq_len, 
            drop_out_prob=model_cfg.DiT.drop_out_prob,
            text_condition=model_cfg.text_condition if "text_condition" in model_cfg.keys() else None,
            pos_encoding=model_cfg.DiT.pos_encoding,
            temporal_bias=model_cfg.DiT.temporal_bias)