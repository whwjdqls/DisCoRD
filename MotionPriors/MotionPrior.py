import os
from collections import OrderedDict

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from einops import pack, rearrange, reduce, repeat, unpack
from einops.layers.torch import Rearrange
from .models import (
    MMM_DiT,
    SpatialTemporalVAE,
    SpatialTemporalVQVAE,
    TemporalVAE,
    TemporalVQVAE,
    TransformerVAE,
    rectifiedflow,
    rf_decoder,
    refinefromnoise,
)
from .models.T2M_VQVAE import t2m_vqvae
from .models.vq.model import RVQVAE
from .models.diffusion import create_diffusion
from .models.ldm import autoencoder1d
from .models.rectifiedflow import simple_rectified_flow
from .models.rf_decoder.rectified_flow import RectifiedFlowDecoder


def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask  # (b, len)

def load_MotionPrior(model_ckpt_path, model_cfg):
    if model_cfg.model.name == "TemporalVAE":
        model = TemporalVAE.TVAE(
            in_dim=model_cfg.model.input_dim,  # 263
            hidden_size=model_cfg.model.hidden_size,  # 512
            dim_feedforward=model_cfg.model.dim_feedforward,  # 2048
            num_attention_heads=model_cfg.model.num_attention_heads,  # 8
            num_hidden_layers=model_cfg.model.num_hidden_layers,  # 6
            max_len=model_cfg.model.max_len,  # 3600
            quant_factor=model_cfg.model.quant_factor,  # 3
            quant_sequence_length=model_cfg.model.quant_sequence_length,  # for encoder
            sequence_length=model_cfg.model.sequence_length,  # for decoder
            pos_encoding=model_cfg.model.pos_encoding,  # None
            temporal_bias=model_cfg.model.temporal_bias,
        )
    if model_cfg.model.name == "TemporalVAEv2":
        model = TemporalVAE.TVAEv2(
            in_dim=model_cfg.model.input_dim,  # 263
            hidden_size=model_cfg.model.hidden_size,  # 512
            embed_dim=model_cfg.model.embed_dim,
            dim_feedforward=model_cfg.model.dim_feedforward,  # 2048
            num_attention_heads=model_cfg.model.num_attention_heads,  # 8
            num_hidden_layers=model_cfg.model.num_hidden_layers,  # 6
            max_len=model_cfg.model.max_len,  # 3600
            quant_factor=model_cfg.model.quant_factor,  # 3
            quant_sequence_length=model_cfg.model.quant_sequence_length,  # for encoder
            sequence_length=model_cfg.model.sequence_length,  # for decoder
            pos_encoding=model_cfg.model.pos_encoding,  # None
            temporal_bias=model_cfg.model.temporal_bias,
        )
    elif model_cfg.model.name == "TrVAE":
        model = TransformerVAE.ActorVae(
            ablation=model_cfg.model.name,
            nfeats=model_cfg.model.input_dim,  # 263
            latent_dim=model_cfg.model.hidden_size,  # 512
            ff_size=model_cfg.model.dim_feedforward,  # 2048
            num_heads=model_cfg.model.num_attention_heads,  # 8
            num_layers=model_cfg.model.num_hidden_layers,  # 6
            dropout=model_cfg.model.dropout,  # 0.1
            is_vae=True,
            activation=model_cfg.model.activation,  # 'gelu'
            position_embedding=model_cfg.model.pos_encoding,
        )
    elif model_cfg.model.name == "LDMVAE":
        model = autoencoder1d.Autoencoder1DKL(
            in_ch=model_cfg.model.input_dim,  # 263
            ch=model_cfg.model.hidden_size,  # 512
            out_ch=model_cfg.model.input_dim,  # 263
            embed_dim=model_cfg.model.embed_dim,
            ch_mult=model_cfg.model.ch_mult,
            resolution=model_cfg.model.sequence_length,
            attn_res=(model_cfg.model.quant_sequence_length,),  # attention at the last layer
            use_variational=True,
        )
    elif model_cfg.model.name == "ResTVAE":
        model = TemporalVAE.ResTVAE(
            in_dim=model_cfg.model.input_dim,
            latent_dim=model_cfg.model.latent_dim,
            hidden_size=model_cfg.model.hidden_size,
            depth=model_cfg.model.depth,
            quant_factor=model_cfg.model.quant_factor,
            stride_t=model_cfg.model.stride_t,
            dilation_growth_rate=model_cfg.model.dilation_growth_rate,
            activation=model_cfg.model.activation,
            norm=model_cfg.model.norm,
        )
    elif model_cfg.model.name == "TemporalVQVAE":
        model = TemporalVQVAE.TVQVAE(
            in_dim=model_cfg.model.input_dim,
            hidden_size=model_cfg.model.hidden_size,
            embed_dim=model_cfg.model.embed_dim,
            n_embed=model_cfg.model.n_embed,
            dim_feedforward=model_cfg.model.dim_feedforward,
            num_attention_heads=model_cfg.model.num_attention_heads,
            num_hidden_layers=model_cfg.model.num_hidden_layers,
            max_len=model_cfg.model.max_len,
            quant_factor=model_cfg.model.quant_factor,
            quant_sequence_length=model_cfg.model.quant_sequence_length,
            sequence_length=model_cfg.model.sequence_length,
            quantizer=model_cfg.model.quantizer,
            pos_encoding=model_cfg.model.pos_encoding,
            temporal_bias=model_cfg.model.temporal_bias,
        )
    elif model_cfg.model.name == "ResTVQVAE":
        model = TemporalVQVAE.ResTVQVAE(
            in_dim=model_cfg.model.input_dim,  # input_width for momask
            hidden_size=model_cfg.model.hidden_size,  # width for momask
            embed_dim=model_cfg.model.embed_dim,  # embed_dim for momask
            n_embed=model_cfg.model.n_embed,
            depth=model_cfg.model.depth,
            quant_factor=model_cfg.model.quant_factor,  # down_t
            stride_t=model_cfg.model.stride_t,
            dilation_growth_rate=model_cfg.model.dilation_growth_rate,
            activation=model_cfg.model.activation,
            quantizer=model_cfg.model.quantizer,
            norm=model_cfg.model.norm,
        )
    elif model_cfg.model.name == "STVAE":
       model = SpatialTemporalVAE.STVAE(
            in_dim=model_cfg.model.input_dim,
            hidden_size_2d=model_cfg.model.hidden_size_2d,
            hidden_size_1d=model_cfg.model.hidden_size_1d,
            embed_dim=model_cfg.model.embed_dim,
            dim_feedforward=model_cfg.model.dim_feedforward,
            num_attention_heads=model_cfg.model.num_attention_heads,
            num_hidden_layers=model_cfg.model.num_hidden_layers,
            max_len=model_cfg.model.max_len,
            spatial_quant_factor=model_cfg.model.spatial_quant_factor,
            spatial_quant_sequence_length=model_cfg.model.spatial_quant_sequence_length,
            spatial_sequence_length=model_cfg.model.spatial_sequence_length,
            quant_factor=model_cfg.model.quant_factor,
            quant_sequence_length=model_cfg.model.quant_sequence_length,
            sequence_length=model_cfg.model.sequence_length,
            joints_num=22, # THIS IS HARD CODED!!!
            pos_encoding=model_cfg.model.pos_encoding,
            temporal_bias=model_cfg.model.temporal_bias,
        )
    elif model_cfg.model.name == "STVQVAE":
        model = SpatialTemporalVQVAE.STVQVAE(
            in_dim=model_cfg.model.input_dim,
            hidden_size_2d=model_cfg.model.hidden_size_2d,
            hidden_size_1d=model_cfg.model.hidden_size_1d,
            embed_dim=model_cfg.model.embed_dim,
            n_embed=model_cfg.model.n_embed,
            dim_feedforward=model_cfg.model.dim_feedforward,
            num_attention_heads=model_cfg.model.num_attention_heads,
            num_hidden_layers=model_cfg.model.num_hidden_layers,
            max_len=model_cfg.model.max_len,
            spatial_quant_factor=model_cfg.model.spatial_quant_factor,
            spatial_quant_sequence_length=model_cfg.model.spatial_quant_sequence_length,
            spatial_sequence_length=model_cfg.model.spatial_sequence_length,
            quant_factor=model_cfg.model.quant_factor,
            quant_sequence_length=model_cfg.model.quant_sequence_length,
            sequence_length=model_cfg.model.sequence_length,
            joints_num=22, # this is hard CODED@@!@@!!@@!
            quantizer=model_cfg.model.quantizer,
            pos_encoding=model_cfg.model.pos_encoding,
            temporal_bias=model_cfg.model.temporal_bias,
        )
    elif model_cfg.model.name == "MMM_DiT":
        model = MMM_DiT.MMM_DiT(
            output_dim=model_cfg.model.output_dim,
            nb_code=model_cfg.model.nb_code,
            code_dim=model_cfg.model.code_dim,
            output_emb_width=model_cfg.model.output_emb_width,
            down_t=model_cfg.model.quant_factor,
            stride_t=model_cfg.model.stride_t,
            width=model_cfg.model.width,
            depth=model_cfg.model.depth,
            dilation_growth_rate=model_cfg.model.dilation_growth_rate,
            activation=model_cfg.model.activation,
            dit_embed_dim=model_cfg.model.dit_embed_dim,
            dit_num_heads=model_cfg.model.dit_num_heads,
            dit_mlp_ratio=model_cfg.model.dit_mlp_ratio,
            dit_depth=model_cfg.model.dit_depth,
            dit_drop_out_prob=model_cfg.model.dit_drop_out_prob,
            quantizer=model_cfg.model.quantizer,
            norm=model_cfg.model.norm,
        )
        weight = torch.load(model_cfg.model.vqvae_weight_path, weights_only=True)['net']
        new_weight = OrderedDict()
        for key, value in weight.items():
            new_weight[key.replace("vqvae.", "")] = value
        model.vqvae.load_state_dict(new_weight)
        
    elif model_cfg.model.name in ["Momask_RVQVAE", "Momask_VQVAE"]:
        # momask RVQVAE is loaded here and returned instantly
        opt = model_cfg.model
        if opt.dataset_name == 't2m':
            pose_dim = 263
        elif opt.dataset_name == 'kit':
            pose_dim = 251
        model = RVQVAE(
            opt,
            pose_dim, 
            opt.nb_code,
            opt.code_dim,
            opt.code_dim,
            opt.down_t,
            opt.stride_t,
            opt.width,
            opt.depth,
            opt.dilation_growth_rate,
            opt.vq_act,
            opt.vq_norm
        )
        ckpt = torch.load(model_ckpt_path, map_location='cpu')
        model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
        model.load_state_dict(ckpt[model_key])
        return model
    
    elif model_cfg.model.name == "MMM_VQVAE":
        model = MMM_DiT.VQVAE_251(
            output_dim=263,
            nb_code=8192,
            code_dim=32,
            output_emb_width=32,
            down_t=2,
            stride_t=2,
            width=512,
            depth=3,
            dilation_growth_rate=3,
            activation='relu',
            quantizer='ema_reset',
            norm=None)
        weight = torch.load(model_ckpt_path, weights_only=True)['net']
        new_weight = OrderedDict()
        for key, value in weight.items():
            new_weight[key.replace("vqvae.", "")] = value
        model.load_state_dict(new_weight)
        print(f"Loaded MMM_VQVAE model from {model_ckpt_path}")
        return model
    
    elif model_cfg.model.name == "T2M_VQVAE":
        opt = model_cfg.model
        if opt.dataset_name == 't2m':
            pose_dim = 263
        elif opt.dataset_name == 'kit':
            pose_dim = 251
        model = t2m_vqvae.HumanVQVAE(
            output_dim=pose_dim,
            nb_code=512,
            code_dim=512,
            output_emb_width=512,
            down_t=2,
            stride_t=2,
            width=512,
            depth=3,
            dilation_growth_rate=3,
            activation='relu',
            quantizer='ema_reset',
            norm=None)
        ckpt = torch.load(model_ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['net'], strict=True)
        print(f"Loaded T2M_VQVAE model from {model_ckpt_path}")
        return model
    
    elif model_cfg.model.name in ["RectifiedFlow"]:
        denoiser = rectifiedflow.get_flow_backbone(model_cfg)
        model = simple_rectified_flow.Flow(
            dim = model_cfg.model.output_dim,
            net = denoiser,
            noise_std = model_cfg.model.noise_std,
            use_diffusion_pos_embed = model_cfg.model.use_diffusion_pos_embed)
    elif model_cfg.model.name in ["RFDecoder"]:
        denoiser = rf_decoder.get_flow_backbone(model_cfg)
        model = RectifiedFlowDecoder(model = denoiser)
        
    elif model_cfg.model.name in ["RFfromNoise"]:
        denoiser = refinefromnoise.get_flow_backbone(model_cfg)
        model = RectifiedFlowDecoder(model = denoiser)
    else:
        raise ValueError(f"Model {model_cfg.model.name} not supported.")

    if model_ckpt_path is not None:
        if model.__class__.__name__ != "MMM_DiT":
            model.load_state_dict(torch.load(model_ckpt_path, map_location="cpu"))
        else:
            model.refiner.load_state_dict(torch.load(model_ckpt_path, map_location="cpu")) # only load the refiner
        print(f"Loaded {model_cfg.model.name} model from {model_ckpt_path}")
    else:
        print(f"Created {model_cfg.model.name} model from scratch")
    return model


class MotionPriorWrapper(nn.Module):
    """
    Wrapper class for loading and using a trained Motion Prior model.
    """

    def __init__(self, model_cfg, model_ckpt, device):
        super(MotionPriorWrapper, self).__init__()
        self.device = device
        self.model_cfg = model_cfg
        self.model_ckpt = model_ckpt
        
        if "vqvae_weight_path" in model_cfg.model.keys():
            vae_ckpt = self.model_cfg.model.vqvae_weight_path
            vae_root = os.path.dirname(os.path.dirname(vae_ckpt))
            vae_cfg_path = os.path.join(vae_root, "configs/config_model.yaml")
            self.vae_cfg = OmegaConf.load(vae_cfg_path)
        
        try:
            if "quant_factor" in model_cfg.model.keys():
                self.quant_factor = model_cfg.model.quant_factor
            else:
                self.quant_factor = self.vae_cfg.model.quant_factor
        except:
            print("WARNING: quant_factor not found in model_cfg or vae_cfg is this intended?")
        
        if self.model_ckpt is not None:
            self.model = self._create_model()
        else:
            self.model = None
            print("No model checkpoint provided.")
            
        if self.model.__class__.__name__ == "MMM_DiT":
            self.diffusion = create_diffusion("", diffusion_steps=int(model_cfg.val.diffusion_steps),predict_xstart=True,learn_sigma=False)
        else:
            self.diffusion = None
            
        self.num_sample_steps = 16
        self.deterministic = False # deterministic sampling is controled by this flag
        self.vqvae = None 

    def set_vqvae(self, frozen=True):
        if self.model.__class__.__name__ == "Flow" or self.model.__class__.__name__ == "RectifiedFlowDecoder":
            self.vqvae = MotionPriorWrapper(self.vae_cfg,self.model_cfg.model.vqvae_weight_path, self.device)
            if frozen:
                for param in self.vqvae.parameters():
                    param.requires_grad = False
            self.vqvae.eval()
        else:
            raise ValueError("Only supported for Flow model")
             
            
    def _create_model(self):
        # Load the model using the provided function
        model = load_MotionPrior(self.model_ckpt, self.model_cfg)
        return model.to(self.device)

    def get_z(self, motion, sample=False):
        net = self.model

        if net.__class__.__name__ in ["TVAE", "TVAEv2", "STVAE"]:
            encoder_features = net.encode(motion)
            mu = net.mean(encoder_features)
            if sample:
                logvar = net.logvar(encoder_features)
                z = net.reparameterize(mu, logvar)
            else:
                z = mu

        elif net.__class__.__name__ == "ActorVae":
            latent, dist, mu, logvar = net.encode(motion)
            if sample:
                z = latent
            else:
                z = mu
        elif net.__class__.__name__ == "Autoencoder1DKL":
            posterior = net.encode(motion)
            if sample:
                z = posterior.sample().permute(0, 2, 1)
            else:
                z = posterior.mean.permute(0, 2, 1)
        elif net.__class__.__name__ in ['RVQVAE']:
            code_idx, all_codes = net.encode(motion) # all codes is (num_layers, B, dim, N) 
            z = all_codes.sum(dim=0) # (B, dim, N) or 
        elif net.__class__.__name__ in ['VQVAE_251','HumanVQVAE']:
            z = net.get_qauntized(motion) #(B, dim, N/q)  
        else:
            raise ValueError(f"Unsupported model type: {net.__class__.__name__}")
        return z

    def decode(self, latent, m_length=None):
        """
        Decodes the latent representation back into motion data.
        """
        net = self.model

        if net.__class__.__name__ in ["TVAE", "TVAEv2", "STVAE"]:
            # TemporalVAE decoding
            pred_pose_eval = net.decoder(latent)
        elif net.__class__.__name__ == "ActorVae":
            # TransformerVAE decoding
            pred_pose_eval = net.decode(latent, m_length)
        elif net.__class__.__name__ == "Autoencoder1DKL":
            latent = latent.permute(0, 2, 1)
            pred_pose_eval = net.decode(latent)


        else:
            raise ValueError(f"Unsupported model type: {net.__class__.__name__}")

        return pred_pose_eval
    
    def get_z_and_recon(self, motion, m_length=None):
        if self.model.__class__.__name__ in ['RVQVAE']:
            x_out, commit_loss, perplexity, z = self.model(motion, return_z=True)

        elif self.model.__class__.__name__ in ['VQVAE_251']:
            x_out, loss, perplexity, z = self.model(motion, return_z=True)
            # x_out (N, c, T)
        return z, x_out
    
    def forward(self, motion, m_length=None, cfg_scale=1.0, text_embedding=None, cb_replace_prob=0.0, cb_replace_topk=3):
        """
        Forward pass through the model.
        motion: Tensor of shape (B, seq_len, 263)
        m_length: Tensor of shape (B,) containing the length of each sequence in motion -> only used for Dit
        """
        net = self.model

        if net.__class__.__name__ in ["TVAE", "TVAEv2", "STVAE"]:
            # Forward pass for TemporalVAE
            pred_pose_eval, mu, logvar = net(motion)
            others = (mu, logvar)
        elif net.__class__.__name__ in ['RVQVAE']:
            if self.model_cfg.model.name == "Momask_VQVAE":
                pred_pose_eval, mu, logvar = net(motion, only_base=True)
                others = (mu, logvar)
            else:
                pred_pose_eval, commit_loss, perplexity = net(motion, cb_replace_prob=cb_replace_prob, cb_replace_topk=cb_replace_topk)
                others = (commit_loss, perplexity)
                
        elif net.__class__.__name__ in ['VQVAE_251','HumanVQVAE']:
            assert cb_replace_prob == 0.0, "cb_replace_prob is not supported for VQVAE_251"
            assert cb_replace_topk == 3, "cb_replace_topk is not supported for VQVAE_251"
            pred_pose_eval, commit_loss, perplexity = net(motion)
            others = (commit_loss, perplexity)
            
        elif net.__class__.__name__ == "ActorVae":
            # Encode motion and then decode using TransformerVAE
            latent, dist, mu, logvar = net.encode(motion) 
            pred_pose_eval = net.decode(latent, m_length)
            others = (mu, logvar)

        elif net.__class__.__name__ == "Autoencoder1DKL":
            pred_pose_eval, posterior = net(motion)
            mu = posterior.mean
            logvar = posterior.logvar
            others = (mu, logvar)
            
        elif net.__class__.__name__ == "ResTVAE":
            pred_pose_eval, mu, logvar = net(motion)
            others = (mu, logvar)
        elif net.__class__.__name__ in  ["TVQVAE","ResTVQVAE","STVQVAE"]:
            pred_pose_eval, commit_loss, perplexity = net(motion)
            others = (commit_loss, perplexity)
            
        elif net.__class__.__name__ == "MMM_DiT":
            # pred_pose_eval, vqvae_out = self.forward_with_cfg(motion, cfg_scale=cfg_scale)
            pred_pose_eval, vqvae_out = self.forward_from_t(motion, t_start=4, cfg_scale=cfg_scale)
            pred_pose_eval = pred_pose_eval.permute(0, 2, 1)
            others = (vqvae_out)
        elif net.__class__.__name__ == 'Flow':
            assert self.vqvae is not None
            assert self.vqvae.__class__.__name__ == "MotionPriorWrapper"
            if net.net.z_condition:
                z, noisy_motion = self.vqvae.get_z_and_recon(motion)
                bs, length, dim = noisy_motion.shape
                others = (noisy_motion,z)
                z = z.repeat_interleave(4, dim=-1)
                z = rearrange(z, 'b c n -> b n c')
                
                if m_length is not None:
                    padding_mask = ~lengths_to_mask(m_length, 196).to(noisy_motion.device)  # (B, L)
                else:
                    padding_mask = None

                pred_pose_eval = net.sample(noisy_motion, z=z, deterministic=self.deterministic, num_sample_steps=self.num_sample_steps,bsz=bs, length=length, padding_mask=padding_mask, text_embedding=text_embedding)
            else:                    
                noisy_motion, others = self.vqvae(motion)
                others = (noisy_motion,) + others
                bs, length, dim = noisy_motion.shape
                
                if m_length is not None:
                    padding_mask = ~lengths_to_mask(m_length, 196).to(noisy_motion.device)  # (B, L)
                else:
                    padding_mask = None
                pred_pose_eval = net.sample(noisy_motion, deterministic=self.deterministic, num_sample_steps=self.num_sample_steps,bsz=bs, length=length, padding_mask=padding_mask, text_embedding=text_embedding)
                
        elif net.__class__.__name__  == 'RectifiedFlowDecoder': # this is when using RF as a decoder!
            # we have two vairants, The RFdecoder and RFfromNoise
            assert self.vqvae is not None
            assert self.vqvae.__class__.__name__ == "MotionPriorWrapper"

            if motion.shape[1] != 196:
                old_shape = motion.shape
                motion = torch.nn.functional.pad(motion, (0, 0, 0, 196-motion.shape[1]), mode='constant', value=0)
                print(f"Padding motion of shape {old_shape} to {motion.shape} in MotionPriorWrapper forward")
                
                
            if  self.model_cfg.model.name in ["RFDecoder"]:
                y = self.vqvae.get_z(motion) # (BS, dim, N/4
                if hasattr(self.vqvae.model, "decoder"):    
                    vqvae_out = self.vqvae.model.decoder(y)
                else:
                    vqvae_out = self.vqvae.model.vqvae.decoder(y)
                # vqvae_out = self.vqvae.model.decoder(y) # as self.vqvae is also motionpriorwrapper
                # y = torch.nn.functional.interpolate(y, scale_factor=4, mode='nearest')# (BS, dim, N)
                # y = rearrange(y, 'b c n -> b n c') # actually this is latent but for the sake of naming
                y = y.permute(0, 2, 1) # (BS, N, dim)

                bs, length, dim = y.shape
                
                if m_length is not None:
                    padding_mask = ~lengths_to_mask(m_length, 196).to(y.device)  # (B, L)
                else:
                    padding_mask = None
                    
                pred_pose_eval = net.sample(y, batch_size=bs, steps=self.num_sample_steps, padding_mask=padding_mask, text_embedding=text_embedding, 
                                            data_shape=(196, self.model_cfg.model.output_dim)) # this is hard coded because we always use humanml3d for now!
                others = (vqvae_out,)
                
                
            elif self.model_cfg.model.name == 'RFfromNoise':
                noisy_motion, others = self.vqvae(motion)
                others = (noisy_motion,) + others
                bs, length, dim = noisy_motion.shape
                
                if m_length is not None:
                    padding_mask = ~lengths_to_mask(m_length, 196).to(noisy_motion.device)
                else:
                    padding_mask = None
                    
                pred_pose_eval = net.sample(noisy_motion, batch_size=bs, steps=self.num_sample_steps, padding_mask=padding_mask, text_embedding=text_embedding, 
                                            data_shape=(196, self.model_cfg.model.output_dim))
        else:
            raise ValueError(f"Unsupported model type: {net.__class__.__name__}")

        return pred_pose_eval, others
    
    def refine(self, noisy_motion, pred_ids=None, m_length=None, text_embedding=None):
        assert self.model.__class__.__name__ in ["Flow"]
        assert self.vqvae.model.__class__.__name__ in  ["RVQVAE", "VQVAE_251"]
        
    # if self.model.__class__.__name__ == "Flow": # this is for refinement network 
        bs, length, dim = noisy_motion.shape
        
        if m_length is not None:
            padding_mask = ~lengths_to_mask(m_length, 196).to(noisy_motion.device)
        else:
            padding_mask = None
        
        if self.model.net.z_condition:
            y = self.vqvae.model.quantizer.get_codes_from_indices(pred_ids) # 'q b n d'
            y = y.sum(dim=0) # B, N//4, dim
            y = y.repeat_interleave(4, dim=1) # B, N, dim
            bs, length, dim = y.shape
            
            pred_pose_eval = self.model.sample(noisy_motion, z=y, deterministic=self.deterministic, num_sample_steps=self.num_sample_steps,bsz=bs, length=length, padding_mask=padding_mask, text_embedding=text_embedding)
        else:
            pred_pose_eval = self.model.sample(noisy_motion, deterministic=self.deterministic, num_sample_steps=self.num_sample_steps, bsz=bs, length=length, padding_mask=padding_mask, text_embedding=text_embedding)
            
        return pred_pose_eval
    
    def refine_from_Noise(self, noisy_motion, m_length=None, text_embedding=None):
        assert self.model.__class__.__name__ == "RectifiedFlowDecoder" # this is for RFfromNoise
        assert self.vqvae.model.__class__.__name__ in  ["RVQVAE", "VQVAE_251"]
        assert self.model_cfg.model.name == 'RFfromNoise'
        
        bs, length, dim = noisy_motion.shape

        if m_length is not None:
            padding_mask = ~lengths_to_mask(m_length, length).to(y.device)  # (B, L)
        else:
            padding_mask = None
        
        pred_pose_eval = self.model.sample(noisy_motion, batch_size=bs, steps=self.num_sample_steps, padding_mask=padding_mask, text_embedding=text_embedding, 
                                    data_shape=(length, self.model_cfg.model.output_dim)) # this is hard coded bec

        return pred_pose_eval
    
    def decode_with_RF(self, pred_ids, m_length=None, text_embedding=None):
        """
        pred_ids: Tensor of shape (B, N//4) containing the indices of the quantized latent vectors.
        """
        assert self.vqvae.model.__class__.__name__ in ["RVQVAE", "VQVAE_251","HumanVQVAE"]
        assert self.model.__class__.__name__ == "RectifiedFlowDecoder"
        if self.vqvae.model.__class__.__name__ == "RVQVAE":
            y = self.vqvae.model.quantizer.get_codes_from_indices(pred_ids)# 'q b n d'
            y = y.sum(dim=0)
        else:# this is for HumanVQVAE
            raise ValueError("Not supported for HumanVQVAE")
            y = self.vqvae.model.vqvae.quantizer.dequantize(pred_ids)# 'q b n d'
         # B, N//4, dim
        # y = y.repeat_interleave(4, dim=1) # B, N, dim
        # y = y.permute(0, 2, 1)
        bs, length, dim = y.shape
        data_length = length*4
        if m_length is not None:
            padding_mask = ~lengths_to_mask(m_length, 196).to(y.device)  # (B, L)
        else:
            padding_mask = None
        
        pred_pose_eval = self.model.sample(y, batch_size=bs, steps=self.num_sample_steps, padding_mask=padding_mask, text_embedding=text_embedding, 
                                    data_shape=(data_length, self.model_cfg.model.output_dim)) # this is hard coded bec
        return pred_pose_eval
    
    def forward_with_cfg(self, motion, cfg_scale=1.0):
        net = self.model
        assert net.__class__.__name__ == "MMM_DiT"
        
        device = motion.device
        vqvqae_out, loss, perplexity = net.vqvae_forward(motion)
        
        bs, t, dim = vqvqae_out.shape
        
        z = torch.randn(bs, dim, t, device = device) # random noise
        condition = vqvqae_out.permute(0, 2, 1) # (BS, dim, t)
        
        model_kwargs = {"c":condition, "cfg_scale":cfg_scale}
        samples = self.diffusion.p_sample_loop(
            net.refiner.forward_with_cfg, z.shape, noise=z, clip_denoised=False, 
            model_kwargs=model_kwargs, progress=False, device=device
        )
        
        return samples, vqvqae_out
                
    def forward_from_t(self, motion, t_start, cfg_scale=1.0):
        net = self.model
        assert net.__class__.__name__ == "MMM_DiT"
        assert int(self.model_cfg.val.diffusion_steps)>=t_start
        assert isinstance(t_start, int)
        device = motion.device
        t_start = torch.tensor(t_start, device=motion.device)
        
        vqvqae_out, loss, perplexity = net.vqvae_forward(motion)
        
        bs, t, dim = vqvqae_out.shape
        vqvae_out_permuted = vqvqae_out.permute(0, 2, 1) # (BS, dim, t)
        model_kwargs = {"c":vqvae_out_permuted, "cfg_scale":cfg_scale}
        samples = self.diffusion.p_sample_loop_from_t(
            net.refiner.forward_with_cfg, vqvae_out_permuted.shape, 
            x_start = vqvae_out_permuted,
            t_start = t_start, clip_denoised=False,
            model_kwargs=model_kwargs, progress=False, device=device)
        
        return samples, vqvqae_out
        
        
        
    # def forward_with_noise(self, motion):
    #     net = self.model
    #     elif net.__class__.__name__ in ['VQVAE_251', 'RVQVAE']:
    #         pred_pose_eval, commit_loss, perplexity = net(motion)
    #         others = (commit_loss, perplexity)
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                
        