import os
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from einops import rearrange

from .models import rf_decoder
from .models.T2M_VQVAE import t2m_vqvae
from .models.vq.model import RVQVAE
from .models.rf_decoder.rectified_flow import RectifiedFlowDecoder


def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask  # (b, len)
def load_MotionPrior(model_ckpt_path, model_cfg):
    if model_cfg.model.name in ["Momask_RVQVAE", "Momask_VQVAE"]:
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
    elif model_cfg.model.name in ["RFDecoder"]:
        denoiser = rf_decoder.get_flow_backbone(model_cfg)
        model = RectifiedFlowDecoder(model = denoiser)
    else:
        raise ValueError(f"Model {model_cfg.model.name} not supported.")

    if model_ckpt_path is not None:
        model.load_state_dict(torch.load(model_ckpt_path, map_location="cpu"))
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

        if net.__class__.__name__ in ['RVQVAE']:
            code_idx, all_codes = net.encode(motion) # all codes is (num_layers, B, dim, N) 
            z = all_codes.sum(dim=0) # (B, dim, N) or 
        else:
            raise ValueError(f"Unsupported model type: {net.__class__.__name__}")
        return z
    
    def get_z_and_recon(self, motion, m_length=None):
        if self.model.__class__.__name__ in ['RVQVAE']:
            x_out, commit_loss, perplexity, z = self.model(motion, return_z=True)
        else:
            raise ValueError(f"Unsupported model type: {self.model.__class__.__name__}")
            
        return z, x_out
    
    def forward(self, motion, m_length=None, cfg_scale=1.0, text_embedding=None, cb_replace_prob=0.0, cb_replace_topk=3):
        """
        Forward pass through the model.
        motion: Tensor of shape (B, seq_len, 263)
        m_length: Tensor of shape (B,) containing the length of each sequence in motion -> only used for Dit
        """
        net = self.model

        if net.__class__.__name__ in ['RVQVAE']:
            if self.model_cfg.model.name == "Momask_VQVAE":
                pred_pose_eval, mu, logvar = net(motion, only_base=True)
                others = (mu, logvar)
            else:
                pred_pose_eval, commit_loss, perplexity = net(motion, cb_replace_prob=cb_replace_prob, cb_replace_topk=cb_replace_topk)
                others = (commit_loss, perplexity)
                
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
                
        else:
            raise ValueError(f"Unsupported model type: {net.__class__.__name__}")

        return pred_pose_eval, others
    
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
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                
        