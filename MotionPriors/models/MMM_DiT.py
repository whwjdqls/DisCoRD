import argparse

import torch
import torch.nn as nn

from .DiT import DiTRefiner
from .TemporalVQVAE_DiT import Decoder, Encoder, QuantizeEMA, QuantizeEMAReset, Quantizer, QuantizeReset


class VQVAE_251(nn.Module):
    def __init__(self,
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
                norm=None):
        
        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code
        self.quant = quantizer
        self.encoder = Encoder(
            input_emb_width=output_dim,
            output_emb_width=output_emb_width,
            down_t=down_t,
            stride_t=stride_t,
            width=width,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation,
            norm=norm)

        self.decoder = Decoder(
            input_emb_width=output_dim,
            output_emb_width=output_emb_width,
            down_t=down_t,
            stride_t=stride_t,
            width=width,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation,
            norm=norm)

        if self.quant == "ema_reset":
            self.quantizer = QuantizeEMAReset(nb_code, code_dim)
        elif self.quant == "orig":
            self.quantizer = Quantizer(nb_code, code_dim, 1.0)
        elif self.quant == "ema":
            self.quantizer = QuantizeEMA(nb_code, code_dim)
        elif self.quant == "reset":
            self.quantizer = QuantizeReset(nb_code, code_dim)


    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1).float()
        return x


    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1)
        return x


    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)
        return code_idx

    def get_qauntized(self, x):
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_quantized, loss, perplexity  = self.quantizer(x_encoder)
        return x_quantized # (Bs, c, T/q)
    
    def forward(self, x, return_z=False):
        
        x_in = self.preprocess(x)
        # Encode
        # _x_in = x_in.reshape( int(x_in.shape[0]*4), x_in.shape[1], 16)
        # x_encoder = self.encoder(_x_in)
        # x_encoder = x_encoder.reshape(x_in.shape[0], -1, int(x_in.shape[2]/4))

        # [Transformer Encoder]
        # _x_in = x_in.reshape( int(x_in.shape[0]*x_in.shape[2]/4), x_in.shape[1], 4)
        # _x_in = _x_in.permute(0,2,1)
        # x_encoder = self.encoder2(_x_in)
        # x_encoder = x_encoder.permute(0,2,1)
        # x_encoder = x_encoder.reshape(x_in.shape[0], -1, int(x_in.shape[2]/4))

        x_encoder = self.encoder(x_in)
        
        ## quantization
        x_quantized, loss, perplexity  = self.quantizer(x_encoder)

        ## decoder
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)
        
        if return_z:
            return x_out, loss, perplexity, x_quantized
    
        return x_out, loss, perplexity


    def forward_decoder(self, x):
        # x = x.clone()
        # pad_mask = x >= self.code_dim
        # x[pad_mask] = 0

        x_d = self.quantizer.dequantize(x)
        x_d = x_d.permute(0, 2, 1).contiguous()

        # pad_mask = pad_mask.unsqueeze(1)
        # x_d = x_d * ~pad_mask
        
        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out



class HumanVQVAE(nn.Module):
    def __init__(self,
                args,
                nb_code=512,
                code_dim=512,
                output_emb_width=512,
                down_t=3,
                stride_t=2,
                width=512,
                depth=3,
                dilation_growth_rate=3,
                activation='relu',
                norm=None):
        
        super().__init__()
        
        self.nb_joints = 21 if args.dataname == 'kit' else 22
        self.vqvae = VQVAE_251(263, nb_code, code_dim, code_dim, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

    def forward(self, x, type='full'):
        '''type=[full, encode, decode]'''
        if type=='full':
            x_out, loss, perplexity = self.vqvae(x)
            return x_out, loss, perplexity
        elif type=='encode':
            b, t, c = x.size()
            quants = self.vqvae.encode(x) # (N, T)
            return quants
        elif type=='decode':
            x_out = self.vqvae.forward_decoder(x)
            return x_out
        else:
            raise ValueError(f'Unknown "{type}" type')

class MMM_DiT(nn.Module):
    def __init__(self,
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
                dit_embed_dim=256,
                dit_num_heads=8,
                dit_mlp_ratio=4,
                dit_depth=2,
                dit_drop_out_prob=0.1,
                quantizer='ema_reset',
                norm=None):
        
        super().__init__()
        
        self.vqvae = VQVAE_251(
            output_dim=output_dim,
            nb_code=nb_code,
            code_dim=code_dim,
            output_emb_width=output_emb_width,
            down_t=down_t,
            stride_t=stride_t,
            width=width,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation,
            quantizer=quantizer,
            norm=norm)
        
        self.refiner = DiTRefiner(
            in_dim=output_dim,
            embed_dim=dit_embed_dim,
            num_heads=dit_num_heads,
            mlp_ratio=dit_mlp_ratio,
            depth=dit_depth,
            drop_out_prob=dit_drop_out_prob,
        )
    
    @torch.no_grad()
    def vqvae_forward(self, x):
        x_out, loss, perplexity = self.vqvae(x)
        return x_out, loss, perplexity
    
    def refiner_forward(self, x_t, t, vae_out):
        x_out = self.refiner(x_t, t, vae_out)
        return x_out
    
    def refiner_forward_with_cfg(self, x_t, t, vae_out, cfg_scale):
        x_out = self.refiner.forward_with_cfg(x_t, t, vae_out, cfg_scale)
        return x_out
    
    def forward(self, x_t, t, x_origin):
        vae_out, loss, perplexity = self.vqvae_forward(x_origin)
        refiner_out = self.refiner_forward(x_t, t, vae_out)
        return refiner_out, (loss, perplexity, vae_out)
    
    def forward_with_cfg(self, x_t, t, x_origin, cfg_scale):
        vae_out, loss, perplexity = self.vqvae_forward(x_origin)
        refiner_out = self.refiner_forward_with_cfg(x_t, t, vae_out, cfg_scale)
        return refiner_out, (loss, perplexity, vae_out)