# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

# Code from https://github.com/facebookresearch/DiT/blob/main/models.py


import math

import clip
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp

from .helpers import *


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class StackLinear(nn.Module): #( 128 *2, 4, 128)
    def __init__(self, quant_factor=2, unstack=False, seq_first=True): 
        super().__init__() 
        self.quant_factor = quant_factor
        self.latent_frame_size = 2**quant_factor
        self.unstack = unstack
        self.seq_first = seq_first

    def forward(self, x):
        if self.seq_first:
            B, T, F = x.shape # (BS,64,256)
        else:
            B, F, T = x.shape
            x = x.permute(0, 2, 1)

        if not self.unstack:# stack
            assert T % self.latent_frame_size == 0, "T must be divisible by latent_frame_size"
            T_latent = T // self.latent_frame_size
            F_stack = F * self.latent_frame_size
            x = x.reshape(B, T_latent, F_stack) 
        else: #unstack
            F_stack = F // self.latent_frame_size
            x = x.reshape(B, T * self.latent_frame_size, F_stack)

        if not self.seq_first:
            x = x.permute(0, 2, 1)

        return x

class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered
    
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                    These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiT1DBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=0.1, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, key_padding_mask=None, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1) 
        scaled_attn = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_output, _ = self.attn(scaled_attn,scaled_attn,scaled_attn,key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = x + gate_msa.unsqueeze(1) * attn_output
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer1D(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim: int, max_seq_len: int):
        super(SinusoidalPositionalEmbedding, self).__init__()
        
        # Create a matrix of shape (max_seq_len, embed_dim) for the positional encodings
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        # Apply sine to even indices in the embedding dimension
        pos_embedding = torch.zeros(max_seq_len, embed_dim)
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices in the embedding dimension
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        
        # Register the embedding as a buffer, so it is not considered a learnable parameter
        self.register_buffer('pos_embedding', pos_embedding.unsqueeze(0))

    def forward(self, x):
        # Get the input shape
        B, T, D = x.shape
        # Add the positional encoding to the input (broadcasting over batch size)
        return x + self.pos_embedding[:, :T, :]
    
class DiT1DforFlow(nn.Module):
    """ DiT Refiner for motion prior learning 
    code adapted from https://github.com/facebookresearch/DiT/blob/main/models.py
    """
    def __init__(self, out_dim=263, embed_dim=384, num_heads=8, mlp_ratio=4, depth=6, max_seq_len=200, drop_out_prob=0.1,
                t_embedder="dit",
                use_last_conv=False,
                pos_encoding=None,
                text_condition=False,
                z_condition=False,
                z_proj_type="linear",# "stack_linear
                z_in_dim=512,
                z_proj_dim=256,
                temporal_bias="alibi_future"):
        super().__init__()
        self.in_dim = out_dim
        self.blocks = nn.ModuleList([
            DiT1DBlock(
                hidden_size=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            ) for _ in range(depth)
        ])
        if z_condition:
            self.in_proj = nn.Linear(out_dim + z_proj_dim, embed_dim)
        else:
            self.in_proj = nn.Linear(out_dim, embed_dim)
            
            
        self.out_proj = FinalLayer1D(embed_dim, out_dim)
        
        if use_last_conv:# will this help smoothing?
            self.out_proj = nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(embed_dim, out_dim, kernel_size=1, stride=1, padding=0),
            )
            
        if t_embedder == "dit":
            self.t_embedder = TimestepEmbedder(hidden_size=embed_dim)
        elif t_embedder == "sin":
            self.t_embedder = nn.Sequential(
            LearnedSinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim + 1, embed_dim),
        )
        self.post_norm = nn.LayerNorm(embed_dim)
        self.text_condition = text_condition
        self.z_condition = z_condition
        self.z_proj_type = z_proj_type
        if self.z_condition:
            if z_proj_type == "linear":
                self.z_proj = nn.Linear(z_in_dim, z_proj_dim, bias=True)
            elif z_proj_type == "stack_linear":
                self.z_proj = nn.Sequential(
                    StackLinear(quant_factor=2, seq_first=True, unstack=False), # (B, T, in_dim) -> (B, T//4, in_dim*4)
                    nn.Linear(z_in_dim*4, z_in_dim*4, bias=True),
                    StackLinear(quant_factor=2, seq_first=True, unstack=True), # (B, T//4, in_dim*4) -> (B, T, in_dim)
                    nn.Linear(z_in_dim, z_proj_dim, bias=True)
                )
            
        if text_condition:
            self.text_embed = nn.Linear(512, embed_dim, bias=True)
            self.clip_model = self.load_and_freeze_clip("ViT-B/32")
            
        if pos_encoding == "sinusoidal":
            self.postional_encoding = SinusoidalPositionalEmbedding(embed_dim=embed_dim, max_seq_len=max_seq_len)
        else:
            self.postional_encoding = None
            
        if temporal_bias == "alibi_future":
            self.attn_mask = init_faceformer_biased_mask_future(num_heads, max_seq_len)
        else:
            self.attn_mask = None
            
    def load_and_freeze_clip(self, clip_version):
        clip_model, _ = clip.load(clip_version, device="cpu", jit=False)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model
    
    def encode_text(self, raw_text):
        device = next(self.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text
            
    def forward(self, x, times, padding_mask=None, text_embedding=None, z=None):  
        # x: (B, T, in_dim) t: (B, 1) c: (B, T, in_dim)  (x: noised_x, c: reconed_x)
        t = times
        t = self.t_embedder(t)  # (B, embed_dim)
        
        if self.text_condition:
            text_embedding = self.text_embed(text_embedding) # (BS, embed_dim)
            t = t + text_embedding
        
        if self.z_condition: # TODO
            z = self.z_proj(z)
            x = torch.cat([x, z], dim=-1)
    
        x = self.in_proj(x)  # (B, T, in_dim) -> (B, T, embed_dim)
        
        if self.postional_encoding is not None:
            x = self.postional_encoding(x)
            
        if self.attn_mask is not None:
            attn_mask = make_temporal_mask(x, self.attn_mask)
        else:
            attn_mask = None
            
        for block in self.blocks:
            x = block(x, t, key_padding_mask=padding_mask, attn_mask=attn_mask)  # (B, T, embed_dim * 2)

            
        x = self.out_proj(x, t)
        return x
    