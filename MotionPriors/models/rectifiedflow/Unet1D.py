import math
from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn
from torch.nn import Module, ModuleList

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

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

        
class Residual(Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class PreNorm(Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(Module):
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, dropout = 0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)
# class LinearAttention(Module):
#     def __init__(self, dim, heads = 4, dim_head = 32):
#         super().__init__()
#         self.identity = nn.Identity()

#     def forward(self, x):
#         return self.identity(x)

class LinearAttention(Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale        

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        return self.to_out(out)

class Attention(Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)

# model

class Unet1DforFlow(Module):
    def __init__(
        self,
        dim=384,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2),
        resnet_per_block = 1,
        # motion_condition=False, # for this to work z_condition must be True
        z_condition = False,
        z_in_dim=512, z_proj_dim=128,
        z_proj_type = "linear",
        channels = 263,
        dropout = 0.,
        use_attention=True, 
        self_condition = False,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4
    ):
        super().__init__()

        # determine dimensions

            
        self.resnet_per_block = resnet_per_block
        self.channels = channels # input channels
    
            
        if z_condition:
            self.init_proj = nn.Conv1d(channels + z_proj_dim, dim, 1)
        else:
            self.init_proj = nn.Conv1d(channels, dim, 1)
            
        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        # time embeddings
        time_dim = dim * 4 # we should make the time_dimension explicit
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        self.z_condition = z_condition
        self.z_proj_type = z_proj_type
        if self.z_condition:
            if z_proj_type == "linear":
                self.z_proj = nn.Conv1d(z_in_dim, z_proj_dim, 1)
            elif z_proj_type == "stack_linear":
                self.z_proj = nn.Sequential(
                    StackLinear(quant_factor=2, seq_first=False, unstack=False), # (B, T, in_dim) -> (B, T//4, in_dim*4)
                    nn.Conv1d(z_in_dim*4, z_in_dim*4,1),
                    StackLinear(quant_factor=2, seq_first=False, unstack=True), # (B, T//4, in_dim*4) -> (B, T, in_dim)
                    nn.Conv1d(z_in_dim, z_proj_dim,1)
                )
                    

        resnet_block = partial(ResnetBlock, time_emb_dim = time_dim, dropout = dropout)

        # layers
        self.use_attention = use_attention 
        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            module = ModuleList([])
            for _ in range(resnet_per_block):
                module.append(resnet_block(dim_in, dim_in))
            if self.use_attention:
                module.append(Residual(PreNorm(dim_in, LinearAttention(dim_in))))
            module.append(Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1))
            self.downs.append(module)


        mid_dim = dims[-1] # mid dimension always has 2 blocks
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        if self.use_attention:
            self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, dim_head = attn_dim_head, heads = attn_heads)))
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            module = ModuleList([])
            for _ in range(resnet_per_block):
                module.append(resnet_block(dim_out + dim_in, dim_out))
            if self.use_attention:
                module.append(Residual(PreNorm(dim_out, LinearAttention(dim_out))))
            module.append(Upsample(dim_out, dim_in) if not is_last else  nn.Conv1d(dim_out, dim_in, 3, padding = 1))
            self.ups.append(module)

        self.out_dim = default(out_dim, channels)

        self.final_res_block = resnet_block(dim * 2, dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

    def forward(self, x, times, padding_mask=None, text_embedding=None, z=None):
        """
        x: tensor of shape (batch, seq_len, channels)
        """
        assert x.shape[-1] == self.channels, f'input channels must be {self.channels}'
        if (self.z_condition is False) and (z is not None):
            raise ValueError(f'z_condition is false but {z} is provided')
        
        assert padding_mask is None and text_embedding is None, 'padding mask and text embedding not supported yet'

        x = rearrange(x, 'b n c -> b c n')
        
        if self.z_condition: 
            assert z is not None, 'z must be provided if z_condition is True'
            z = rearrange(z, 'b n c -> b c n')
            z = self.z_proj(z)
            x = torch.cat([x, z], dim=1)
            
        x = self.init_proj(x) # (b, dim, n)
        r = x.clone()
        t = self.time_mlp(times)
        h = []
        
        for layers in self.downs:
            if self.use_attention:
                *blocks, attn, downsample = layers
            else:
                *blocks, downsample = layers
            
            if self.resnet_per_block == 2:
                block1, block2 = blocks
                x = block1(x, t)
                h.append(x)
            elif self.resnet_per_block == 1:
                block2 = blocks[0]
                
            x = block2(x, t)
            if self.use_attention:
                x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        if self.use_attention:
            x = self.mid_attn(x) + x

        x = self.mid_block2(x, t)


        for layers in self.ups:
            if self.use_attention:
                *blocks, attn, upsample = layers
            else:
                *blocks, upsample = layers
                
            if self.resnet_per_block == 2:
                block1, block2 = blocks
                x = torch.cat((x, h.pop()), dim = 1)
                x = block1(x, t)
                
            elif self.resnet_per_block == 1:
                block2 = blocks[0]
                
            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            if self.use_attention:
                x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        x = rearrange(x, 'b c n -> b n c')
        return x
  