import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn, pi
from torch.nn import Module, ModuleList

from .helpers import *  # import all helpers


class AdaptiveLayerNorm(Module):
    def __init__(
        self,
        dim,
        dim_condition = None
    ):
        super().__init__()
        dim_condition = default(dim_condition, dim)

        self.ln = nn.LayerNorm(dim, elementwise_affine = False)
        self.to_gamma = nn.Linear(dim_condition, dim, bias = False)
        nn.init.zeros_(self.to_gamma.weight)

    def forward(self, x, *, condition):
        normed = self.ln(x)
        gamma = self.to_gamma(condition)
        return normed * (gamma + 1.)

class LearnedSinusoidalPosEmb(Module):
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



# simple mlp
class MLPAdaLN(Module):
    def __init__(
        self,
        dim_input,
        dim_cond = 16,
        depth = 3,
        width = 1024,
        dropout = 0.
    ):
        super().__init__()
        layers = ModuleList([])

        self.to_time_emb = nn.Sequential(
            LearnedSinusoidalPosEmb(dim_cond),
            nn.Linear(dim_cond + 1, dim_cond),
        )

        for _ in range(depth):
            adaptive_layernorm = AdaptiveLayerNorm(
                dim_input,
                dim_condition = dim_cond
            )

            block = nn.Sequential(
                nn.Linear(dim_input, width),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(width, dim_input)
            )

            block_out_gamma = nn.Linear(dim_cond, dim_input, bias = False)
            nn.init.zeros_(block_out_gamma.weight)

            layers.append(ModuleList([
                adaptive_layernorm,
                block,
                block_out_gamma
            ]))

        self.layers = layers

    def forward(
        self,
        noised,
        *,
        times,
        cond = None,
    ):
        assert noised.ndim == 2

        time_emb = self.to_time_emb(times)
        if cond is not None:
            cond = F.silu(time_emb + cond)
        else:
            cond = F.silu(time_emb)

        denoised = noised

        for adaln, block, block_out_gamma in self.layers:
            residual = denoised
            denoised = adaln(denoised, condition = cond)

            block_out = block(denoised) * (block_out_gamma(cond) + 1.)
            denoised = block_out + residual

        return denoised




