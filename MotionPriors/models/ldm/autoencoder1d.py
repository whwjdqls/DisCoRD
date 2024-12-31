# Adopted from LDM's KL-VAE: https://github.com/CompVis/latent-diffusion
import torch
import torch.nn as nn

import numpy as np


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


class Upsample1D(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv1d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample1D(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv1d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock1D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv1d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock1D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv1d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv1d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv1d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv1d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, t = q.shape  # Input shape is (batch_size, channels, sequence_length)
        q = q.permute(0, 2, 1)  # q: (b, t, c)
        k = k.permute(0, 2, 1)  # k: (b, t, c)
        v = v.permute(0, 2, 1)  # v: (b, t, c)

        # Compute attention scores
        scores = torch.bmm(q, k.transpose(1, 2)) * (int(c) ** (-0.5))  # scores: (b, t, t)
        attention = torch.nn.functional.softmax(scores, dim=-1)  # attention: (b, t, t)

        # Attend to values
        h_ = torch.bmm(attention, v)  # h_: (b, t, c)
        h_ = h_.permute(0, 2, 1)  # h_: (b, c, t)

        h_ = self.proj_out(h_)

        return x + h_

# Original code for reference:
# def forward(self, x):
#     h_ = x
#     h_ = self.norm(h_)
#     q = self.q(h_)
#     k = self.k(h_)
#     v = self.v(h_)
#
#     # compute attention
#     b, c, h, w = q.shape  # b, c, h, w = q.shape
#     q = q.reshape(b, c, h * w)
#     q = q.permute(0, 2, 1)  # q: (b, hw, c)
#     k = k.reshape(b, c, h * w)  # k: (b, c, hw)
#     w_ = torch.bmm(q, k)  # w_: (b, hw, hw)
#     w_ = w_ * (int(c) ** (-0.5))
#     w_ = torch.nn.functional.softmax(w_, dim=2)
#
#     # attend to values
#     v = v.reshape(b, c, h * w)
#     w_ = w_.permute(0, 2, 1)  # w_: (b, hw, hw)
#     h_ = torch.bmm(v, w_)  # h_: (b, c, hw)
#     h_ = h_.reshape(b, c, h, w)
#
#     h_ = self.proj_out(h_)
#
#     return x + h_

class Encoder1D(nn.Module):
    def __init__(
        self,
        *,
        ch=128, # base channel size, in_channel -> ch right at the start
        out_ch=3,
        ch_mult=(1, 1, 2, 2, 4), # in_ch -> ch -> ch*ch_mult[0] -> ch*ch_mult[1] -> ...-> ch*ch_mult[-1] -> out_ch
        num_res_blocks=2, # how many resnet blocks per resolution
        attn_resolutions=(16,), # resolutions at which to have attention layers
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=256,
        z_channels=16,
        double_z=True,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv1d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock1D(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock1D(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample1D(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock1D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock1D(block_in)
        self.mid.block_2 = ResnetBlock1D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        # assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder1D(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        out_ch=3,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=256,
        z_channels=16,
        give_pre_end=False,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res)
        print(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        # z to block_in
        self.conv_in = torch.nn.Conv1d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock1D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock1D(block_in)
        self.mid.block_2 = ResnetBlock1D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock1D(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock1D(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample1D(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        # self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )
        return x # (batch_size, embed_dim, seq_len)

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    # dim=[1, 2],
                    dim = 2,
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=2,
                )

    def nll(self, sample, dims=[1, 2]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean


class Autoencoder1DKL(nn.Module):
    def __init__(self, in_ch, ch, out_ch, embed_dim, ch_mult, resolution, attn_res=(16,), use_quant_MLP=False, use_variational=True, ckpt_path=None):
        super().__init__()
        self.encoder = Encoder1D(in_channels=in_ch, ch=ch, out_ch=out_ch, 
                                 resolution=resolution, attn_resolutions=attn_res, 
                                 ch_mult=ch_mult, z_channels=embed_dim)
        self.decoder = Decoder1D(in_channels=in_ch, ch=ch, out_ch=out_ch, 
                                 resolution=resolution, attn_resolutions=(), # should we use attn_resolution for the ddcoder too?
                                 ch_mult=ch_mult, z_channels=embed_dim)            # LDM VAE does not use attn for the decoder
        self.use_variational = use_variational
        mult = 2 if self.use_variational else 1
        self.use_quant_MLP = use_quant_MLP
        if use_quant_MLP:
            self.mean_MLP = torch.nn.Linear(embed_dim, embed_dim)
            self.logvar_MLP = torch.nn.Linear(embed_dim, embed_dim)
        else:
            self.quant_conv = torch.nn.Conv1d(2 * embed_dim, mult * embed_dim, 1)
            self.post_quant_conv = torch.nn.Conv1d(embed_dim, embed_dim, 1)
        
        self.embed_dim = embed_dim
        self.quant_factor = len(ch_mult) - 1
        self.loss = None # this needs to be set by the training script
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, path):
        sd = torch.load(path, map_location="cpu")["model"]
        msg = self.load_state_dict(sd, strict=False)
        print("Loading pre-trained KL-VAE")
        print("Missing keys:")
        print(msg.missing_keys)
        print("Unexpected keys:")
        print(msg.unexpected_keys)
        print(f"Restored from {path}")

    def encode(self, x):
        # input x: (batch_size, seq_len, in_channels)
        x = x.permute(0, 2, 1).float() # (batch_size, in_channels, seq_len)
        h = self.encoder(x)
        if self.use_quant_MLP:
            h = h.permute(0, 2, 1) # (batch_size, seq_len, embed_dim)
            mean = self.mean_MLP(h) # (batch_size, seq_len, embed_dim)
            logvar = self.logvar_MLP(h) # (batch_size, seq_len, embed_dim)
            moments = torch.cat((mean, logvar), 2) # (batch_size, seq_len, 2*embed_dim)
            moments = moments.permute(0, 2, 1) # (batch_size, 2*embed_dim, seq_len)
        else:
            moments = self.quant_conv(h)
        if not self.use_variational:
            moments = torch.cat((moments, torch.ones_like(moments)), 1)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        if not self.use_quant_MLP:
            z = self.post_quant_conv(z)
        dec = self.decoder(z) # (batch_size, out_channels, seq_len)
        dec = dec.permute(0, 2, 1) # (batch_size, seq_len, out_channels)
        return dec
    
    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior
    
    def get_last_layer(self):
        return self.decoder.conv_out.weight
    
    def training_step(self, inputs, optimizer_idx=0, global_step=0):
        assert self.loss is not None, "Loss function not defined"
        # inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        if self.loss.__class__.__name__ == "VAELoss": # optimizer_idx is not used
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, split="train")
            return aeloss, log_dict_ae
        
        elif self.loss.__class__.__name__ == "VAELosswithPatchDisc":
            if optimizer_idx == 0:
                # train encoder+decoder+logvar
                aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, global_step,
                                                last_layer=self.get_last_layer(), split="train")
                return aeloss, log_dict_ae

            if optimizer_idx == 1:
                # train the discriminator
                discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, global_step,
                                                    last_layer=self.get_last_layer(), split="train")
                return discloss, log_dict_disc
            
    
    def validation_step(self, inputs):
        assert self.loss is not None, "Loss function not defined"
        # inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, split="val")
        # self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        # self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return aeloss, log_dict_ae

    # def forward(self, inputs, disable=True, train=True, optimizer_idx=0):
    #     if train:
    #         return self.training_step(inputs, disable, optimizer_idx)
    #     else:
    #         return self.validation_step(inputs, disable)
