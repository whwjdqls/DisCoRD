import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_models import LinearEmbedding, PositionalEncoding, PositionEmbedding, Transformer
from .quantizer import *
from .Resnet import Resnet1D
from .TransformerMasking import *


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        in_dim,  # dimension of inputs
        hidden_size,
        dim_feedforward,
        num_attention_heads,
        num_hidden_layers,
        max_len,
        quant_factor,
        quant_sequence_length,  # if we do not use positional encoding, this does not matter
        pos_encoding=None,
        temporal_bias="alibi_future",
    ):
        super().__init__()

        self.quant_factor = quant_factor
        self.in_dim = in_dim
        self.hidden_size = hidden_size

        layers = [
            nn.Sequential(
                nn.Conv1d(in_dim, hidden_size, 5, stride=2, padding=2, padding_mode="replicate"),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm1d(hidden_size),
            )
        ]
        for _ in range(1, quant_factor):
            layers += [
                nn.Sequential(
                    nn.Conv1d(hidden_size, hidden_size, 5, stride=1, padding=2, padding_mode="replicate"),
                    nn.LeakyReLU(0.2, True),
                    nn.BatchNorm1d(hidden_size),
                    nn.MaxPool1d(2),
                )
            ]
        self.squasher = nn.Sequential(*layers)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size,  # 128
            nhead=num_attention_heads,  #
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder_transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_hidden_layers)
        # define positional encoding. if false, None
        if pos_encoding == "learned":
            self.encoder_pos_encoding = PositionEmbedding(  # for L2L use learnable PE but for FLINT use ALIBI PE
                quant_sequence_length, hidden_size
            )
        elif pos_encoding == "sin":
            self.encoder_pos_encoding = PositionalEncoding(hidden_size)
        else:
            self.encoder_pos_encoding = None

        # Temperal bias
        if temporal_bias == "alibi_future":
            self.attention_mask = init_alibi_biased_mask_future(num_attention_heads, max_len)
        else:
            self.attention_mask = None

        self.encoder_linear_embedding = LinearEmbedding(hidden_size, hidden_size)

    def forward(self, inputs):
        ## downdample into path-wise length seq before passing into transformer
        dummy_mask = {"max_mask": None, "mask_index": -1, "mask": None}
        inputs = self.squasher(inputs.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # (BS, T/q, 128)->(BS , 128, T/q) -> (BS, T/q, 128)
        encoder_features = self.encoder_linear_embedding(inputs)

        if self.encoder_pos_encoding is not None:
            decoder_features = self.encoder_pos_encoding(encoder_features)

        # add attention mask bias (if any)
        mask = None
        B, T = encoder_features.shape[:2]
        if self.attention_mask is not None:
            mask = self.attention_mask[:, :T, :T].clone().detach().to(device=encoder_features.device)
            if mask.ndim == 3:  # the mask's first dimension needs to be num_head * batch_size
                mask = mask.repeat(B, 1, 1)

        encoder_features = self.encoder_transformer(encoder_features, mask=mask)
        return encoder_features


class TransformerDecoder(nn.Module):
    """Decoder class for VQ-VAE with Transformer backbone"""

    def __init__(
        self,
        in_dim,  # input dimension of latent space
        out_dim,  # output dimension of the reconstructed output
        hidden_size,  # input size of transformer
        dim_feedforward,  # feedforward dimension of transformer
        num_attention_heads,  # number of attention heads
        num_hidden_layers,  # number of hidden layers
        max_len,  # maximum length of the sequence
        quant_factor,  # quantization factor
        sequence_length,  # sequence length for positional encoding
        pos_encoding=None,
        temporal_bias="alibi_future",
    ):
        super().__init__()

        self.expander = nn.ModuleList()
        self.expander.append(
            nn.Sequential(
                # https://github.com/NVIDIA/tacotron2/issues/182
                # After installing torch, please make sure to modify the site-packages/torch/nn/modules/conv.py
                # file by commenting out the self.padding_mode != 'zeros' line to allow for replicated padding
                # for ConvTranspose1d as shown in the above link ->
                nn.ConvTranspose1d(
                    in_dim, hidden_size, 5, stride=2, padding=2, output_padding=1, padding_mode="zeros"
                ),  # set to zero for now, change latter
                # padding_mode='replicate'),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm1d(hidden_size),
            )
        )
        num_layers = quant_factor  # we never take audio into account for TVAE -> num_layer = 3
        seq_len = sequence_length  # we never take audio into account for TVAE -> seq_len = 32
        for _ in range(1, num_layers):
            self.expander.append(
                nn.Sequential(
                    nn.Conv1d(hidden_size, hidden_size, 5, stride=1, padding=2, padding_mode="replicate"),
                    nn.LeakyReLU(0.2, True),
                    nn.BatchNorm1d(hidden_size),
                )
            )
        # following INFERNO, use nn.TransformerEncoder
        decoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.decoder_transformer = torch.nn.TransformerEncoder(decoder_layer, num_layers=num_hidden_layers)

        # positional Encoding
        if pos_encoding == "learned":
            self.decoder_pos_encoding = PositionEmbedding(seq_len, hidden_size)
        elif pos_encoding == "sin":
            self.decoder_pos_encoding = PositionalEncoding(hidden_size)
        else:  # if self.config['pos_encoding'] == false
            self.decoder_pos_encoding = None

        # Temperal bias
        if temporal_bias == "alibi_future":
            self.attention_mask = init_alibi_biased_mask_future(num_attention_heads, max_len)
        else:
            self.attention_mask = None

        self.decoder_linear_embedding = torch.nn.Linear(hidden_size, hidden_size)
        # smooth layer
        self.cross_smooth_layer = nn.Conv1d(hidden_size, out_dim, 5, padding=2)

        # linear embedding
        self.post_transformer_linear = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs):
        ## upsample into original length seq before passing into transformer
        for i, module in enumerate(self.expander):
            inputs = module(inputs.permute(0, 2, 1)).permute(0, 2, 1)
            if i > 0:
                inputs = inputs.repeat_interleave(2, dim=1)
        decoder_features = self.decoder_linear_embedding(inputs)

        if self.decoder_pos_encoding is not None:
            decoder_features = self.decoder_pos_encoding(decoder_features)

        # add attention mask bias (if any)
        mask = None
        B, T = decoder_features.shape[:2]
        if self.attention_mask is not None:
            mask = self.attention_mask[:, :T, :T].clone().detach().to(device=decoder_features.device)
            if mask.ndim == 3:  # the mask's first dimension needs to be num_head * batch_size
                mask = mask.repeat(B, 1, 1)

        decoder_features = self.decoder_transformer(decoder_features, mask=mask)
        post_transformer_linear_features = self.post_transformer_linear(decoder_features)
        pred_recon = self.cross_smooth_layer(decoder_features.permute(0, 2, 1)).permute(0, 2, 1)
        return pred_recon


class TVQVAE(nn.Module):
    """Temporal VAE using Transformer backbone"""

    def __init__(
        self,
        in_dim=263,  # 53
        hidden_size=512,  # 128
        embed_dim=256,
        n_embed=256,
        dim_feedforward=1024,
        num_attention_heads=8,
        num_hidden_layers=1,
        max_len=400,
        quant_factor=2,
        quant_sequence_length=16,  # for encoder
        sequence_length=64,  # for decoder
        quantizer="ema_reset",
        pos_encoding=None,
        temporal_bias="alibi_future",
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            in_dim=in_dim,
            hidden_size=hidden_size,
            dim_feedforward=dim_feedforward,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            max_len=max_len,
            quant_factor=quant_factor,
            quant_sequence_length=quant_sequence_length,
            pos_encoding=pos_encoding,
            temporal_bias=temporal_bias,
        )

        self.decoder = TransformerDecoder(
            in_dim=hidden_size,
            out_dim=in_dim,
            hidden_size=hidden_size,
            dim_feedforward=dim_feedforward,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            max_len=max_len,
            quant_factor=quant_factor,
            sequence_length=sequence_length,
            pos_encoding=pos_encoding,
            temporal_bias=temporal_bias,
        )

        self.pre_quant = torch.nn.Linear(hidden_size, embed_dim)
        self.post_quant = torch.nn.Linear(embed_dim, hidden_size)

        self.quant_factor = quant_factor

        if quantizer == "ema_reset":
            self.quantizer = QuantizeEMAReset(n_embed, embed_dim)
        elif quantizer == "orig":
            self.quantizer = Quantizer(n_embed, embed_dim, 1.0)
        elif quantizer == "ema":
            self.quantizer = QuantizeEMA(n_embed, embed_dim)
        elif quantizer == "reset":
            self.quantizer = QuantizeReset(n_embed, embed_dim)
        else:
            raise ValueError(f"Model {quantizer} not supported.")

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x):
        N, T, _ = x.shape  # Bs, T, 263
        x_encoder = self.encoder(x)  # BS, T/q, embed_dim
        x_encoder = self.pre_quant(x_encoder)  # BS, T/q, embed_dim
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)
        return code_idx

    def forward(self, x):  # input (Bs, T, 263)
        x_encoder = self.encoder(x)  # (Bs, T/q, embed_dim)
        x_encoder = self.pre_quant(x_encoder)  # BS, T/q, embed_dim
        ## quantization
        x_encoder = x_encoder.permute(0, 2, 1)  # (Bs, embed_dim, T/q) quantizer inputs are always like this
        x_quantized, loss, perplexity = self.quantizer(x_encoder)  # (BS, embed_dim, T/q)
        ## decoder
        x_quantized = x_quantized.permute(0, 2, 1)  # (Bs, T/q, embed_dim)
        x_quantized = self.post_quant(x_quantized)
        x_out = self.decoder(x_quantized)  # (Bs, T/q, 263)

        return x_out, loss, perplexity

    def decode_with_emb(self, quant):  # input (BS, T/q, embed_dim)
        quant = self.post_quant(quant)
        dec = self.decoder(quant)  ## z' --> x
        return dec

    def decode_with_idx(self, x):  # inputs (BS, T)
        x_d = self.quantizer.dequantize(x)  # (BS, T, embed_dim)
        # decoder
        quant = self.post_quant(x_d)
        x_decoder = self.decoder(quant)
        return x_decoder


class Encoder(nn.Module):
    def __init__(
        self,
        input_emb_width=3,
        output_emb_width=512,
        down_t=2,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
    ):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_emb_width=3,
        output_emb_width=512,
        down_t=2,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
    ):
        super().__init__()
        blocks = []

        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv1d(width, out_dim, 3, 1, 1),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class ResTVQVAE(nn.Module):
    """Temporal VqAE using Resnet backbone"""

    def __init__(
        self,
        in_dim=263,  # input_width for momask
        hidden_size=512,  # width for momask
        embed_dim=512,  # embed_dim for momask
        n_embed=512,
        depth=3,
        quant_factor=2,  # down_t
        stride_t=2,
        dilation_growth_rate=3,
        activation="relu",
        quantizer="ema_reset",
        norm=None,
    ):
        super().__init__()

        # self.quant = args.quantizer
        self.encoder = Encoder(
            in_dim,
            embed_dim,
            quant_factor,
            stride_t,
            hidden_size,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=norm,
        )
        self.decoder = Decoder(
            in_dim,
            embed_dim,
            quant_factor,
            stride_t,
            hidden_size,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=norm,
        )

        if quantizer == "ema_reset":
            self.quantizer = QuantizeEMAReset(n_embed, embed_dim)
        elif quantizer == "orig":
            self.quantizer = Quantizer(n_embed, embed_dim, 1.0)
        elif quantizer == "ema":
            self.quantizer = QuantizeEMA(n_embed, embed_dim)
        elif quantizer == "reset":
            self.quantizer = QuantizeReset(n_embed, embed_dim)
        else:
            raise ValueError(f"Model {quantizer} not supported.")

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
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

    def forward(self, x):  # input (BS, T, 263)
        x_in = self.preprocess(x)  # (BS, 263, T)
        x_encoder = self.encoder(x_in)  # (Bs, embed_dim, T/q)
        ## quantization
        x_quantized, loss, perplexity = self.quantizer(x_encoder)  # (Bs, embed_dim, T/q)
        ## decoder
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)
        return x_out, loss, perplexity

    def decode_with_emb(self, quant):  # input (BS, T/q, embed_dim)
        quant = self.preprocess(quant)
        dec = self.decoder(quant)  ## z' --> x
        dec = self.postprcess(dec)
        return dec

    def decode_with_indx(self, x):  # any size (BS, T/q)
        x_d = self.quantizer.dequantize(x)  # (BS, T/q, latent dim)
        x_d = x_d.permute(0, 2, 1).contiguous()  # (BS, latentdim, T/q)
        x_decoder = self.decoder(x_d)  # (Bs, 263, T)
        x_out = self.postprocess(x_decoder)  # (BS, T, 263)
        return x_out


# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = TVQVAE(in_dim=53, hidden_size=128, embed_dim=64, n_embed=512,
#                    dim_feedforward=512, num_attention_heads=8, num_hidden_layers=3,
#                    max_len=1000, quant_factor=3, quant_sequence_length=32,
#                    sequence_length=32, pos_encoding=None, temporal_bias="alibi_future")
#     inputs = torch.randn(2, 53, 64).to(device)
#     print(inputs.shape)
#     model.to(device)
#     print("######### Encoder #########")
#     quant, emb_loss, info = model.encode(inputs)
#     print(quant.shape)
#     print(emb_loss)
#     print(info)
#     outputs = model(inputs)
#     print("######### Forward #########")
#     print(outputs[0].shape)
