import random
import torch
import torch.nn as nn
from .encdec import Encoder, Decoder
from .residual_vq import ResidualVQ
    
class RVQVAE(nn.Module):
    def __init__(self,
                 args,
                 input_width=263,
                 nb_code=1024,
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
        assert output_emb_width == code_dim
        self.code_dim = code_dim
        self.num_code = nb_code
        # self.quant = args.quantizer
        self.encoder = Encoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Decoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        rvqvae_config = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code,
            'code_dim':code_dim, 
            'args': args,
        }
        self.quantizer = ResidualVQ(**rvqvae_config)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x): #
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        # print(x_encoder.shape)
        code_idx, all_codes = self.quantizer.quantize(x_encoder, return_latent=True)
        # print(code_idx.shape)
        # code_idx = code_idx.view(N, -1)
        # (N, T, Q)
        # print()
        return code_idx, all_codes
        # all_codes (num_layers, B, dim, N)
        
        # this is compatible with base tokens, then (b, n, 1)
    
    def forward(self, x, return_z=False, only_base = False, cb_replace_prob=0.0, cb_replace_topk=3):
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)

        ## quantization
        # x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5,
        #                                                                 force_dropout_index=0) #TODO hardcode
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5)
        if cb_replace_prob > 0:
            # Access codebook embeddings
            codebooks = self.quantizer.codebooks  # Shape: [num_levels, num_embeddings, embedding_dim]
            num_levels, num_embeddings, embedding_dim = codebooks.shape

            B, D, N = x_encoder.shape
            _, _, L = code_idx.shape  # Number of codebooks (levels)

            x_quantized = torch.zeros_like(x_encoder)

            for l in range(L):
                # Get codebook for level l
                codebook_l = codebooks[l]  # Shape: [num_embeddings, D]
                
                # Decide whether to replace each codebook index (before computing distances)
                replace_mask = torch.bernoulli(cb_replace_prob * torch.ones(B, N, device=x_encoder.device)).bool()  # (B, N)

                if replace_mask.any():
                    # Only compute distances for tokens that need replacement
                    replace_indices = torch.nonzero(replace_mask, as_tuple=True)  # Get indices where replacements will happen
                    x_to_replace = x_encoder[replace_indices[0], :, replace_indices[1]]  # Shape: [num_replacements, D]

                    # Compute distances between the x_to_replace and codebook embeddings
                    x_expanded = x_to_replace.view(-1, 1, D)  # Shape: [num_replacements, 1, D]
                    codebook_expanded = codebook_l.unsqueeze(0)  # Shape: [1, num_embeddings, D]

                    distances = torch.sum((x_expanded - codebook_expanded) ** 2, dim=-1)  # Shape: [num_replacements, num_embeddings]

                    # Get top-k closest codebook indices for the points being replaced
                    _, topk_indices = torch.topk(distances, k=cb_replace_topk, largest=False)  # Shape: [num_replacements, k]

                    # Randomly select one of the top-k indices for each replacement
                    random_choices = torch.randint(0, cb_replace_topk, (topk_indices.shape[0],), device=x_encoder.device)
                    new_code_idx = topk_indices[torch.arange(topk_indices.shape[0]), random_choices]

                    # Update the codebook indices for the points being replaced
                    code_idx[replace_indices[0], replace_indices[1], l] = new_code_idx

                # Reconstruct x_quantized
                code_idx_l = code_idx[:, :, l]
                code_embeddings = codebook_l[code_idx_l]  # Shape: [B, N, D]
                x_quantized += code_embeddings.permute(0, 2, 1)
                
        if only_base:
            base_code_idx = code_idx[:, :, 0].unsqueeze(-1)
            x_out = self.forward_decoder(base_code_idx)
            return x_out, commit_loss, perplexity
        # print(code_idx[0, :, 1])
        ## decoder
        x_out = self.decoder(x_quantized)
        # x_out = self.postprocess(x_decoder)
        if return_z:
            return x_out, commit_loss, perplexity, x_quantized
        return x_out, commit_loss, perplexity

    def forward_decoder(self, x): #(b,n,q) # q is the quantize dim (layers)
        # this is compatible with only base tokens then (b, n, 1)
        # q can be 1 if we are using only base tokens
        x_d = self.quantizer.get_codes_from_indices(x)
        # x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x = x_d.sum(dim=0).permute(0, 2, 1) # this is summing all the residuals

        # decoder
        x_out = self.decoder(x)
        # x_out = self.postprocess(x_decoder)
        return x_out

class LengthEstimator(nn.Module):
    def __init__(self, input_size, output_size):
        super(LengthEstimator, self).__init__()
        nd = 512
        self.output = nn.Sequential(
            nn.Linear(input_size, nd),
            nn.LayerNorm(nd),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(0.2),
            nn.Linear(nd, nd // 2),
            nn.LayerNorm(nd // 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(0.2),
            nn.Linear(nd // 2, nd // 4),
            nn.LayerNorm(nd // 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(nd // 4, output_size)
        )

        self.output.apply(self.__init_weights)

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, text_emb):
        return self.output(text_emb)