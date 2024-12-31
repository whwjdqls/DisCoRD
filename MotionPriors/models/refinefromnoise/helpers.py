import math

import torch
from einops import pack, unpack


def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def cast_tuple(t):
    return (t,) if not isinstance(t, tuple) else t

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def safe_div(num, den, eps = 1e-5):
    return num / den.clamp(min = eps)

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

def pack_one(t, pattern):
    packed, ps = pack([t], pattern)

    def unpack_one(to_unpack, unpack_pattern = None):
        unpacked, = unpack(to_unpack, ps, default(unpack_pattern, pattern))
        return unpacked

    return packed, unpack_one


def init_faceformer_biased_mask_future(num_heads, max_seq_len, period=1):
    """
    Returns a mask for the self-attention layer. The mask is initialized according to the FaceFormer paper but 
    with not with the future masked out. The biased mask has a perioud, after which it repeats
    The diagonal is filled with 0 and the lower triangle is filled with a linear function of the position. 
    The upper triangle is filled symmetrically with the lower triangle.
    That lowers the attention to the past and the future (the number gets lower the further away from the diagonal it is).
    The biased mask has a period, if larger than 1, the bias is repeated period times before coming to the next value.
    If the period is 1, the mask is the same as the alibi mask.
    """
    slopes = torch.Tensor(get_slopes(num_heads))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    # mask = alibi - torch.flip(alibi, [1, 2])
    mask = alibi + torch.flip(alibi, [1, 2])
    return mask

def get_slopes(n):
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start*ratio**i for i in range(n)]
    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)                   
    else:                                                 
        closest_power_of_2 = 2**math.floor(math.log2(n)) 
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    
def make_temporal_mask(trans_inputs, attention_mask):  # input tensor is (BS, T, 2d_hidden_dim)
    # add attention mask bias (if any)
    mask = None
    B, T = trans_inputs.shape[:2]
    if attention_mask is not None:
        mask = attention_mask[:, :T, :T].clone().detach().to(device=trans_inputs.device)
        if mask.ndim == 3:  # the mask's first dimension needs to be num_head * batch_size
            mask = mask.repeat(B, 1, 1)
    return mask