from __future__ import annotations

import math
from copy import deepcopy
from collections import namedtuple
from typing import Tuple, List, Literal, Callable
from ema_pytorch import EMA
import torch
from torch import Tensor
from torch import nn, pi, from_numpy
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from torchdiffeq import odeint

import torchvision
from torchvision.utils import save_image
from torchvision.models import VGG16_Weights

import einx
from einops import einsum, reduce, rearrange, repeat
from einops.layers.torch import Rearrange

from scipy.optimize import linear_sum_assignment

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

# tensor helpers

def append_dims(t, ndims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * ndims))

# normalizing helpers

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# noise schedules

def cosmap(t):
    # Algorithm 21 in https://arxiv.org/abs/2403.03206
    return 1. - (1. / (torch.tan(pi / 2 * t) + 1))

class MSELoss(Module):
    def forward(self, pred, target, padding_mask=None,**kwargs):
        if padding_mask is None:
            return F.mse_loss(pred, target)
        else:
            loss = F.mse_loss(pred, target, reduction = 'none')
            mask = ~padding_mask # padding mask is True for padding, we want to mask out padding
            mask = mask.unsqueeze(-1)
            mask = mask.float()
            masked_loss = (loss * mask).mean(dim=-1).sum()/ (mask.sum() + 1e-8)
            return masked_loss


LossBreakdown = namedtuple('LossBreakdown', ['total', 'main', 'data_match', 'velocity_match'])


class RectifiedFlowDecoder(Module):
    def __init__(
        self,
        model: dict | Module,
        time_cond_kwarg: str | None = 'times', # defualt is times
        odeint_kwargs: dict = dict(
            atol = 1e-5,
            rtol = 1e-5,
            method = 'midpoint'
        ),
        predict: Literal['flow', 'noise'] = 'flow', # default is flow
        loss_fn: Literal[
            'mse',
            'pseudo_huber',
            'pseudo_huber_with_lpips'
        ] | Module = 'mse', # default is mse
        noise_schedule: Literal[ # default is identity
            'cosmap'
        ] | Callable = identity, 
        loss_fn_kwargs: dict = dict(),
        ema_update_after_step: int = 100, # onlu used when use_consistency is True
        ema_kwargs: dict = dict(), # only used when use_consistency is True
        data_shape: Tuple[int, ...] | None = None,
        immiscible = False,
        use_consistency = False,
        consistency_decay = 0.9999,
        consistency_velocity_match_alpha = 1e-5,
        consistency_delta_time = 1e-3,
        consistency_loss_weight = 1.,
        data_normalize_fn = None, # default if identity
        data_unnormalize_fn = None, # default if identity
        clip_during_sampling = False,
        clip_values: Tuple[float, float] = (-1., 1.),
        clip_flow_during_sampling = None, # this seems to help a lot when training with predict epsilon, at least for me
        clip_flow_values: Tuple[float, float] = (-3., 3)
    ):
        super().__init__()


        self.net = model
        self.time_cond_kwarg = time_cond_kwarg # whether the model is to be conditioned on the times

        # objective - either flow or noise (proposed by Esser / Rombach et al in SD3)
        self.predict = predict

        # automatically default to a working setting for predict epsilon
        clip_flow_during_sampling = default(clip_flow_during_sampling, predict == 'noise')

        # loss fn
        if loss_fn == 'mse':
            loss_fn = MSELoss()
        elif not isinstance(loss_fn, Module):
            raise ValueError(f'unknown loss function {loss_fn}')

        self.loss_fn = loss_fn

        # noise schedules
        if noise_schedule == 'cosmap':
            noise_schedule = cosmap
        elif not callable(noise_schedule):
            raise ValueError(f'unknown noise schedule {noise_schedule}')

        self.noise_schedule = noise_schedule
        # sampling
        self.odeint_kwargs = odeint_kwargs
        self.data_shape = data_shape

        # clipping for epsilon prediction
        self.clip_during_sampling = clip_during_sampling
        self.clip_flow_during_sampling = clip_flow_during_sampling

        self.clip_values = clip_values
        self.clip_flow_values = clip_flow_values

        # consistency flow matching

        self.use_consistency = use_consistency
        self.consistency_decay = consistency_decay
        self.consistency_velocity_match_alpha = consistency_velocity_match_alpha
        self.consistency_delta_time = consistency_delta_time
        self.consistency_loss_weight = consistency_loss_weight

        if use_consistency:
            self.ema_model = EMA(
                model,
                beta = consistency_decay,
                update_after_step = ema_update_after_step,
                include_online_model = False,
                **ema_kwargs
            )

        # immiscible diffusion paper, will be removed if does not work

        self.immiscible = immiscible

        # normalizing fn

        self.data_normalize_fn = default(data_normalize_fn, identity)
        self.data_unnormalize_fn = default(data_unnormalize_fn, identity)

    @property
    def device(self):
        return next(self.net.parameters()).device

    def predict_flow(self, model: Module, noised, *, times, y, padding_mask = None, text_embedding = None, eps = 1e-10):
        """
        returns the model output as well as the derived flow, depending on the `predict` objective
        """

        batch = noised.shape[0]

        # prepare maybe time conditioning for model

        model_kwargs = dict()
        time_kwarg = self.time_cond_kwarg

        if exists(time_kwarg):
            times = rearrange(times, '... -> (...)')

            if times.numel() == 1:
                times = repeat(times, '1 -> b', b = batch)

            model_kwargs.update(**{time_kwarg: times})
        model_kwargs.update(y=y) 
        model_kwargs.update(padding_mask = padding_mask)
        model_kwargs.update(text_embedding = text_embedding)

        output = self.net(noised, **model_kwargs)

        # depending on objective, derive flow

        if self.predict == 'flow':
            flow = output

        elif self.predict == 'noise':
            noise = output
            padded_times = append_dims(times, noised.ndim - 1)

            flow = (noised - noise) / padded_times.clamp(min = eps)

        else:
            raise ValueError(f'unknown objective {self.predict}')

        return output, flow

    @torch.no_grad()
    def sample(
        self,
        y, # the data to be conditioned on
        batch_size = 1,
        steps = 16,
        noise = None,
        padding_mask = None,
        text_embedding = None,
        data_shape: Tuple[int, ...] | None = None,
        use_ema: bool = False,
        **model_kwargs
    ):
        use_ema = default(use_ema, self.use_consistency)
        assert not (use_ema and not self.use_consistency), 'in order to sample from an ema model, you must have `use_consistency` turned on'

        model = self.ema_model if use_ema else self.net

        was_training = self.training
        self.eval()

        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), 'you need to either pass in a `data_shape` or have trained at least with one forward'

        # clipping still helps for predict noise objective
        # much like original ddpm paper trick

        maybe_clip = (lambda t: t.clamp_(*self.clip_values)) if self.clip_during_sampling else identity

        maybe_clip_flow = (lambda t: t.clamp_(*self.clip_flow_values)) if self.clip_flow_during_sampling else identity

        # ode step function

        def ode_fn(t, x):
            x = maybe_clip(x)

            _, flow = self.predict_flow(model, x, times = t, y=y, padding_mask=padding_mask, text_embedding=text_embedding, **model_kwargs)

            flow = maybe_clip_flow(flow)

            return flow

        # start with random gaussian noise - y0

        noise = default(noise, torch.randn((batch_size, *data_shape), device = self.device))

        # time steps

        times = torch.linspace(0., 1., steps, device = self.device)

        # ode

        trajectory = odeint(ode_fn, noise, times, **self.odeint_kwargs)

        sampled_data = trajectory[-1]

        self.train(was_training)

        return self.data_unnormalize_fn(sampled_data)

    def forward(
        self,
        data,
        y, 
        padding_mask = None,
        text_embedding = None,
        noise: Tensor | None = None,
        return_loss_breakdown = False,
        **model_kwargs
    ):
        batch, *data_shape = data.shape

        data = self.data_normalize_fn(data)

        self.data_shape = default(self.data_shape, data_shape)

        # x0 - gaussian noise, x1 - data

        noise = default(noise, torch.randn_like(data))

        # maybe immiscible flow

        if self.immiscible:
            cost = torch.cdist(data.flatten(1), noise.flatten(1))
            _, reorder_indices = linear_sum_assignment(cost.cpu())
            noise = noise[from_numpy(reorder_indices).to(cost.device)]

        # times, and times with dimension padding on right

        times = torch.rand(batch, device = self.device)
        padded_times = append_dims(times, data.ndim - 1)

        # time needs to be from [0, 1 - delta_time] if using consistency loss

        if self.use_consistency:
            padded_times *= 1. - self.consistency_delta_time

        def get_noised_and_flows(model, t, y, padding_mask = None, text_embedding= None):

            # maybe noise schedule

            t = self.noise_schedule(t)

            # Algorithm 2 in paper
            # linear interpolation of noise with data using random times
            # x1 * t + x0 * (1 - t) - so from noise (time = 0) to data (time = 1.)

            noised = t * data + (1. - t) * noise

            # the model predicts the flow from the noised data

            flow = data - noise

            model_output, pred_flow = self.predict_flow(model, noised, times = t, y=y, padding_mask = padding_mask, text_embedding = text_embedding)

            # predicted data will be the noised xt + flow * (1. - t)

            pred_data = noised + pred_flow * (1. - t)

            return model_output, flow, pred_flow, pred_data

        # getting flow and pred flow for main model

        output, flow, pred_flow, pred_data = get_noised_and_flows(self.net, padded_times, y=y, padding_mask = padding_mask, text_embedding = text_embedding)

        # if using consistency loss, also need the ema model predicted flow

        if self.use_consistency:
            delta_t = self.consistency_delta_time
            ema_output, ema_flow, ema_pred_flow, ema_pred_data = get_noised_and_flows(self.ema_model, padded_times + delta_t)

        # determine target, depending on objective

        if self.predict == 'flow':
            target = flow
        elif self.predict == 'noise':
            target = noise
        else:
            raise ValueError(f'unknown objective {self.predict}')

        # losses

        main_loss = self.loss_fn(output, target, padding_mask=padding_mask, pred_data = pred_data, times = times, data = data)

        consistency_loss = data_match_loss = velocity_match_loss = 0.

        if self.use_consistency:
            # consistency losses from consistency fm paper - eq (6) in https://arxiv.org/html/2407.02398v1

            data_match_loss = F.mse_loss(pred_data, ema_pred_data)
            velocity_match_loss = F.mse_loss(pred_flow, ema_pred_flow)

            consistency_loss = data_match_loss + velocity_match_loss * self.consistency_velocity_match_alpha

        # total loss

        total_loss = main_loss + consistency_loss * self.consistency_loss_weight

        if not return_loss_breakdown:
            return total_loss

        # loss breakdown

        return total_loss, LossBreakdown(total_loss, main_loss, data_match_loss, velocity_match_loss)