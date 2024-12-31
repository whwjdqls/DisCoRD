import torch
import torch.nn as nn

# from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?
from discriminator import patchdisc 
from .util import ActNorm, hinge_d_loss, vanilla_d_loss

class VAELoss(nn.Module):
    def __init__(self,  kl_weight=1.0, recon_weight=1.0, recon_loss='l2'):
        super().__init__()
        self.kl_weight = kl_weight
        self.recon_weight = recon_weight
        if recon_loss == 'l1': 
            self.recon_loss = torch.nn.L1Loss()
        elif recon_loss == 'l2' : 
            self.recon_loss = torch.nn.MSELoss()
        elif recon_loss == 'l1_smooth' : 
            self.recon_loss = torch.nn.SmoothL1Loss()
        else:
            raise ValueError("Invalid reconstruction loss")
            
    def forward(self, inputs, reconstructions, posteriors, split="train"):
        rec_loss = self.recon_loss(inputs.contiguous(),reconstructions.contiguous())
        kl_loss = posteriors.kl()
        # kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        kl_loss = torch.mean(kl_loss)
        loss = self.kl_weight * kl_loss + self.recon_weight * rec_loss
        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/kl_loss".format(split): kl_loss.detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
                "{}/mean".format(split): posteriors.mean.detach().mean(),
                "{}/std".format(split): posteriors.std.detach().mean(),
                }
        return loss, log

class VAELosswithPatchDisc(nn.Module):
    def __init__(self, disc_start, kl_weight=1.0, recon_weight=1.0,
                 disc_num_layers=3, fist_conv_channels=256, disc_in_channels=263, disc_factor=1.0, disc_weight=1.0,
                 use_actnorm=False, disc_loss="hinge"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.recon_weight = recon_weight
        if recon_loss == 'l1': 
            self.recon_loss = torch.nn.L1Loss()
        elif recon_loss == 'l2' : 
            self.recon_loss = torch.nn.MSELoss()
        elif recon_loss == 'l1_smooth' : 
            self.recon_loss = torch.nn.SmoothL1Loss()
        else:
            raise ValueError("Invalid reconstruction loss")


        self.discriminator = patchdisc.NLayerDiscriminator1D(input_nc=disc_in_channels,
                                                 ndf=fist_conv_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight


    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, split="train",weights=None):
        rec_loss = self.recon_loss(inputs.contiguous() - reconstructions.contiguous())
        kl_loss = posteriors.kl()
        kl_loss = torch.mean(kl_loss)
        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            logits_fake = self.discriminator(reconstructions.contiguous())
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:# the original implementation uses perceptual loss(nll loss) can we just use the recon loss?
                    d_weight = self.calculate_adaptive_weight(rec_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training # if not training use d_weight = 0
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = self.recon_weight * rec_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean()
                   "{}/kl_loss".format(split): kl_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                    "{}/mean".format(split): posteriors.mean.detach().mean(),
                    "{}/std".format(split): posteriors.std.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log

# this is the original loss from LDM
class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update

            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))


            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log