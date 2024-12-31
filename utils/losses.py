import torch
import torch.nn as nn

# code adopted from MMM https://github.com/exitudio/MMM/blob/7dcc7965379adbbb230e3782889fb5294a8885cc/utils/losses.py
# this is the reconstruction loss for the motion prediction
class ReconLoss(nn.Module):
    def __init__(self, recon_loss, nb_joints):
        super(ReconLoss, self).__init__()
        
        if recon_loss == 'l1': 
            self.Loss = torch.nn.L1Loss()
        elif recon_loss == 'l2' : 
            self.Loss = torch.nn.MSELoss()
        elif recon_loss == 'l1_smooth' : 
            self.Loss = torch.nn.SmoothL1Loss()
        
        # 4 global motion associated to root
        # 12 local motion (3 local xyz, 3 vel xyz, 6 rot6d)
        # 3 global vel xyz
        # 4 foot contact
        self.nb_joints = nb_joints
        self.motion_dim = 4 + (nb_joints - 1) * 12  + 3 + 4
        
    def forward(self, motion_pred, motion_gt) : # for all the motion dimension
        loss = self.Loss(motion_pred[..., : self.motion_dim], motion_gt[..., :self.motion_dim])
        return loss
    
    def forward_joint(self, motion_pred, motion_gt) :  # only for the joint positions
        loss = self.Loss(motion_pred[..., 4 : (self.nb_joints - 1) * 3 + 4], motion_gt[..., 4 : (self.nb_joints - 1) * 3 + 4])
        return loss
    
def calc_recon_loss(pred, target, recon_loss = 'l1_smooth', only_joint=False):
    if recon_loss == 'l1': 
        reconstruction_loss = nn.L1Loss()(pred, target)
    elif recon_loss == 'l2' :
        reconstruction_loss = nn.MSELoss()(pred, target)
    elif recon_loss == 'l1_smooth' :
        reconstruction_loss = nn.SmoothL1Loss()(pred, target)
    elif recon_loss == 'nll':
        reconstruction_loss = torch.abs(target.contiguous() - pred.contiguous())
        reconstruction_loss = reconstruction_loss / torch.exp(logvar) + logvar
        reconstruction_loss = torch.sum(reconstruction_loss) / reconstruction_loss.shape[0]
    return reconstruction_loss

    
def calc_kld_loss(mu, logvar):
    KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)) 
    return KLD

def calc_vae_loss(pred,target,mu, logvar,recon_weight=1, kl_weight=0.0001, recon_loss = 'l1_smooth'):                            
    """ function that computes the various components of the VAE loss """
    if recon_loss == 'l1': 
        reconstruction_loss = nn.L1Loss()(pred, target)
    elif recon_loss == 'l2' :
        reconstruction_loss = nn.MSELoss()(pred, target)
    elif recon_loss == 'l1_smooth' :
        reconstruction_loss = nn.SmoothL1Loss()(pred, target)
    elif recon_loss == 'nll':
        reconstruction_loss = torch.abs(target.contiguous() - pred.contiguous())
        reconstruction_loss = reconstruction_loss / torch.exp(logvar) + logvar
        reconstruction_loss = torch.sum(reconstruction_loss) / reconstruction_loss.shape[0]
        
    KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)) 
    # this is batchsize, time axis invariant

    return recon_weight * reconstruction_loss + kl_weight * KLD, recon_weight *reconstruction_loss,kl_weight * KLD

def calc_vq_loss(pred, target, quant_loss, quant_loss_wight,recon_weight=1.0, recon_loss = 'l1_smooth'):
    """ function that computes the various components of the VQ loss """
    if recon_loss == 'l1': 
        reconstruction_loss = nn.L1Loss()(pred, target)
    elif recon_loss == 'l2' :
        reconstruction_loss = nn.MSELoss()(pred, target)
    elif recon_loss == 'l1_smooth' :
        reconstruction_loss = nn.SmoothL1Loss()(pred, target)
    ## loss is VQ reconstruction + weighted pre-computed quantization loss
    return quant_loss.mean()*quant_loss_wight + reconstruction_loss*recon_weight, reconstruction_loss