import torch
import torch.nn.functional as F


def KLAnnealer(opt, epoch):
    return opt.KLA_ini_beta + opt.KLA_inc_beta * ((epoch + 1) - opt.KLA_beg_epoch)


def loss_function(opt, beta, preds_prop, preds_mol, ys_cond, ys_mol, mu, log_var):
    RCE_mol = F.cross_entropy(
        preds_mol.contiguous().view(-1, preds_mol.size(-1)),
        ys_mol,
        ignore_index=opt.trg_pad,
        reduction='mean'
    )

    KLD = beta * -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    if opt.use_cond2dec == True:
        RCE_prop = F.mse_loss(preds_prop, ys_cond, reduction='mean')
        loss = RCE_mol + RCE_prop + KLD
    else:
        RCE_prop = torch.zeros(1)
        loss = RCE_mol + beta * KLD

    return loss, RCE_mol, RCE_prop, KLD