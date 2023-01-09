import torch
import random
import numpy as np
from torch.autograd import Variable


def save_checkpoint(epoch, model, optimizer):

    if isinstance(model, torch.nn.DataParallel):
        checkpoint = dict(
            epoch=epoch,
            model=model.module.state_dict(),
            optimizer=optimizer.state_dict()
        )
    else:
        checkpoint = dict(
            epoch=epoch,
            model=model.state_dict(),
            optimizer=optimizer.state_dict()
        )

    torch.save(checkpoint, f'weights/{epoch}.pt')


def set_randomness(random_seed: int = 2022):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def nopeak_mask(size, opt):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    if opt.use_cond2dec == True:
        cond_mask = np.zeros((1, opt.cond_dim, opt.cond_dim))
        cond_mask_upperright = np.ones((1, opt.cond_dim, size))
        cond_mask_upperright[:, :, 0] = 0
        cond_mask_lowerleft = np.zeros((1, size, opt.cond_dim))
        upper_mask = np.concatenate([cond_mask, cond_mask_upperright], axis=2)
        lower_mask = np.concatenate([cond_mask_lowerleft, np_mask], axis=2)
        np_mask = np.concatenate([upper_mask, lower_mask], axis=1)
    np_mask = Variable(torch.from_numpy(np_mask) == 0)

    return np_mask.to(opt.device)


def create_masks(src, trg, cond, opt):
    torch.set_printoptions(profile="full")
    src_mask = (src != opt.src_pad).unsqueeze(-2)
    cond_mask = torch.unsqueeze(cond, -2)
    cond_mask = torch.ones_like(cond_mask, dtype=bool)
    src_mask = torch.cat([cond_mask, src_mask], dim=2)

    if trg is not None:
        trg_mask = (trg != opt.trg_pad).unsqueeze(-2)
        if opt.use_cond2dec == True:
            trg_mask = torch.cat([cond_mask, trg_mask], dim=2)
        np_mask = nopeak_mask(trg.size(1), opt)
        trg_mask = trg_mask & np_mask

    else:
        trg_mask = None

    return src_mask, trg_mask
