import torch
import argparse

def get_args():
    p = argparse.ArgumentParser()
    
    p.add_argument('--random_seed', type=int, default=2022)

    # Learning hyperparameters
    p.add_argument('-epochs', type=int, default=100)
    p.add_argument('-no_cuda', type=str, default=False)
    p.add_argument('-lr_scheduler', type=str,default="WarmUpDefault", 
                        help="WarmUpDefault, SGDR")
    p.add_argument('-lr_WarmUpSteps', type=int,default=8000, 
                        help="only for WarmUpDefault")
    p.add_argument('-lr', type=float, default=0.0003)
    p.add_argument('-lr_beta1', type=float, default=0.9)
    p.add_argument('-lr_beta2', type=float, default=0.98)
    p.add_argument('-lr_eps', type=float, default=1e-9)

    # KL Annealing
    p.add_argument('-use_KLA', type=bool, default=True)
    p.add_argument('-KLA_ini_beta', type=float, default=0.02)
    p.add_argument('-KLA_inc_beta', type=float, default=0.02)
    p.add_argument('-KLA_max_beta', type=float, default=1.0)
    p.add_argument('-KLA_beg_epoch', type=int,
                        default=1)  # KL annealing begin

    # Network sturucture
    p.add_argument('-use_cond2dec', type=bool, default=False)
    p.add_argument('-use_cond2lat', type=bool, default=True)
    
    p.add_argument('-latent_dim', type=int, default=128)
    p.add_argument('-cond_dim', type=int, default=3)
    p.add_argument('-d_model', type=int, default=512)
    p.add_argument('-n_layers', type=int, default=8)
    p.add_argument('-heads', type=int, default=8)
    p.add_argument('-dropout', type=int, default=0.2)
    p.add_argument('-batch_size', type=int, default=512)
    p.add_argument('-max_len', type=int, default=120)  # max 80
    
    p.add_argument('-k', type=int, default=10)

    # History
    p.add_argument('-verbose', type=bool, default=False)
    p.add_argument('-save_folder_name', type=str, default='saved_model')
    p.add_argument('-print_model', type=bool, default=False)
    p.add_argument('-printevery', type=int, default=5)
    
    # must be a multiple of printevery
    p.add_argument('-create_valset', action='store_true')
    p.add_argument('-checkpoint', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    return p.parse_args()
    
    