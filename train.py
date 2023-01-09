import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from args import get_args
from data_loader import *
from losses import loss_function
from models.transformer import Transformer
from utils import set_randomness, create_masks, save_checkpoint


def train(opt, model, optimizer, scheduler, train_loader, valid_loader):

    beta = 1
    best_loss = np.inf
    for epoch in range(opt.epochs):
        total_loss = 0
        RCE_mol_loss = 0
        RCE_prop_loss = 0
        KLD_loss = 0

        model.train()
        train_pbar = tqdm(train_loader, total=len(train_loader), ncols=100)
        for i, (src, trg, cond) in enumerate(train_pbar):
            src = src.transpose(0, 1).to(opt.device)
            trg = trg.transpose(0, 1).to(opt.device)
            cond = cond.float().to(opt.device)

            trg_input = trg[:, :-1]

            src_mask, trg_mask = create_masks(src, trg_input, cond, opt)
            preds_prop, preds_mol, mu, log_var, z = model(
                src, trg_input, cond, src_mask, trg_mask)

            preds_mol = preds_mol.contiguous().view(-1, preds_mol.size(-1))
            preds_prop = preds_prop.view(-1, 3)
            ys_mol = trg[:, 1:].contiguous().view(-1)
            cond = cond.view(-1, 3)

            loss, RCE_mol, RCE_prop, KLD = loss_function(
                opt, beta, preds_prop, preds_mol, cond, ys_mol, mu, log_var
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += (loss.item() / opt.batch_size)
            RCE_mol_loss += (RCE_mol.item() / opt.batch_size)
            RCE_prop_loss += (RCE_prop.item() / opt.batch_size)
            KLD_loss += (KLD.item() / opt.batch_size)
            train_pbar.set_postfix({
                'total': total_loss,
                'RCE_mol': RCE_mol_loss,
                'RCE_prop': RCE_prop_loss,
                'KLD': KLD_loss,
                'beta': beta
            })

        print(
            f'train {epoch} | {total_loss} {RCE_mol_loss} {RCE_prop_loss} {KLD_loss}')

        model.eval()
        valid_loss, valid_RCE_mol_loss, valid_RCE_prop_loss, valid_KLD_loss = 0, 0, 0, 0
        valid_pbar = tqdm(valid_loader, total=len(valid_loader), ncols=100)
        with torch.no_grad():
            for i, (src, trg, cond) in enumerate(valid_pbar):
                src = src.transpose(0, 1).to(opt.device)
                trg = trg.transpose(0, 1).to(opt.device)
                cond = cond.float().to(opt.device)

                trg_input = trg[:, :-1]

                src_mask, trg_mask = create_masks(
                    src, trg_input, cond, opt
                )
                preds_prop, preds_mol, mu, log_var, z = model(
                    src, trg_input, cond, src_mask, trg_mask
                )
                ys_mol = trg[:, 1:].contiguous().view(-1)

                preds_mol = preds_mol.contiguous().view(-1, preds_mol.size(-1))
                ys_mol = trg[:, 1:].contiguous().view(-1)
                preds_prop = preds_prop.view(-1, 3)
                cond = cond.view(-1, 3)

                loss_te, RCE_mol_te, RCE_prop_te, KLD_te = loss_function(
                    opt, beta, preds_prop, preds_mol, cond, ys_mol, mu, log_var
                )
                valid_loss += (loss_te.item() / opt.batch_size)
                valid_RCE_mol_loss += (RCE_mol_te.item() / opt.batch_size)
                valid_RCE_prop_loss += (RCE_prop_te.item() / opt.batch_size)
                valid_KLD_loss += (KLD_te.item() / opt.batch_size)
                valid_pbar.set_postfix({
                    'total': valid_loss,
                    'RCE_mol': valid_RCE_mol_loss,
                    'RCE_prop': valid_RCE_prop_loss,
                    'KLD': valid_KLD_loss
                })

        scheduler.step()
        print(
            f'valid {epoch} | {valid_loss} {valid_RCE_mol_loss} {valid_RCE_prop_loss} {valid_KLD_loss}')

        if best_loss > valid_loss:
            best_loss = valid_loss
            save_checkpoint(epoch, model, optimizer)
            print('best update !!! best loss: ', best_loss)


def get_optimizer(model, config):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.lr,
                                 betas=(.9, .98))

    return optimizer


def get_scheduler(optimizer, config):

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(range(0, config.epochs, 10)),
        gamma=0.95,  # config.lr_gamma,
        last_epoch=-1,
    )

    return lr_scheduler


if __name__ == "__main__":
    set_randomness()

    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    opt = get_args()
    MAX_LEN = opt.max_len
    df = pd.read_csv('smiles.csv')
    train_data, valid_data = train_test_split(
        df, test_size=0.2, random_state=opt.random_seed
    )
    train_data = train_data[train_data['smiles'].str.len() <= MAX_LEN]
    valid_data = valid_data[valid_data['smiles'].str.len() <= MAX_LEN]

    train_data, valid_data = property_normalize(train_data, valid_data)
    train_data.to_csv('train_prop.csv', index=False)
    valid_data.to_csv('valid_prop.csv', index=False)
    print('create train data and valid data...')

    train = pd.read_csv('train_prop.csv')
    train_dataset = MoleculeDataset(train, 'smiles', 'smiles')
    print(train.loc[1], train_dataset[1])
    print(train_dataset.source_vocab.itos,
          train_dataset.source_vocab.stoi)

    valid = pd.read_csv('valid_prop.csv')
    valid_dataset = MoleculeDataset(valid, 'smiles', 'smiles')

    train_loader = get_train_loader(train_dataset,
                                    batch_size=opt.batch_size,
                                    num_workers=2)
    valid_loader = get_valid_loader(valid_dataset,
                                    train_dataset,
                                    batch_size=opt.batch_size,
                                    num_workers=2)
    model = Transformer(
        opt,
        len(train_dataset.source_vocab.itos),
        len(train_dataset.source_vocab.itos)
    )
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(opt.device)

    optimizer = get_optimizer(model, opt)
    scheduler = get_scheduler(optimizer, opt)

    opt.src_pad = train_dataset.source_vocab.stoi['<PAD>']
    opt.trg_pad = train_dataset.source_vocab.stoi['<PAD>']
    print('model training start...')
    train(opt, model, optimizer, scheduler, train_loader, valid_loader)