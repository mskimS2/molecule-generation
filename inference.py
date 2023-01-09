import torch
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from beam_search import beam_search
from pickle import load
from sklearn.model_selection import train_test_split

from args import get_args
from data_loader import *
from utils import set_randomness
from models.transformer import Transformer


def get_sampled_element(myCDF):
    a = np.random.uniform(0, 1)
    return np.argmax(myCDF >= a)-1


def run_sampling(xc, dxc, myPDF, myCDF, nRuns):
    sample_list = []
    X = np.zeros_like(myPDF, dtype=int)
    for k in np.arange(nRuns):
        idx = get_sampled_element(myCDF)
        sample_list.append(xc[idx] + dxc * np.random.normal() / 2)
        X[idx] += 1
    return np.array(sample_list).reshape(nRuns, 1), X/np.sum(X)


def tokenlen_gen_from_data_distribution(data, nBins, size):
    count_c, bins_c, = np.histogram(data, bins=nBins)
    myPDF = count_c / np.sum(count_c)
    dxc = np.diff(bins_c)[0]
    xc = bins_c[0:-1] + 0.5 * dxc

    myCDF = np.zeros_like(bins_c)
    myCDF[1:] = np.cumsum(myPDF)

    tokenlen_list, X = run_sampling(xc, dxc, myPDF, myCDF, size)

    return tokenlen_list


def inference(
    opt,
    scaler,
    model,
    SRC,
    TRG,
    n_samples: int = 1,
    n_per_samples: int = 100
):
    molecules = []
    valid_check = []
    conds_rdkit = []
    conds_trg = []

    toklen_data = pd.read_csv('valid_prop_toklen_list.csv').values

    data = pd.read_csv('valid_prop.csv')
    data['length'] = data['smiles'].apply(lambda x: len(str(x)))
    toklen_data = data['length'].values
    print('toklen_data.max(), toklen_data.min(): ',
          toklen_data.max(), toklen_data.min())
    conds = data[['weight', 'logp', 'TPSA']][:n_per_samples].values
    toklen_data = tokenlen_gen_from_data_distribution(
        data=toklen_data,
        nBins=int(toklen_data.max()-toklen_data.min()),
        size=n_samples*n_per_samples
    )

    for idx in tqdm(range(n_per_samples), ncols=100):
        for i in range(n_samples):
            # +3 due to cond2enc
            toklen = int(toklen_data[idx]) + 3
            z = torch.Tensor(np.random.normal(
                size=(1, toklen, opt.latent_dim)))

            cond = torch.autograd.Variable(torch.Tensor(conds[idx]))
            gen_mol = beam_search(cond, model, SRC, TRG, toklen, opt, z)
            gen_mol = ''.join(gen_mol).replace(' ', '')
            molecules.append(gen_mol)
            conds_trg.append(
                scaler.inverse_transform(conds[[idx]]).reshape(1, 3)[0]
            )

            m = Chem.MolFromSmiles(gen_mol)
            if m is None:
                valid_check.append(0)
                conds_rdkit.append([None, None, None, None])
            else:
                valid_check.append(1)
                conds_rdkit.append(np.array([Descriptors.MolWt(m),
                                            Descriptors.MolLogP(m),
                                            Descriptors.TPSA(m),
                                            QED.qed(m)]))

    np_conds_trg = np.array(conds_trg)
    np_conds_rdkit = np.array(conds_rdkit)
    print(np_conds_trg.shape)
    gen_list = pd.DataFrame({
        "mol": molecules,
        "validity": valid_check,
        "condition(weight)": np_conds_trg[:, 0],
        "condition(logP)": np_conds_trg[:, 1],
        "condition(TPSA)": np_conds_trg[:, 2],
        "rdkit(weight)": np_conds_rdkit[:, 0],
        "rdkit(logP)": np_conds_rdkit[:, 1],
        "rdkit(TPSA)": np_conds_rdkit[:, 2],
        "rdkit(QED)": np_conds_rdkit[:, 3],
    })

    gen_list.to_csv(f'results/generation results_best_{n_samples}.csv'.format(
        opt.latent_dim, opt.epochs, opt.k
    ), index=True)

    print(f'generation success ratio: {sum(valid_check)/len(valid_check)}')


if __name__ == "__main__":
    set_randomness()

    save_path = f'weights/best.pt'

    opt = get_args()

    df = pd.read_csv('data/smiles.csv')
    train_data, valid_data = train_test_split(
        df, test_size=0.2, random_state=opt.random_seed)
    train_data = train_data[train_data['smiles'].str.len() <= opt.max_len]
    valid_data = valid_data[valid_data['smiles'].str.len() <= opt.max_len]

    # 2. train data and valid data smile properpy normalization
    train_data, valid_data = property_normalize(train_data, valid_data)
    train_data.to_csv('data/train_prop2.csv', index=False)
    valid_data.to_csv('data/valid_prop2.csv', index=False)

    train_dataset = MoleculeDataset(train_data, 'smiles', 'smiles')

    opt.src_pad = train_dataset.source_vocab.stoi['<PAD>']
    opt.trg_pad = train_dataset.source_vocab.stoi['<PAD>']

    SRC = train_dataset.source_vocab.stoi
    TRG = train_dataset.source_vocab.itos
    model = Transformer(
        opt,
        len(train_dataset.source_vocab.stoi),
        len(train_dataset.source_vocab.itos)
    ).load_state_dict(torch.load(save_path)['model'])
    model.to(opt.device).eval()
    robust_scaler = load(open('weights/robust_scaler.pkl', 'rb'))
    inference(opt, robust_scaler, model, SRC, TRG)
