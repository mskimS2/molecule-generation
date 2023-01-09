import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

from tokenizer import Moltokenize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


class Vocabulary:
    """
    __init__ method is called by default as soon as an object of this class is initiated
    we use this method to initiate our vocab dictionaries
    """

    def __init__(self, freq_threshold, max_size=120):
        """
        freq_threshold : the minimum times a word must occur in corpus to be treated in vocab
        max_size : max source vocab size.
        # initiate the index to token dict
        # <PAD> -> padding, used for padding the shorter sentences in a batch to match the length of longest sentence in the batch
        # <SOS> -> start token, added in front of each sentence to signify the start of sentence
        # <EOS> -> End of sentence token, added to the end of each sentence to signify the end of sentence
        # <UNK> -> words which are not found in the vocab are replace by this token
        """

        # initiate the token to index dict
        self.itos = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.stoi = {k: j for j, k in self.itos.items()}
        self.pad = self.stoi['<PAD>']
        self.bos = self.stoi['<SOS>']
        self.eos = self.stoi['<EOS>']
        self.freq_threshold = freq_threshold
        self.max_size = max_size

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        tokenizer = Moltokenize()
        return tokenizer.tokenizer(text)

    def build_vocabulary(self, sentence_list):
        """
        build the vocab: create a dictionary mapping of index to string (itos) and string to index (stoi)
        output ex. for stoi -> {'the':5, 'a':6, 'an':7}
        """
        # calculate the frequencies of each word first to remove the words with freq < freq_threshold
        frequencies = {}  # init the freq dict
        idx = 4  # index from which we want our dict to start. We already used 4 indexes for pad, start, end, unk

        # calculate freq of words
        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                if word not in frequencies.keys():
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

        # limit vocab by removing low freq words
        frequencies = {k: v for k, v in frequencies.items()
                       if v > self.freq_threshold}

        # limit vocab to the max_size specified
        # idx =4 for pad, start, end , unk
        frequencies = dict(sorted(frequencies.items(),
                                  key=lambda x: -x[1])[:self.max_size-idx])

        # create vocab
        for word in frequencies.keys():
            self.stoi[word] = idx
            self.itos[idx] = word
            idx += 1

    def numericalize(self, text):
        """
        convert the list of words to a list of corresponding indexes
        """
        # tokenize text
        tokenized_text = self.tokenizer(text)
        numericalized_text = []
        for token in tokenized_text:
            if token in self.stoi.keys():
                numericalized_text.append(self.stoi[token])
            else:  # out-of-vocab words are represented by UNK token index
                numericalized_text.append(self.stoi['<UNK>'])

        return numericalized_text


class MoleculeDataset(Dataset):
    """
    Initiating Variables
    - df: the training dataframe
    - source_column : the name of source text column in the dataframe
    - target_columns : the name of target text column in the dataframe
    - transform : If we want to add any augmentation
    - freq_threshold : the minimum times a word must occur in corpus to be treated in vocab
    - source_vocab_max_size : max source vocab size
    - target_vocab_max_size : max target vocab size
    """

    def __init__(
        self,
        df,
        source_column,
        target_column,
        transform=None,
        freq_threshold=0,
        source_vocab_max_size=10000,
        target_vocab_max_size=10000,
    ):

        self.df = df
        self.transform = transform

        # get source and target texts
        self.source_texts = self.df[source_column]
        self.target_texts = self.df[target_column]
        self.weight = self.df['weight']
        self.logp = self.df['logp']
        self.tpsa = self.df['TPSA']
        # print(self.weight.shape, self.logp.shape, self.tpsa.shape )

        self.source_vocab = Vocabulary(freq_threshold, source_vocab_max_size)
        self.source_vocab.build_vocabulary(self.source_texts.tolist())
        self.target_vocab = Vocabulary(freq_threshold, target_vocab_max_size)
        self.target_vocab.build_vocabulary(self.target_texts.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        source_text = self.source_texts[index]
        target_text = self.target_texts[index]

        if self.transform is not None:
            source_text = self.transform(source_text)

        # numericalize texts ['<SOS>','cat', 'in', 'a', 'bag','<EOS>'] -> [1,12,2,9,24,2]
        numerialized_source = [self.source_vocab.stoi["<SOS>"]]
        numerialized_source += self.source_vocab.numericalize(source_text)
        numerialized_source.append(self.source_vocab.stoi["<EOS>"])

        numerialized_target = [self.source_vocab.stoi["<SOS>"]]
        numerialized_target += self.source_vocab.numericalize(target_text)
        numerialized_target.append(self.source_vocab.stoi["<EOS>"])

        weight = torch.tensor(self.weight[index]).reshape(1, 1)
        logp = torch.tensor(self.logp[index]).reshape(1, 1)
        tpsa = torch.tensor(self.tpsa[index]).reshape(1, 1)
        condition = torch.cat([weight, logp, tpsa], dim=1)

        # convert the list to tensor and return
        numerialized_source = torch.tensor(numerialized_source)
        numerialized_target = torch.tensor(numerialized_target)

        return numerialized_source, numerialized_target, condition


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # get all source indexed sentences of the batch
        source = [item[0] for item in batch]
        # pad them using pad_sequence method from pytorch.
        source = pad_sequence(source,
                              batch_first=False,
                              padding_value=self.pad_idx)

        # get all target indexed sentences of the batch
        target = [item[1] for item in batch]
        # pad them using pad_sequence method from pytorch.
        target = pad_sequence(target,
                              batch_first=False,
                              padding_value=self.pad_idx)

        condition = [item[2] for item in batch]
        condition = pad_sequence(condition,
                                 batch_first=False,
                                 padding_value=self.pad_idx)

        return source, target, condition[0]


def get_train_loader(
    dataset,
    batch_size,
    num_workers=0,
    shuffle=True,
    pin_memory=True
):
    pad_idx = dataset.source_vocab.stoi['<PAD>']
    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=shuffle,
                              pin_memory=pin_memory,
                              collate_fn=MyCollate(pad_idx=pad_idx))
    
    return train_loader


def get_valid_loader(
    dataset,
    train_dataset,
    batch_size,
    num_workers=0,
    shuffle=False,
    pin_memory=True
):
    pad_idx = train_dataset.source_vocab.stoi['<PAD>']
    valid_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=shuffle,
                              pin_memory=pin_memory,
                              collate_fn=MyCollate(pad_idx=pad_idx))
    
    return valid_loader


def property_normalize(train_data, valid_data, scale='robust'):

    scaler = RobustScaler() if scale == 'robust' else StandardScaler()
    train_prop = train_data[['weight', 'logp', 'TPSA']]
    valid_prop = valid_data[['weight', 'logp', 'TPSA']]

    scaler.fit(train_prop)
    norm_train_prop = scaler.transform(train_prop)
    norm_valid_prop = scaler.transform(valid_prop)

    train_data[['weight', 'logp', 'TPSA']] = norm_train_prop
    valid_data[['weight', 'logp', 'TPSA']] = norm_valid_prop

    print('smiles property normalization...')
    print(f'train property shape: {norm_train_prop.shape}')
    print(f'valid property shape: {norm_valid_prop.shape}')

    return train_data, valid_data
