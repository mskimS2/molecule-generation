import copy
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


def init_weights(module):
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=1.0)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
            / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        output = self.out(concat)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.gelu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

# build a decoder layer with two multi-head attention layers and
# one feed-forward layer


class DecoderLayer(nn.Module):
    def __init__(self, opt, d_model, heads, dropout=0.1):
        super().__init__()
        self.use_cond2dec = opt.use_cond2dec
        self.use_cond2lat = opt.use_cond2lat
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, cond_input, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        if self.use_cond2lat == True:
            cond_mask = torch.unsqueeze(cond_input, -2)
            cond_mask = torch.ones_like(cond_mask, dtype=bool)
            src_mask = torch.cat([cond_mask, src_mask], dim=2)

        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=200, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, opt, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.cond_dim = opt.cond_dim
        self.d_model = d_model
        self.embed_sentence = Embedder(vocab_size, d_model)
        self.embed_cond2enc = nn.Linear(opt.cond_dim, d_model*opt.cond_dim)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

        self.fc_mu = nn.Linear(d_model, opt.latent_dim)
        self.fc_log_var = nn.Linear(d_model, opt.latent_dim)

    def forward(self, src, cond_input, mask):
        cond2enc = self.embed_cond2enc(cond_input).view(
            cond_input.size(0), cond_input.size(1), -1)
        x = self.embed_sentence(src)
        x = torch.cat([cond2enc, x], dim=1)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        x = self.norm(x)

        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return self.sampling(mu, log_var), mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu)

    def get_attn(self, src, cond_input, mask):
        cond2enc = self.embed_cond2enc(cond_input).view(
            cond_input.size(0), cond_input.size(1), -1)
        x = self.embed_sentence(src)
        x = torch.cat([cond2enc, x], dim=1)
        x = self.pe(x)

        for i in range(self.N):
            x = self.layers[i](x, mask)
        x = self.norm(x)

        return x


class Decoder(nn.Module):
    def __init__(self, opt, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.cond_dim = opt.cond_dim
        self.d_model = d_model
        self.use_cond2dec = opt.use_cond2dec
        self.use_cond2lat = opt.use_cond2lat
        self.embed = Embedder(vocab_size, d_model)
        if self.use_cond2dec == True:
            self.embed_cond2dec = nn.Linear(
                opt.cond_dim, d_model * opt.cond_dim)  # concat to trg_input
        if self.use_cond2lat == True:
            self.embed_cond2lat = nn.Linear(
                opt.cond_dim, d_model * opt.cond_dim)  # concat to trg_input
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.fc_z = nn.Linear(opt.latent_dim, d_model)
        self.layers = get_clones(DecoderLayer(opt, d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, cond_input, src_mask, trg_mask):
        x = self.embed(trg)
        e_outputs = self.fc_z(e_outputs)
        if self.use_cond2dec == True:
            cond2dec = self.embed_cond2dec(cond_input).view(
                cond_input.size(0), cond_input.size(1), -1)
            x = torch.cat([cond2dec, x], dim=1)
        if self.use_cond2lat == True:
            cond2lat = self.embed_cond2lat(cond_input).view(
                cond_input.size(0), cond_input.size(1), -1)
            e_outputs = torch.cat([cond2lat, e_outputs], dim=1)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, cond_input, src_mask, trg_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, opt, src_vocab, trg_vocab):
        super(Transformer, self).__init__()

        assert opt.d_model % opt.heads == 0
        assert opt.dropout < 1

        self.use_cond2dec = opt.use_cond2dec
        self.use_cond2lat = opt.use_cond2lat
        self.encoder = Encoder(
            opt,
            src_vocab,
            opt.d_model,
            opt.n_layers,
            opt.heads,
            opt.dropout
        )
        self.decoder = Decoder(
            opt,
            trg_vocab,
            opt.d_model,
            opt.n_layers,
            opt.heads,
            opt.dropout
        )
        self.out = nn.Linear(opt.d_model, trg_vocab)
        if self.use_cond2dec == True:
            self.prop_fc = nn.Linear(trg_vocab, 1)

        self.apply(init_weights)

    def forward(self, src, trg, cond, src_mask, trg_mask):
        z, mu, log_var = self.encoder(src, cond, src_mask)
        d_output = self.decoder(trg, z, cond, src_mask, trg_mask)
        output = self.out(d_output)
        if self.use_cond2dec == True:
            output_prop, output_mol = self.prop_fc(
                output[:, :3, :]), output[:, 3:, :]
        else:
            output_prop, output_mol = torch.zeros(
                output.size(0), 3, 1).to('cuda'), output

        return output_prop, output_mol, mu, log_var, z
