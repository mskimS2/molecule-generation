import math
import torch
import torch.nn.functional as F
from utils import nopeak_mask


def init_vars(cond, model, SRC, TRG, toklen, opt, z):
    init_tok = SRC['<SOS>']

    src_mask = (torch.ones(1, 1, toklen) != 0)
    trg_mask = nopeak_mask(1, opt)

    trg_in = torch.LongTensor([[init_tok]])

    trg_in = trg_in.to(opt.device)
    z = z.to(opt.device)
    src_mask = src_mask.to(opt.device)
    trg_mask = trg_mask.to(opt.device)

    if opt.use_cond2dec == True:
        output_mol = model.out(model.decoder(
            trg_in, z, cond, src_mask, trg_mask))[:, 3:, :]
    else:
        output_mol = model.out(model.decoder(
            trg_in, z, cond, src_mask, trg_mask))
    out_mol = F.softmax(output_mol, dim=-1)

    probs, ix = out_mol[:, -1].data.topk(opt.k)
    log_scores = torch.Tensor([math.log(prob)
                              for prob in probs.data[0]]).unsqueeze(0)

    outputs = torch.zeros(opt.k, opt.max_len).long()

    outputs = outputs.to(opt.device)
    outputs[:, 0] = init_tok
    outputs[:, 1] = ix[0]

    e_outputs = torch.zeros(opt.k, z.size(-2), z.size(-1))

    e_outputs = e_outputs.to(opt.device)
    e_outputs[:, :] = z[0]

    return outputs, e_outputs, log_scores


def k_best_outputs(outputs, out, log_scores, i, k):
    probs, ix = out[:, -1].data.topk(k)
    log_probs = torch.Tensor(
        [math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0, 1)
    k_probs, k_ix = log_probs.view(-1).topk(k)

    row = k_ix // k
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]

    log_scores = k_probs.unsqueeze(0)

    return outputs, log_scores


def beam_search(cond, model, SRC, TRG, toklen, opt, z):
    cond = cond.to(opt.device).view(1, -1)

    outputs, e_outputs, log_scores = init_vars(
        cond, model, SRC, TRG, toklen, opt, z)

    cond = cond.repeat(opt.k, 1)
    src_mask = (torch.ones(1, 1, toklen) != 0)
    src_mask = src_mask.repeat(opt.k, 1, 1)

    src_mask = src_mask.to(opt.device)
    eos_tok = SRC['<EOS>']

    ind = None
    for i in range(2, opt.max_len):
        trg_mask = nopeak_mask(i, opt)
        trg_mask = trg_mask.repeat(opt.k, 1, 1)

        if opt.use_cond2dec == True:
            output_mol = model.out(model.decoder(
                outputs[:, :i], e_outputs, cond, src_mask, trg_mask))[:, 3:, :]
        else:
            output_mol = model.out(model.decoder(
                outputs[:, :i], e_outputs, cond, src_mask, trg_mask))
        out_mol = F.softmax(output_mol, dim=-1)

        outputs, log_scores = k_best_outputs(
            outputs, out_mol, log_scores, i, opt.k)
        # Occurrences of end symbols for all input sentences.
        ones = (outputs == eos_tok).nonzero()
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long).cuda()
        for vec in ones:
            i = vec[0]
            if sentence_lengths[i] == 0:  # First end symbol has not been found yet
                sentence_lengths[i] = vec[1]  # Position of first end symbol

        num_finished_sentences = len([s for s in sentence_lengths if s > 0])

        if num_finished_sentences == opt.k:
            alpha = 0.7
            div = 1/(sentence_lengths.type_as(log_scores)**alpha)
            _, ind = torch.max(log_scores * div, 1)
            ind = ind.data[0]
            break

    if ind is None:
        length = (outputs[0] == eos_tok).nonzero()
        if length.nelement() == 0:
            length = opt.max_len
        else:
            length = length[0]

        return ' '.join([TRG[tok.item()] for tok in outputs[0][1:length]])

    else:
        length = (outputs[0] == eos_tok).nonzero()

        if length.nelement() == 0:
            length = opt.max_len
        else:
            length = length[0]

        return ' '.join([TRG[tok.item()] for tok in outputs[ind][1:length]])
