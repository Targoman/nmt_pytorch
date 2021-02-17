# coding: utf-8
"""
Sanity checks for the implementation
"""

# pylint: disable=no-member,no-value-for-parameter
import torch

from nmt.common import get_logger

def equal(a, b):
    return torch.allclose(a, b, 1e-5, 1e-6, equal_nan=True)

def check_smooth_cross_entropy_loss(logger):
    from nmt.loss import SmoothedCrossEntropyLoss

    n = 15
    pad_index = 1
    valid = True
    for alpha in [0.0, 0.1, 0.2, 0.5, 0.7]:
        loss_function = SmoothedCrossEntropyLoss(
            pad_index=pad_index, smoothing_amount=alpha
        )

        targets = torch.arange(n, dtype=torch.long)
        target_probs = (
            (1 - alpha) - alpha / (n - (2 if pad_index is not None else 1))
        ) * torch.eye(n) + alpha / (n - (2 if pad_index is not None else 1)
                                   ) * torch.ones(n, n)
        target_probs[:, pad_index] = 1e-10
        target_probs[pad_index, :] = 1e-10

        log_probs = torch.nn.functional.log_softmax(torch.rand(n, n), -1)

        loss = loss_function(log_probs, targets)
        sane_loss = -(target_probs * log_probs).sum()
        if alpha > 0.0:
            sane_loss += (target_probs * target_probs.log()).sum()

        if not equal(loss, sane_loss):
            valid = False
            logger.error(
                'SmoothedCrossEntropyLoss failed for n={}, alpha={}. ({} vs {})'
                .format(n, alpha,
                        loss.item(),
                        sane_loss.item())
            )
            break
    if valid:
        logger.info('SmoothedCrossEntropyLoss ... OK!')


def check_bleu(logger):
    from nmt.metric import update_bleu_params, get_bleu

    pad_index = 1
    sos_index = 2
    eos_index = 3

    src = [sos_index, 12, 5, 4, 13, 13, 333, eos_index, pad_index, pad_index]
    ref = [sos_index, 133, 12, 5, 4, 15, 13, 333, eos_index, pad_index]

    x0 = torch.LongTensor([src])
    x1 = torch.LongTensor([ref])

    params = update_bleu_params(x0, x1, pad_index)

    expected_params = {
        'correct': [5, 3, 1, 0],
        'total': [6, 5, 4, 3],
        'sys_len': 6,
        'ref_len': 7
    }

    valid = True
    for p in params:
        if params[p] != expected_params[p]:
            logger.error(
                'update_bleu_params failed, `{}` is different than expected. ({} vs {})'
                .format(p, params[p], expected_params[p])
            )
            valid = False

    if valid:
        logger.info('update_bleu_params ... OK!')


def check_multihead_attention(logger):

    from nmt.transformer import MultiheadAttention

    mha = MultiheadAttention(
        num_heads=1, embedding_size=4, dropout=0.0, causual=False
    )
    mha.eval()

    mha.k_linear.weight.data = torch.eye(4)
    mha.v_linear.weight.data = torch.eye(4)
    mha.q_linear.weight.data = torch.eye(4)
    mha.o_linear.weight.data = torch.eye(4)

    x = torch.eye(4).unsqueeze(0)

    z = {
        False:
            torch.nn.functional.softmax(torch.eye(4) / 2, dim=-1),
        True:
            torch.FloatTensor(
                [
                    [
                        [1.0000, 0.0000, 0.0000, 0.0000],
                        [0.37754, 0.62246, 0.0000, 0.0000],
                        [0.274068, 0.274068, 0.451862, 0.0000],
                        [0.215112, 0.215112, 0.215112, 0.354663]
                    ]
                ]
            )
    }

    valid = True
    for causual in [False, True]:
        mha.causual = causual
        y = mha(x, x, x)
        if not equal(y, z[causual]):
            logger.error(
                'MultiheadAttention failed for eye matrix (causual={}).'.
                format(causual)
            )
            valid = False
    if valid:
        logger.info('MultiheadAttention ... OK!')


def check_beam_search(logger):

    from nmt.search import beam_search

    class DummyVocab(object):
        def __init__(self):

            self.itos = list(range(16))
            self.unk_index = 0
            self.pad_index = 1
            self.sos_index = 2
            self.eos_index = 3

    class DummyModel(object):
        def __init__(self):

            self.tgt_vocab = DummyVocab()

        def decode_one_step(self, y, xe, y_mask, x_mask, state):

            log_probs = torch.zeros(xe.size(0), len(self.tgt_vocab.itos))
            if y.size(1) < 4:
                log_probs[:, 5] = 10 / y.size(1)
                log_probs[:, 10] = 5 / y.size(1)
                log_probs[:, 12] = 3 / y.size(1)
            else:
                log_probs[:, self.tgt_vocab.eos_index] = 10

            return log_probs, None

    x_e = torch.rand(1, 3, 8)
    model = DummyModel()

    y = beam_search(x_e, torch.ones(1, 1, 3) != 0, model)
    z = torch.LongTensor([[2, 5, 5, 5, 3]])
    w = torch.FloatTensor([28.333333333])
    if (y[0] == z).all() and equal(y[1], w):
        logger.error('beam_search ... OK!')
    else:
        logger.error('beam_search failed.')

def sanity_check():
    logger = get_logger()

    check_multihead_attention(logger)
    check_smooth_cross_entropy_loss(logger)
    check_beam_search(logger)
    check_bleu(logger)
