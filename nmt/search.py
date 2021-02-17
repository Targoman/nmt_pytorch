# coding: utf-8
"""
Implementation of beam search
"""

import math
import sys
from typing import Callable

# pylint: disable=no-member
import torch

from torch import Tensor

from nmt.common import Ignore, configured
from nmt.encoderdecoder import EncoderDecoder


def short_sent_penalty(length, src_length, log_probs):
    # Divide by ((5 + |Y|)/(5 + 1)) ** alpha
    scale = (1 + length / 6)**-1
    return log_probs * scale

@configured('search')
def beam_search(
    x_e: Tensor,
    x_mask: Tensor,
    model: EncoderDecoder,
    get_scores: Callable[[int, Tensor, Tensor], Tensor] = None,
    beam_size: int = 3,
    length_pentalty_alpha: float = 1.0,
    max_target_length: int = 80,
):

    b, _, _ = x_e.size()
    v = len(model.tgt_vocab.itos)
    tv = model.tgt_vocab

    pad_index = model.tgt_vocab.pad_index
    sos_index = model.tgt_vocab.sos_index
    eos_index = model.tgt_vocab.eos_index

    x_l = x_mask.sum(dim=-1).squeeze(1)
    y = torch.LongTensor([[sos_index] for _ in range(b)]).to(x_e.device)
    freezed = torch.BoolTensor([False for _ in range(b)]).to(x_e.device)
    log_probs = torch.FloatTensor([[0] for _ in range(b)]).to(x_e.device)

    if get_scores is None:
        get_scores = lambda length, src_length, log_probs: log_probs

    state = None

    for length in range(max_target_length):

        y_mask = y != pad_index
        y_mask = y_mask.unsqueeze(1)

        log_p_n, state = model.decode_one_step(y, x_e, y_mask, x_mask, state)
        log_p_n[freezed, :] = -math.inf
        log_p_n[freezed, pad_index] = 0
        log_p_n[~freezed, pad_index] = -math.inf
        log_probs = log_probs + log_p_n
        log_probs = log_probs.view(b, -1)

        scores = get_scores(length, x_l, log_probs)

        _, indexes = scores.topk(beam_size)

        log_probs = log_probs.gather(-1, indexes).view(-1).unsqueeze(1)

        row_indexes = indexes.div(v) + (y.size(0) // b) * torch.arange(
            indexes.size(0), dtype=torch.long, device=indexes.device
        ).unsqueeze(1)
        row_indexes = row_indexes.view(-1)
        indexes = indexes.fmod(v).view(-1)

        x_e = x_e.index_select(dim=0, index=row_indexes)
        x_l = x_l.index_select(dim=0, index=row_indexes)
        y = y.index_select(dim=0, index=row_indexes)
        x_mask = x_mask.index_select(dim=0, index=row_indexes)
        freezed = freezed.index_select(dim=0, index=row_indexes)

        y = torch.cat([y, indexes.unsqueeze(1)], dim=-1)
        freezed = freezed | (indexes == eos_index)

        if freezed.all():
            break

    assert y.size(
        0
    ) == b * beam_size, 'Beam search results has invalid number of candidates (Batch count={}, Beam size={}, Candidates={})'.format(
        b, beam_size, y.size(0)
    )
    return y[::beam_size, :], log_probs[::beam_size, 0]
