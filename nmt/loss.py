# coding: utf-8
"""
Loss functions for neural machine translation
"""

import os
import json
import math

# pylint: disable=no-member
import torch
from torch import nn
from torch.nn import functional as F

from nmt.common import Ignore, get_configuration, push_configuration,\
                       pop_configuration, relative_to_config_path,\
                       configured, get_device
from nmt.predict import get_vocabularies, find_best_model
from nmt.model import build_model, update_and_ensure_model_output_path

@configured('train')
class SmoothedCrossEntropyLoss(nn.Module):
    def __init__(self, pad_index: Ignore[int], smoothing_amount: float = 0.2):
        nn.Module.__init__(self)
        self.smoothing_amount = smoothing_amount
        self.pad_index = pad_index

    @staticmethod
    def __smoothed_dist_entropy(m: int, n: int, alpha: float):
        beta = 1 - alpha
        return m * (beta * math.log(beta) + alpha * math.log(alpha / (n - 1)))

    def uniform_baseline_loss(self, log_probs, batch, model):
        n = log_probs.size(-1)
        if self.smoothing_amount <= 0:
            return math.log(n)
        if self.pad_index is None:
            m = batch[1][:, 1:].numel()
        else:
            m = (batch[1][:, 1:] != self.pad_index).sum()
        return math.log(n -
                        1) + SmoothedCrossEntropyLoss.__smoothed_dist_entropy(
                            m, n, self.smoothing_amount
                        ) / m

    # pylint: disable=arguments-differ
    def forward(self, log_probs, batch, model):
        n = log_probs.size(-1)
        log_probs = log_probs.view(-1, n)
        targets = batch[1][:, 1:].contiguous().view(-1)
        if self.pad_index is not None:
            log_probs = log_probs[targets != self.pad_index, :]
            targets = targets[targets != self.pad_index]

        loss = F.nll_loss(
            log_probs,
            targets,
            reduction='sum',
        )

        if self.smoothing_amount > 0:
            if self.pad_index is not None:
                n -= 1
                log_probs = torch.cat(
                    [
                        log_probs[:, :self.pad_index],
                        log_probs[:, self.pad_index + 1:]
                    ],
                    dim=-1
                )

            alpha = self.smoothing_amount / (n - 1)
            beta = 1 - self.smoothing_amount * n / (n - 1)

            loss = -alpha * log_probs.sum(
            ) + beta * loss + SmoothedCrossEntropyLoss.__smoothed_dist_entropy(
                targets.numel(), n, self.smoothing_amount
            )

        return loss

@configured('train')
class TeacherStudentLoss(nn.Module):
    def __init__(self, pad_index: Ignore[int], teacher_config_path: str = '/DOES_NOT_EXIST'):
        teacher_config_path = relative_to_config_path(teacher_config_path)
        assert os.path.exists(teacher_config_path), "Teacher model config does not exist."
        nn.Module.__init__(self)
        teacher_model_config = get_configuration().clone()
        with open(teacher_config_path) as f:
            teacher_model_config.load(json.load(f))

        push_configuration(teacher_model_config)
        update_and_ensure_model_output_path('test', None)

        best_model_path = find_best_model()
        if best_model_path is None:
            raise ValueError('Could not find the teacher model.')
        (src_vocab, tgt_vocab), _ = get_vocabularies()
        self.teacher_model = build_model(src_vocab, tgt_vocab)
        state_dict = torch.load(best_model_path)
        self.teacher_model.load_state_dict(state_dict['model_state'])
        self.teacher_model.to(get_device())
        self.teacher_model.eval()
        self.src_pad_index = src_vocab.pad_index
        self.tgt_pad_index = tgt_vocab.pad_index

        pop_configuration()

        self.pad_index = pad_index

    def uniform_baseline_loss(self, log_probs, batch, model):
        return math.nan

    def get_teacher_probs(self, batch):
        with torch.no_grad():
            x_mask = batch[0] != self.src_pad_index
            x_mask = x_mask.unsqueeze(1)
            y_mask = batch[1][:, :-1] != self.tgt_pad_index
            y_mask = y_mask.unsqueeze(1)
            x_e = self.teacher_model.encode(batch[0], x_mask)
            teacher_probs = self.teacher_model.decode(
                batch[1][:, :-1],
                x_e,
                y_mask,
                x_mask,
                teacher_forcing=False
            ).exp()
        return teacher_probs, y_mask

    # pylint: disable=arguments-differ
    def forward(self, log_probs, batch, model):
        teacher_probs, y_mask = self.get_teacher_probs(batch)
        loss = teacher_probs * log_probs
        loss = -loss.view(-1, loss.size(-1))
        loss = loss[y_mask.view(-1), :].sum()
        return loss

@configured('train')
def get_loss_function(
    pad_index: Ignore[int], type: str = 'smoothed_cross_entropy'
):

    LOSS_EVALUATORS = {
        'smoothed_cross_entropy':
            lambda: SmoothedCrossEntropyLoss(pad_index=pad_index),
        'teacher_student':
            lambda: TeacherStudentLoss(pad_index=pad_index)
    }

    if type not in LOSS_EVALUATORS:
        raise Exception('`{}` loss function is not registered.'.format(type))

    return LOSS_EVALUATORS[type]()
