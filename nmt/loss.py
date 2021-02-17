# coding: utf-8
"""
Loss functions for neural machine translation
"""

import math

# pylint: disable=no-member
import torch
from torch import nn
from torch.nn import functional as F

from nmt.common import Ignore, configured

@configured('train')
class EnhancedCrossEntropyLoss(nn.Module):
    def __init__(self, pad_index: Ignore[int], smoothing_amount: float = 0.2):
        nn.Module.__init__(self)
        self.smoothing_amount = smoothing_amount
        self.pad_index = pad_index

    @staticmethod
    def __smoothed_dist_entropy(m: int, n: int, alpha: float):
        beta = 1 - alpha
        return m * (beta * math.log(beta) + alpha * math.log(alpha / (n - 1)))

    def uniform_baseline_loss(self, log_probs, targets):
        n = log_probs.size(-1)
        if self.smoothing_amount <= 0:
            return math.log(n)
        if self.pad_index is None:
            m = targets.numel()
        else:
            m = (targets != self.pad_index).sum()
        return math.log(n -
                        1) + EnhancedCrossEntropyLoss.__smoothed_dist_entropy(
                            m, n, self.smoothing_amount
                        ) / m

    # pylint: disable=arguments-differ
    def forward(self, log_probs, targets, embeddings):
        n = log_probs.size(-1)
        log_probs = log_probs.view(-1, n)
        targets = targets.contiguous().view(-1)
        if self.pad_index is not None:
            log_probs = log_probs[targets != self.pad_index, :]
            targets = targets[targets != self.pad_index]

        loss = F.nll_loss(
            log_probs,
            targets,
            reduction='sum',
            ignore_index=-100 if self.pad_index is None else self.pad_index
        )

        target_embeddings = embeddings.index_select(dim=0, index=targets)
        target_prob_estimate = torch.matmul(target_embeddings, embeddings.t()).view(-1, n)
        target_prob_estimate = F.softmax(target_prob_estimate, dim=-1)

        loss1 = -(target_prob_estimate * log_probs).sum()

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
            ) + beta * loss + EnhancedCrossEntropyLoss.__smoothed_dist_entropy(
                targets.numel(), n, self.smoothing_amount
            )

        return 0.5 * (loss + loss1)


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

    def uniform_baseline_loss(self, log_probs, targets):
        n = log_probs.size(-1)
        if self.smoothing_amount <= 0:
            return math.log(n)
        if self.pad_index is None:
            m = targets.numel()
        else:
            m = (targets != self.pad_index).sum()
        return math.log(n -
                        1) + SmoothedCrossEntropyLoss.__smoothed_dist_entropy(
                            m, n, self.smoothing_amount
                        ) / m

    # pylint: disable=arguments-differ
    def forward(self, log_probs, targets, embeddings):
        n = log_probs.size(-1)
        log_probs = log_probs.view(-1, n)
        targets = targets.contiguous().view(-1)
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

class SimpleLoss(nn.Module):
    def __init__(self, pad_index: Ignore[int]):
        nn.Module.__init__(self)
        self.pad_index = pad_index

    def uniform_baseline_loss(self, log_probs, targets):
        return log_probs.max()

    # pylint: disable=arguments-differ
    def forward(self, products, targets, embeddings):
        n = products.size(-1)
        sum_products = products.exp().sum(-1).log()
        sum_products = sum_products.view(-1)
        products = products.view(-1, n)
        targets = targets.contiguous().view(-1)
        products = products[targets != self.pad_index, :]
        sum_products = sum_products[targets != self.pad_index]
        targets = targets[targets != self.pad_index]
        loss = -torch.gather(products, -1, targets.unsqueeze(-1)).sum() + sum_products.sum()
        return loss


@configured('train')
def get_loss_function(
    pad_index: Ignore[int], type: str = 'smoothed_cross_entropy'
):

    LOSS_EVALUATORS = {
        'simple_loss':
            lambda: SimpleLoss(pad_index=pad_index),
        'smoothed_cross_entropy':
            lambda: SmoothedCrossEntropyLoss(pad_index=pad_index),
        'enhanced_cross_entropy':
            lambda: EnhancedCrossEntropyLoss(pad_index=pad_index)
    }

    if type not in LOSS_EVALUATORS:
        raise Exception('`{}` loss function is not registered.'.format(type))

    return LOSS_EVALUATORS[type]()
