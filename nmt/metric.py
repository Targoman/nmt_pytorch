# coding: utf-8
"""
Implementation of metrics of evaluating translations
"""

from typing import List, Tuple

import torch
from torch import Tensor

from sacrebleu import BLEU

from nmt.dataset import Vocabulary

class Metric(object):

    def __init__(self):
        self._score = None
    
    def get_score(self):
        if self._score is None:
            self._score = self._compute_score()
        return self._score

    def update_params(self, src_item: List, ref_item: List):
        if self._score is not None:
            raise Exception(f'Can not update params past evaluating the metric ({self.__class__.__name__})')
        return self._do_update_params(src_item, ref_item)

    def _do_update_params(self, src_item: List, ref_item: List):
        raise NotImplementedError('Metric is an abstract class.')

    def _compute_score(self):
        raise NotImplementedError('Metric is an abstract class.')

class BleuMetric(Metric):
    def __init__(self):
        super(BleuMetric, self).__init__()
        self.sys_len = 0
        self.ref_len = 0

        self.correct = [0 for _ in range(BLEU.NGRAM_ORDER)]
        self.total = [0 for _ in range(BLEU.NGRAM_ORDER)]

        self._bleu = None

    @staticmethod
    def _get_n_grams(item):

        result = [{} for _ in range(BLEU.NGRAM_ORDER)]
        for n in range(BLEU.NGRAM_ORDER):
            sink = result[n]
            for i in range(len(item) - n):
                n_gram = tuple(item[i:i + n + 1])
                if n_gram in sink:
                    sink[n_gram] += 1
                else:
                    sink[n_gram] = 1
        return result

    def _do_update_params(self, src_item: List, ref_item: List):

        self.sys_len += len(src_item)
        self.ref_len += len(ref_item)

        src_ngrams = self._get_n_grams(src_item)
        ref_ngrams = self._get_n_grams(ref_item)

        for n in range(BLEU.NGRAM_ORDER):
            for n_gram, count in src_ngrams[n].items():
                self.correct[n] += min(count, ref_ngrams[n].get(n_gram, 0))
                self.total[n] += count

    def _ensure_blue(self):
        if not self._bleu:
            self._bleu = BLEU.compute_bleu(
                correct=self.correct,
                total=self.total,
                sys_len=self.sys_len,
                ref_len=self.ref_len,
                smooth_method='exp',
                smooth_value=None,
                use_effective_order=False
            )
        return self._bleu

    def _compute_score(self):
        bleu = self._ensure_blue()
        return bleu.score

    def __str__(self):
        bleu = self._ensure_blue()
        precisions = '/'.join(f'{p:.1f}' for p in bleu.precisions)
        return f'bleu={bleu.score:.3f}({precisions},g/r:{bleu.sys_len / bleu.ref_len:.3f})'

class WerMetric(Metric):

    def __init__(self):

        self.count = 0
        self.mean_wer = 0

    def _do_update_params(self, src_item: List, ref_item: List):

        d = torch.zeros(len(ref_item) + 1, len(src_item) + 1, dtype=torch.int)
        torch.arange(len(ref_item) + 1, out=d.narrow(dim=1, start=0, length=1))
        torch.arange(len(src_item) + 1, out=d.narrow(dim=0, start=0, length=1))
        
        for i, r in enumerate(ref_item):
            for j, s in enumerate(src_item):
                if r == s:
                    d[i + 1, j + 1] = d[i, j]
                else:
                    d[i + 1, j + 1] = min(
                        d[i][j],
                        d[i + 1][j],
                        d[i][j + 1]
                    ) + 1
        
        self.mean_wer = self.count * self.mean_wer + float(d[-1, -1]) / len(ref_item)
        self.count += 1
        self.mean_wer /= self.count

    def _compute_score(self):
        return self.mean_wer

class MeteorMetric(Metric):

    def __init__(self, vocab: Vocabulary):

        self.reference = []
        self.hypothesis = []

    def _do_update_params(self, src_item: List, ref_item: List):

        self.reference.append(ref_item)
        self.hypothesis.append(src_item)

    def _compute_score(self):
        pass


def update_metric_params(
    src: Tensor, ref: Tensor, pad_index: int, metrics: Tuple
):
    def to_list(x: Tensor):
        def remove_pad_tokens(s):
            n = sum(e == pad_index for e in s)
            return s[1:-(n + 1)]

        return (remove_pad_tokens(s.tolist()) for s in x)

    for src_item, ref_item in zip(to_list(src), to_list(ref)):
        for metric in metrics:
            metric.update_params(src_item, ref_item)

    return metrics
