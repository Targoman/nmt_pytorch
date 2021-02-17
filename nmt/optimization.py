# coding: utf-8
"""
Optimizers
"""

from typing import Any, Tuple

from torch.nn import Parameter
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau, StepLR, ExponentialLR
from nmt.common import configuration, configured

configuration.ensure_submodule('train').ensure_submodule('optimizer'
                                                        ).ensure_param(
                                                            'type', 'adam'
                                                        )


@configured('train.optimizer')
def get_adam_optimizer_dummy(
    lr: float = 1e-4,
    betas: Tuple[float] = (0.9, 0.999),
    eps: float = 1e-7,
    weight_decay: float = 0.8
):
    pass


def build_optimizer(params: Any):
    @configured('train.optimizer')
    def get_adam_optimizer(
        lr: float = 1e-4,
        betas: Tuple[float] = (0.9, 0.999),
        eps: float = 1e-7,
        weight_decay: float = 0.8
    ):
        return Adam(params, lr, betas, eps, weight_decay)

    OPTIMIZERS = {'adam': get_adam_optimizer}

    type_ = configuration.ensure_submodule('train'
                                          ).ensure_submodule('optimizer').type
    if type_ not in OPTIMIZERS:
        raise Exception('Optimizer `{}` is not registered.'.format(type_))

    return OPTIMIZERS[type_]()


class NoamScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        embedding_size: int,
        factor: float,
        warmup_steps: int,
        last_epoch: int = -1
    ):
        self.factor = factor * embedding_size**(-0.5)
        self.alpha = warmup_steps**(-1.5)
        _LRScheduler.__init__(self, optimizer, last_epoch=last_epoch)

    def get_lr(self):
        lr = self.factor * min(
            self._step_count**(-0.5), self.alpha * self._step_count
        )
        return (lr for _ in self.optimizer.param_groups)


@configured('train.noam_scheduler')
def get_noam_scheduler_dummy(
    embedding_size: int = 512, factor: float = 1.0, warmup_steps: int = 4000
):
    pass

@configured('train')
def build_scheduler(optimizer: Optimizer, schedule_mode: str = 'noam'):

    if schedule_mode is None:
        return None

    @configured('train.noam_scheduler')
    def get_noam_scheduler(
        embedding_size: int = 512,
        factor: float = 1.0,
        warmup_steps: int = 4000
    ):
        return NoamScheduler(optimizer, embedding_size, factor, warmup_steps)

    SCHEDULER_CREATORS = {'noam': get_noam_scheduler, 'none': lambda: None}

    return SCHEDULER_CREATORS[schedule_mode]()
