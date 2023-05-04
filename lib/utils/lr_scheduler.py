import numpy as np
from typing import List
from bisect import bisect_right

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler


def get_lr_scheduler(name, **kwargs):
    assert name in _lr_scheduler, f'{name} LR Schuduler is Not implemented!'
    if kwargs['last_epoch'] == 0:
        kwargs['last_epoch'] = -1
    else:
        kwargs['last_epoch'] = kwargs['last_epoch'] * kwargs['epoch_iters']

    return _lr_scheduler[name](**kwargs)


class WarmupMultiStepLR(_LRScheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            milestones: List[int],
            gamma: float = 0.1,
            warmup_factor: float = 0.001,
            warmup_iters: int = 1000,
            warmup_method: str = 'linear',
            last_epoch: int = -1,
            **kwargs):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of"
                " increasing integers. Got {}".format(milestones)
            )

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        return [
                base_lr
                * warmup_factor
                * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


class WarmupPolyLR(_LRScheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            max_iters: int,
            eta_min: float = 1e-6,
            power: float = 0.9,
            warmup_factor: float = 0.001,
            warmup_iters: int = 1000,
            warmup_method: str = 'linear',
            last_epoch: int = -1,
            **kwargs):
        self.max_iters = max_iters
        self.eta_min = eta_min
        self.power = power
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        return [
                self.eta_min
                + (base_lr - self.eta_min)
                * warmup_factor
                * pow((1 - 1.0 * self.last_epoch / self.max_iters), self.power)
                for base_lr in self.base_lrs]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


class WarmupCosineLR(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        eta_min: float = 1e-06,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
        **kwargs
    ):
        self.max_iters = max_iters
        self.eta_min = eta_min
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        # Different definitions of half-cosine with warmup are possible. For
        # simplicity we multiply the standard half-cosine schedule by the warmup
        # factor. An alternative is to start the period of the cosine at warmup_iters
        # instead of at 0. In the case that warmup_iters << max_iters the two are
        # very close to each other.
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * warmup_factor
            * 0.5
            * (1.0 + np.cos(np.pi * self.last_epoch / self.max_iters))
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


def _get_warmup_factor_at_iter(
    method: str, iter: int, warmup_iters: int, warmup_factor: float
) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See https://arxiv.org/abs/1706.02677 for more details.
    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).
    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))


_lr_scheduler = {
    'poly': WarmupPolyLR,
    'cos': WarmupCosineLR,
    'step': WarmupMultiStepLR,
}
