"""
Debiased Exponential Moving Average (EMA) mostly copy-paste from Sonnet.
"""

from typing import Optional

import torch
import torch.nn as nn

from utils.dist_helper import is_dist_avail_and_initialized, reduce_across_processes


class ExponentialMovingAverage(nn.Module):
    """Maintains an exponential moving average for a value.
    Note this module uses debiasing by default. If you don't want this please use
    an alternative implementation.
    This module keeps track of a hidden exponential moving average that is
    initialized as a vector of zeros which is then normalized to give the average.
    This gives us a moving average which isn't biased towards either zero or the
    initial value. Reference (https://arxiv.org/pdf/1412.6980.pdf)
    Initially:
      hidden_0 = 0
    Then iteratively:
      hidden_i = (hidden_{i-1} - value) * (1 - decay)
      average_i = hidden_i / (1 - decay^i)
    Attributes:
        average: Variable holding average. Note that this is None until the first
        value is passed.
    """

    def __init__(self, decay: float = 0.999):
        """Creates a debiased moving average module.
        Args:
            decay: The decay to use. Note values close to 1 result in a slow decay
            whereas values close to 0 result in faster decay, tracking the input
            values more closely.
        """
        super(ExponentialMovingAverage, self).__init__()

        self._decay = decay

        self.register_buffer("_counter", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_hidden", None)
        self.register_buffer("average", None)

    def forward(self, value: torch.Tensor):
        """Applies EMA to the value given."""
        if self._hidden is None or self.average is None:
            self._initialize(value)

        value = self._synchronize_between_processes(value)
        self._counter.add_(1)
        self._hidden.sub_((self._hidden - value), alpha=(1. - self._decay))
        self.average.copy_(self._hidden / (1. - torch.pow(self._decay, self._counter)))

    def _synchronize_between_processes(self, value: torch.Tensor):

        if not is_dist_avail_and_initialized():
            return value

        return reduce_across_processes(value)

    @property
    def value(self) -> torch.Tensor:
        """Returns the current EMA."""
        return self.average.data

    def reset(self):
        self._counter.copy_(torch.zeros_like(self._counter))
        self._hidden.copy_(torch.zeros_like(self._hidden))
        self.average.copy_(torch.zeros_like(self.average))

    def _initialize(self, value: torch.Tensor):
        self._hidden = torch.zeros_like(value)
        self.average = torch.zeros_like(value)
