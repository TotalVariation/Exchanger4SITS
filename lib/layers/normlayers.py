# Distributed under MIT License.
from typing import Optional, Union, List, Tuple

import torch
import torch.nn.functional as F
import torch.linalg as LA
from torch import nn, Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from timm.models.layers import trunc_normal_, get_act_layer


class NormLayer(nn.Module):

    def __init__(self, norm_type, num_features, **kwargs):
        super(NormLayer, self).__init__()

        if norm_type == 'maskedbatchnorm':
            self.norm_layer = MaskedBatchNorm1d(num_features)
        elif norm_type == 'batchnorm':
            self.norm_layer = nn.BatchNorm1d(num_features)
        elif norm_type == 'layernorm':
            self.norm_layer = nn.LayerNorm(num_features)
        elif norm_type == 'rmsnorm':
            self.norm_layer = RMSNorm(num_features)
        elif norm_type == 'adanorm':
            adanorm_scale = kwargs.get('adanorm_scale', 1.0)
            self.norm_layer = AdaNorm(adanorm_scale)
        elif norm_type == 'identity':
            self.norm_layer = nn.Identity()
        else:
            raise NotImplementedError

        self.norm_type = norm_type

    def forward(
            self,
            x: Tensor,
            mask: Optional[Tensor] = None
    ):
        """
        Args Shape:
            -x: [B, T, C] or [B, C]
            -mask: [B, T]
        """

        if self.norm_type == 'maskedbatchnorm' and x.dim() == 3:
            assert mask is not None, f'requires mask is not None'
            assert mask.dim() == 2, f'expected dim of mask is 2, but got mask: {mask.shape}'
            mask = ~(mask.bool()) # for non-padded elements
            mask = mask.unsqueeze(dim=1).expand(-1, x.shape[-1], -1)
            x = x.transpose(-2, -1) # [B, T, C] --> [B, C, T]
            x = self.norm_layer(x, mask)
            x = x.transpose(-2, -1) # convert back
        elif self.norm_type == 'batchnorm' and x.dim() == 3:
            x = x.transpose(-2, -1) # [B, T, C] --> [B, C, T]
            x = self.norm_layer(x)
            x = x.transpose(-2, -1) # convert back
        else:
            x = self.norm_layer(x)

        return x


class AdaNorm(nn.Module):

    def __init__(self, adanorm_scale: float = 1.0, eps: float = 1e-5):
        super(AdaNorm, self).__init__()

        self.adanorm_scale = adanorm_scale
        self.eps = eps

    def forward(self, x: Tensor):

        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = x - mean

        graNorm = (0.1 * x / (std + self.eps)).detach()
        x_norm = (x - x * graNorm) / (std + self.eps)

        return x_norm * self.adanorm_scale


# Masked Batch Normalization
def masked_batch_norm(x: Tensor, mask: Tensor, weight: Optional[Tensor],
                      bias: Optional[Tensor], running_mean: Optional[Tensor],
                      running_var: Optional[Tensor], training: bool,
                      momentum: float, eps: float = 1e-5) -> Tensor:
    r"""Applies Masked Batch Normalization for each channel in each data sample in a batch.
    """
    if not training and (running_mean is None or running_var is None):
        raise ValueError('Expected running_mean and running_var to be not None when training=False')

    num_dims = len(x.shape[2:])
    _dims = (0,) + tuple(range(-num_dims, 0))
    _slice = (None, ...) + (None,) * num_dims

    if training:
        num_elements = mask.sum(_dims)
        mean = (x * mask).sum(_dims) / (num_elements + eps)  # (C,)
        x_masked_centered = (x - mean[_slice]) * mask
        var = (x_masked_centered * x_masked_centered).sum(_dims) / (num_elements + eps)  # (C,)

        if running_mean is not None:
            running_mean.copy_(running_mean * (1 - momentum) + momentum * mean.detach())
        if running_var is not None:
            running_var.copy_(running_var * (1 - momentum) + momentum * var.detach())
    else:
        mean, var = running_mean, running_var

    out = (x - mean[_slice]) / torch.sqrt(var[_slice] + eps) # (N, C, ...)

    if weight is not None and bias is not None:
        out = out * weight[_slice] + bias[_slice]

    return out


class _MaskedBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_MaskedBatchNorm, self).__init__(num_features, eps, momentum, affine,
                                               track_running_stats)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        self._check_input_dim(x)
        if mask is not None:
            self._check_input_dim(mask)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        if mask is None:
            return F.batch_norm(
                x,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean if not self.training or self.track_running_stats else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.weight, self.bias, bn_training, exponential_average_factor, self.eps
            )
        else:
            return masked_batch_norm(
                x, mask, self.weight, self.bias,
                self.running_mean if not self.training or self.track_running_stats else None,
                self.running_var if not self.training or self.track_running_stats else None,
                bn_training, exponential_average_factor, self.eps
            )


class MaskedBatchNorm1d(torch.nn.BatchNorm1d, _MaskedBatchNorm):
    r"""Applies Batch Normalization over a masked 3D input
    (a mini-batch of 1D inputs with additional channel dimension)..
    See documentation of :class:`~torch.nn.BatchNorm1d` for details.
    Shape:
        - Input: :math:`(N, C, L)`
        - Mask: :math:`(N, 1, L)`
        - Output: :math:`(N, C, L)` (same shape as input)
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True) -> None:
        super(MaskedBatchNorm1d, self).__init__(num_features, eps, momentum, affine,
                                                track_running_stats)


class RMSNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-8):
        """
        Root Mean Square Layer Normalization
        :param dim: model size
        :param eps:  epsilon value, default 1e-8
        No Bias as RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.num_features = num_features

        self.scale = nn.parameter.Parameter(torch.ones(num_features))

    def forward(self, x: torch.Tensor, **kwargs):

        norm_x = LA.vector_norm(x, ord=2, dim=-1, keepdim=True)
        rms_x = norm_x * self.num_features ** -0.5
        x_normed = x / (rms_x + self.eps)

        return self.scale * x_normed


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False,
                 detach: bool = False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.detach = detach
        if self.affine:
            self._init_params()

    def forward(self, x: torch.Tensor, mode: str):

        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError

        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        # Expected Input Shape: [B, ..., C]
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True)
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps)
        if self.detach:
            # mean & stdev have gradient normalization effects
            self.mean = self.mean.detach()
            self.stdev = self.stdev.detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
