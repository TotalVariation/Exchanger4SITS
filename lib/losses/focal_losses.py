from typing import Optional

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, Tensor


def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean()


class FocalCELoss(nn.Module):
    """
    FocalLoss copied from github.com/VSainteuf/utae-paps
    """
    def __init__(self, gamma=1.0, size_average=True, ignore_index: int = -100, weight: Optional[Tensor] = None):
         super(FocalCELoss, self).__init__()
         self.gamma = gamma
         self.size_average = size_average
         self.ignore_index = ignore_index
         self.weight = weight

    def forward(self, preds, target):
        # preds shape (B, C), target shape (B,)
        target = target.view(-1,1)

        if preds.ndim > 2: # e.g., (B, C, H, W)
            preds = preds.permute(0, 2, 3, 1).flatten(0, 2)

        keep = target[:, 0] != self.ignore_index
        preds = preds[keep, :]
        target = target[keep, :]

        logpt = F.log_softmax(preds, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.weight is not None:
            w = self.weight.expand_as(preds)
            w = w.gather(1, target)
            loss = -1 * (1 - pt) ** self.gamma * w * logpt
        else:
            loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

