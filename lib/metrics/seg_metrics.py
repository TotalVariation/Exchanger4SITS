"""
Taken from https://github.com/davidtvs/PyTorch-ENet/blob/master/metric/confusionmatrix.py
"""

import numpy as np
import torch

from .cls_metrics import *
from utils.dist_helper import reduce_across_processes


class IoU(Metric):
    """Computes the intersection over union (IoU) per class and corresponding
    mean (mIoU).

    Intersection over union (IoU) is a common evaluation metric for semantic
    segmentation. The predictions are first accumulated in a confusion matrix
    and the IoU is computed from it as follows:

        IoU = true_positive / (true_positive + false_positive + false_negative).

    Keyword arguments:
    - num_classes (int): number of classes in the classification problem
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.
    - ignore_index (int or iterable, optional): Index of the classes to ignore
    when computing the IoU. Can be an int, or any iterable of ints.
    """

    def __init__(self, num_classes, normalized=False, ignore_index=None, cm_device='cpu', lazy=True):
        super().__init__()
        self.conf_metric = ConfusionMatrix(num_classes, normalized, device=cm_device, lazy=lazy)
        self.lazy = lazy
        self.ignore_index = ignore_index

    def reset(self):
        self.conf_metric.reset()

    def add(self, predicted, target):
        """Adds the predicted and target pair to the IoU metric.

        Keyword arguments:
        - predicted (Tensor): Can be a (N, K, H, W) tensor of
        predicted scores obtained from the model for N examples and K classes,
        or (N, H, W) tensor of integer values between 0 and K-1.
        - target (Tensor): Can be a (N, K, H, W) tensor of
        target scores for N examples and K classes, or (N, H, W) tensor of
        integer values between 0 and K-1.

        """
        # Dimensions check
        assert predicted.size(0) == target.size(0), \
            'number of targets and predicted outputs do not match'
        assert predicted.dim() == 3 or predicted.dim() == 4, \
            "predictions must be of dimension (N, H, W) or (N, K, H, W)"
        assert target.dim() == 3 or target.dim() == 4, \
            "targets must be of dimension (N, H, W) or (N, K, H, W)"

        # If the tensor is in categorical format convert it to integer format
        if predicted.dim() == 4:
            _, predicted = predicted.max(1)
        if target.dim() == 4:
            _, target = target.max(1)

        if self.ignore_index is not None:
            keep = target != self.ignore_index
            predicted = predicted[keep]
            target = target[keep]

        self.conf_metric.add(predicted.view(-1), target.view(-1))

    def value(self):
        """Computes the IoU and mean IoU.

        The mean computation ignores NaN elements of the IoU array.

        Returns:
            Dict: (IoU, mIoU, Acc, OA). The first output is the per class IoU,
            for K classes it's numpy.ndarray with K elements. The second output,
            is the mean IoU. The third output is the per class accuracy. The fourth
            output is the overall accuracy.
        """
        conf_matrix = self.conf_metric.value()
        conf_matrix = reduce_across_processes(conf_matrix)
        if torch.is_tensor(conf_matrix):
            conf_matrix = conf_matrix.cpu().numpy()

        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, 0) - true_positive
        false_negative = np.sum(conf_matrix, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)
            acc = true_positive / np.sum(conf_matrix, 1)

        all_acc = true_positive.sum() / conf_matrix.sum()

        metrics = {'IoU': iou, 'mIoU': np.nanmean(iou), 'Acc': acc, 'OA': all_acc}

        return metrics
