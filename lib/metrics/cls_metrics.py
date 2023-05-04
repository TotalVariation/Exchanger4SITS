"""
Taken from https://github.com/davidtvs/PyTorch-ENet/blob/master/metric/confusionmatrix.py
"""

import numpy as np

import torch

from utils.dist_helper import reduce_across_processes


class Metric(object):
    """Base class for all metrics.
    From: https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py
    """

    def reset(self):
        pass

    def add(self):
        pass

    def value(self):
        pass


class ConfusionMatrix(Metric):
    """Constructs a confusion matrix for a multi-class classification problems.

    Does not support multi-label, multi-class problems.

    Keyword arguments:
    - num_classes (int): number of classes in the classification problem.
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.

    Modified from: https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py
    """

    def __init__(self, num_classes, normalized=False, device='cpu', lazy=True):
        super().__init__()
        if device == 'cpu':
            self.conf = np.ndarray((num_classes, num_classes), dtype=np.int64)
        else:
            self.conf = torch.zeros((num_classes, num_classes)).cuda()
        self.normalized = normalized
        self.num_classes = num_classes
        self.device = device
        self.reset()
        self.lazy = lazy

    def reset(self):
        if self.device == 'cpu':
            self.conf.fill(0)
        else:
            self.conf = torch.zeros(self.conf.shape).cuda()

    def add(self, predicted, target):
        """Computes the confusion matrix

        The shape of the confusion matrix is K x K, where K is the number
        of classes.

        Keyword arguments:
        - predicted (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        predicted scores obtained from the model for N examples and K classes,
        or an N-tensor/array of integer values between 0 and K-1.
        - target (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        ground-truth classes for N examples and K classes, or an N-tensor/array
        of integer values between 0 and K-1.

        """

        # If target and/or predicted are tensors, convert them to numpy arrays
        if self.device == 'cpu':
            if torch.is_tensor(predicted):
                predicted = predicted.cpu().numpy()
            if torch.is_tensor(target):
                target = target.cpu().numpy()

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if len(predicted.shape) != 1:
            assert predicted.shape[1] == self.num_classes, \
                'number of predictions does not match size of confusion matrix'
            predicted = predicted.argmax(1)
        else:
            if not self.lazy:
                assert (predicted.max() < self.num_classes) and (predicted.min() >= 0), \
                    'predicted values are not between 0 and k-1'

        if len(target.shape) != 1:
            if not self.lazy:
                assert target.shape[1] == self.num_classes, \
                    'Onehot target does not match size of confusion matrix'
                assert (target >= 0).all() and (target <= 1).all(), \
                    'in one-hot encoding, target values should be 0 or 1'
                assert (target.sum(1) == 1).all(), \
                    'multi-label setting is not supported'
            target = target.argmax(1)
        else:
            if not self.lazy:
                assert (target.max() < self.num_classes) and (target.min() >= 0), \
                    'target values are not between 0 and k-1'

        # hack for bincounting 2 arrays together
        x = predicted + self.num_classes * target

        if self.device == 'cpu':
            bincount_2d = np.bincount(
                x.astype(np.int64), minlength=self.num_classes ** 2)
            assert bincount_2d.size == self.num_classes ** 2
            conf = bincount_2d.reshape((self.num_classes, self.num_classes))
        else:
            bincount_2d = torch.bincount(
                x, minlength=self.num_classes ** 2)

            conf = bincount_2d.view((self.num_classes, self.num_classes))
        self.conf += conf

    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf


def cal_f_score(precision, recall, beta=1):
    """calculate the f-score value.
    Args:
        precision (float | torch.Tensor): The precision value.
        recall (float | torch.Tensor): The recall value.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.
    Returns:
        [torch.tensor]: The f-score value.
    """
    score = (1 + beta**2) * (precision * recall) / (
        (beta**2 * precision) + recall)
    return score


class Precision_Recall_Fscore(Metric):

    def __init__(self, num_classes, normalized=False, ignore_index=None, cm_device='cpu', lazy=True):
        super().__init__()
        self.conf_metric = ConfusionMatrix(num_classes, normalized, device=cm_device, lazy=lazy)
        self.lazy = lazy
        self.ignore_index = ignore_index

    def reset(self):
        self.conf_metric.reset()

    def add(self, predicted, target):
        if self.ignore_index is not None:
            keep = target != self.ignore_index
            predicted = predicted[keep]
            target = target[keep]
        self.conf_metric.add(predicted, target)

    def value(self,):
        conf_matrix = self.conf_metric.value() # unnormalized confusion matrix
        conf_matrix = reduce_across_processes(conf_matrix)
        if torch.is_tensor(conf_matrix):
            conf_matrix = conf_matrix.cpu().numpy()

        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, 0) - true_positive
        false_negative = np.sum(conf_matrix, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            f_score = cal_f_score(precision, recall, beta=1)
            IoU = true_positive / (true_positive + false_positive + false_negative)

        OA = np.sum(true_positive) / np.sum(conf_matrix)

        metrics = {
            'Precision': np.nanmean(precision),
            'Recall': np.nanmean(recall),
            'F_score': np.nanmean(f_score),
            'mIoU': np.nanmean(IoU),
            'OA': OA
        }

        return metrics


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]
