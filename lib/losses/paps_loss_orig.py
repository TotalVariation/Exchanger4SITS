"""
PaPs Implementation
Author: Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
Modified by X.Cai
"""

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class PaPsLoss(nn.Module):
    def __init__(
        self,
        l_center=1,
        l_size=1,
        l_shape=1,
        l_class=1,
        alpha=0,
        beta=4,
        gamma=0,
        eps=1e-8,
        ignore_index=-100,
        binary_threshold=0.4,
    ):
        """
        Loss for training PaPs.
        Args:
            l_center (float): Coefficient for the centerness loss (default 1)
            l_size (float): Coefficient for the size loss (default 1)
            l_shape (float): Coefficient for the shape loss (default 1)
            l_class (float): Coefficient for the classification loss (default 1)
            alpha (float): Parameter for the centerness loss (default 0)
            beta (float): Parameter for the centerness loss (default 4)
            gamma (float): Focal exponent for the classification loss (default 0)
            eps (float): Stability epsilon
            void_label (int): Label to ignore in the classification loss
        """
        super(PaPsLoss, self).__init__()
        self.l_center = l_center
        self.l_size = l_size
        self.l_shape = l_shape
        self.l_class = l_class
        self.eps = eps
        self.binary_threshold = binary_threshold

        self.center_loss = CenterLoss(alpha=alpha, beta=beta, eps=eps)
        self.class_loss = FocalLoss(gamma=gamma, ignore_label=ignore_index)
        self.shape_loss = FocalLoss(gamma=0)
        #self.value = (0, 0, 0, 0, 0)

        # Keep track of the predicted confidences and ious between predicted and gt binary masks.
        # This is usefull for tuning the confidence threshold of the pseudo-nms.
        #self.predicted_confidences = None
        #self.achieved_ious = None

    def forward(self, predictions, target, **kwargs):

        (
            target_heatmap,
            true_instances,
            zones,
            size,
            sem_obj,
            sem_pix,
        ) = target.split((1, 1, 1, 2, 1, 1), dim=1)

        target_heatmap = target_heatmap.squeeze(dim=1)
        true_instances = true_instances.squeeze(dim=1)
        zones = zones.squeeze(dim=1)
        size = size.permute(0, 2, 3, 1)
        sem_obj = sem_obj.squeeze(dim=1)

        center_mask = predictions["center_mask"] # [B, H, W]
        center_mapping = {
            (int(b), int(i), int(j)): k
            for k, (b, i, j) in enumerate(torch.nonzero(center_mask, as_tuple=False))
        }

        loss_center = 0
        loss_size = 0
        loss_shape = 0
        loss_class = 0

        if self.l_center != 0:
            loss_center = self.center_loss(predictions["heatmap"], target_heatmap)

        if predictions["size"].shape[0] != 0:
            if self.l_size != 0:
                true_size = size[center_mask]  # (N, 2)
                loss_size = torch.abs(true_size - predictions["size"]) / (
                    true_size + self.eps
                )
                loss_size = loss_size.sum(dim=-1).mean()

            if self.l_class != 0:
                loss_class = self.class_loss(
                    predictions["semantic"],
                    sem_obj[center_mask].long(),
                )

            if self.l_shape != 0:
                #confidence_pred = []
                #ious = []
                flatten_preds = []
                flatten_targets = []
                for b, instance_mask in enumerate(true_instances):
                    for inst_id in torch.unique(instance_mask):
                        # center matching
                        centers = center_mask[b] * (zones[b] == inst_id)
                        if not centers.any():
                            continue
                        for x, y in centers.nonzero(as_tuple=False):
                            true_mask = (instance_mask == inst_id).float()

                            pred_id = center_mapping[(int(b), int(x), int(y))]
                            xtl, ytl, xbr, ybr = predictions["instance_boxes"][pred_id]
                            crop_true = true_mask[ytl:ybr, xtl:xbr].reshape(-1, 1)
                            mask = predictions["instance_masks"][pred_id].reshape(-1, 1)

                            flatten_preds.append(mask)
                            flatten_targets.append(crop_true)

                            """
                            # For DEBUG
                            confidence_pred.append(predictions["confidence"][pred_id])
                            bmask = mask > self.binary_threshold
                            inter = (bmask * crop_true).sum().float()
                            union = ((bmask + crop_true) != 0).sum()
                            true_mask[ytl:ybr, xtl:xbr] = 0
                            union = (
                                union + true_mask.sum()
                            )  # parts of shape outside of bbox
                            iou = inter / union
                            if torch.isnan(iou) or torch.isinf(iou):
                                iou.zero_()
                            ious.append(iou)
                            """

                if len(flatten_targets) > 0:
                    p = torch.cat(flatten_preds, dim=0)
                    preds = torch.cat([1 - p, p], dim=1)
                    t = torch.cat(flatten_targets, dim=0).long()
                    loss_shape = self.shape_loss(preds, t)
                else:
                    loss_shape = 0.

                #self.predicted_confidences = torch.stack(confidence_pred)
                #self.achieved_ious = torch.stack(ious).unsqueeze(-1)

        losses = {
            'loss_center': self.l_center * loss_center,
            'loss_size': self.l_size * loss_size,
            'loss_shape': self.l_shape * loss_shape,
            'loss_class': self.l_class * loss_class,
        }

        """
        self.value = (
            float(loss_center.detach().cpu())
            if isinstance(loss_center, torch.Tensor)
            else loss_center,
            float(loss_size.detach().cpu())
            if isinstance(loss_size, torch.Tensor)
            else loss_size,
            float(loss_shape.detach().cpu())
            if isinstance(loss_shape, torch.Tensor)
            else loss_shape,
            float(loss_class.detach().cpu())
            if isinstance(loss_class, torch.Tensor)
            else loss_class,
        )
        """
        return losses


class CenterLoss(nn.Module):
    """
    Adapted from the github repo of the CornerNet paper
    https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py
    """

    def __init__(self, alpha=0, beta=4, eps=1e-8):
        super(CenterLoss, self).__init__()
        self.a = alpha
        self.b = beta
        self.eps = eps

    def forward(self, preds, gt):
        #preds = preds.permute(0, 2, 3, 1).contiguous().view(-1, preds.shape[1])
        #g = gt.view(-1, preds.shape[1])
        preds = preds.flatten()
        g = gt.flatten()

        pos_inds = g.eq(1)
        neg_inds = g.lt(1)
        num_pos = pos_inds.float().sum()
        loss = 0

        pos_loss = torch.log(preds[pos_inds] + self.eps)
        pos_loss = pos_loss * torch.pow(1 - preds[pos_inds], self.a)
        pos_loss = pos_loss.sum()

        neg_loss = torch.log(1 - preds[neg_inds] + self.eps)
        neg_loss = neg_loss * torch.pow(preds[neg_inds], self.a)
        neg_loss = neg_loss * torch.pow(1 - g[neg_inds], self.b)
        neg_loss = neg_loss.sum()

        if preds[pos_inds].nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True, ignore_label=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, preds, target):
        if preds.dim() > 2:
            #preds = preds.view(preds.size(0), preds.size(1), -1)  # N,C,H,W => N,C,H*W
            #preds = preds.transpose(1, 2)  # N,C,H*W => N,H*W,C
            #preds = preds.contiguous().view(-1, preds.size(2))  # N,H*W,C => N*H*W,C
            preds = preds.flatten(2).transpose(1, 2).flatten(0, 1)
        target = target.view(-1, 1)

        if self.ignore_label is not None:
            keep = target[:, 0] != self.ignore_label
            preds = preds[keep, :]
            target = target[keep, :]

        if preds.squeeze(1).dim() == 1:
            logpt = torch.sigmoid(preds)
            logpt = logpt.view(-1)
        else:
            logpt = F.log_softmax(preds, dim=-1)
            logpt = logpt.gather(1, target)
            logpt = logpt.view(-1)
        pt = logpt.exp()

        """
        if self.alpha is not None:
            if self.alpha.type() != preds.data.type():
                self.alpha = self.alpha.type_as(preds.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)
        """

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
