from typing import List, Tuple, Dict, Optional

import numpy as np

import torch
from torch import nn, Tensor

from .cls_metrics import Metric
from utils.dist_helper import reduce_across_processes


class PanopticMeter(Metric):

    def __init__(
            self, num_classes, background_label=0, void_label=19,
            iou_threshold=0.5, device='cpu', target_type='paps_pano',
    ):
        """
        Meter class for the panoptic metrics as defined by Kirilov et al. :
        Segmentation Quality (SQ)
        Recognition Quality (RQ)
        Panoptic Quality (PQ)
        The behavior of this meter mimics that of torchnet meters, each predicted batch
        is added via the add method and the global metrics are retrieved with the value
        method.
        Args:
            num_classes (int): Number of semantic classes (including background and void class).
            void_label (int): Label for the void class (default 19).
            background_label (int): Label for the background class (default 0).
            iou_threshold (float): Threshold used on the IoU of the true vs predicted
            instance mask. Above the threshold a true instance is counted as True Positive.
        """
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.target_type = target_type
        self.class_list = [c for c in range(num_classes) if c != background_label]
        self.void_label = void_label
        if void_label is not None:
            self.class_list = [c for c in self.class_list if c != void_label]
        self.counts = torch.zeros((len(self.class_list), 3), device=device) # only for things
        self.cumulative_ious = torch.zeros(len(self.class_list), device=device)

    def reset(self):
        nn.init.zeros_(self.counts)
        nn.init.zeros_(self.cumulative_ious)

    def add(self, outputs: Dict[str, Tensor], target: Tensor):

        if self.target_type == 'paps_pano':
            (
                target_heatmap,
                true_instances,
                zones,
                size,
                sem_obj,
                sem_pix,
            ) = target.split((1, 1, 1, 2, 1, 1), dim=1)
        elif self.target_type == 'maskformer_pano':
            (
                true_instances,
                sem_pix,
             ) = target.split((1, 1), dim=1)
        else:
            raise NotImplementedError

        instance_true = true_instances.squeeze(1)
        semantic_true = sem_pix.squeeze(1)

        instance_pred = outputs["pano_instance"]
        semantic_pred = outputs["pano_semantic"]
        if semantic_pred.ndim == 4:
            semantic_pred = semantic_pred.softmax(dim=1).argmax(dim=1)

        if self.void_label is not None:
            void_masks = semantic_true == self.void_label

            # Ignore Void Objects/Instances
            for batch_idx, void_mask in enumerate(void_masks):
                if void_mask.any():
                    for void_inst_id, void_inst_area in zip(
                        *torch.unique(instance_true[batch_idx] * void_mask, return_counts=True)
                    ):
                        # skip background class (stuff)
                        if void_inst_id == 0:
                            continue
                        for pred_inst_id, pred_inst_area in zip(
                            *torch.unique(instance_pred[batch_idx], return_counts=True)
                        ):
                            if pred_inst_id == 0:
                                continue
                            inter = (
                                (instance_true[batch_idx] == void_inst_id)
                                * (instance_pred[batch_idx] == pred_inst_id)
                            ).sum()
                            iou = (
                                inter.float()
                                / (void_inst_area + pred_inst_area - inter).float()
                            )
                            if iou > self.iou_threshold:
                                instance_pred[batch_idx][
                                    instance_pred[batch_idx] == pred_inst_id
                                ] = 0
                                semantic_pred[batch_idx][
                                    instance_pred[batch_idx] == pred_inst_id
                                ] = self.void_label

            # Ignore Void Pixels
            instance_pred[void_masks] = 0
            semantic_pred[void_masks] = self.void_label

        # Compute metrics for each thing class
        for i, class_id in enumerate(self.class_list):
            TP = 0
            n_preds = 0
            n_true = 0
            ious = []
            for batch_idx, instance_mask in enumerate(instance_true):
                class_mask_gt = semantic_true[batch_idx] == class_id
                class_mask_p = semantic_pred[batch_idx] == class_id
                n_preds += (
                    int(torch.unique(instance_pred[batch_idx] * class_mask_p).shape[0])
                    - 1
                )  # do not count 0 (masked zones)
                n_true += int(torch.unique(instance_mask * class_mask_gt).shape[0]) - 1
                if n_preds == 0 or n_true == 0:
                    continue  # no true positives in that case

                for true_inst_id, true_inst_area in zip(
                    *torch.unique(instance_mask * class_mask_gt, return_counts=True)
                ):
                    if true_inst_id == 0:  # masked segments
                        continue
                    for pred_inst_id, pred_inst_area in zip(
                        *torch.unique(
                            instance_pred[batch_idx] * class_mask_p, return_counts=True
                        )
                    ):
                        if pred_inst_id == 0:
                            continue
                        inter = (
                            (instance_mask == true_inst_id)
                            * (instance_pred[batch_idx] == pred_inst_id)
                        ).sum()
                        iou = (
                            inter.float()
                            / (true_inst_area + pred_inst_area - inter).float()
                        )

                        if iou > self.iou_threshold:
                            TP += 1
                            ious.append(iou)
            FP = n_preds - TP
            FN = n_true - TP

            self.counts[i] += torch.tensor([TP, FP, FN], device=target.device)
            if len(ious) > 0:
                self.cumulative_ious[i] += torch.stack(ious).sum()

    def value(self):
        counts = reduce_across_processes(self.counts)
        cum_ious = reduce_across_processes(self.cumulative_ious)
        if torch.is_tensor(counts):
            counts = counts.cpu().numpy()
        if torch.is_tensor(cum_ious):
            cum_ious = cum_ious.cpu().numpy()
        TP, FP, FN = np.split(counts, 3, axis=1)
        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):
            SQ = cum_ious / TP[:, 0]
            #SQ[np.isnan(SQ) | np.isinf(SQ)] = 0  # if TP==0
            RQ = TP / (TP + 0.5 * FP + 0.5 * FN)
            RQ = RQ[:, 0]
        PQ = SQ * RQ

        metrics = {'SQ': np.nanmean(SQ), 'RQ': np.nanmean(RQ), 'PQ': np.nanmean(PQ)}

        return metrics

    def get_table(self):
        table = (
            torch.cat([self.counts, self.cumulative_ious[:, None]], dim=-1).cpu().numpy()
        )
        return table
