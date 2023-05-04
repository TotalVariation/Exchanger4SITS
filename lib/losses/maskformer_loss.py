from typing import List, Tuple, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .matcher import HungarianMatcher, SegMatcher
from .criterion import SetCriterion


class MaskFormerLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        num_dec_layers,
        task_type='maskformer_pano',
        mask_weight=20.0,
        dice_weight=1.0,
        cls_weight=1.0,
        loss_weight=1.0,
        num_points=128,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        no_object_weight=0.1,
        ignore_index=-1
    ):

        super(MaskFormerLoss, self).__init__()

        weight_dict = {"loss_ce": cls_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
        aux_weight_dict = {}
        for i in range(num_dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

        if task_type == 'maskformer_pano':
            matcher = HungarianMatcher(
                cost_class=cls_weight,
                cost_mask=mask_weight,
                cost_dice=dice_weight,
                num_points=int(num_points * oversample_ratio)
            )
        elif task_type == 'sem_seg':
            matcher = SegMatcher() # Naive Match
        else:
            raise NotImplementedError

        self.criterion = SetCriterion(
            num_classes,
            matcher,
            weight_dict=weight_dict,
            losses=["labels", "masks"],
            eos_coef=no_object_weight,
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio
        )

        self.task_type = task_type
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(
            self,
            outputs: Dict[str, Tensor],
            labels: Tensor
    ) -> Dict[str, Tensor]:

        targets = self.prepare_targets(labels)
        losses = self.criterion(outputs, targets)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] = losses[k] * self.criterion.weight_dict[k] * self.loss_weight
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        return losses

    def prepare_targets(self, targets):
        targets = targets.long()
        if self.task_type == 'sem_seg':
            # decouple category id and binary mask
            new_targets = []
            for targets_per_image in targets:
                """
                targets shape: [B, H, W]
                """
                # gt_cls
                gt_cls = targets_per_image.unique(sorted=True) # deterministic
                gt_cls = gt_cls[gt_cls != self.ignore_index]
                masks = []
                for cls in gt_cls:
                    masks.append(targets_per_image == cls)
                if len(gt_cls) == 0:
                    masks.append(targets_per_image == self.ignore_index)

                masks = torch.stack(masks, dim=0)
                new_targets.append(
                    {
                        "labels": gt_cls,
                        "masks": masks,
                    }
                )
        elif self.task_type == 'maskformer_pano':
            # decouple category id and binary mask
            new_targets = []
            for targets_per_image in targets:
                instance_ids = targets_per_image[0]
                pixel_sem_map = targets_per_image[1]

                gt_cls = []
                masks = []
                # background stuff class
                gt_cls.append(torch.tensor(0, dtype=torch.long, device=targets.device))
                masks.append(pixel_sem_map == 0)
                for instance_id in instance_ids.unique(sorted=True):
                    if instance_id == 0: # skip background
                        continue
                    class_id = pixel_sem_map[instance_ids == instance_id].unique()
                    class_id = [c for c in class_id if c != self.ignore_index]
                    if len(class_id) > 0:
                        assert len(class_id) == 1, f'class id should be unique but got {class_id}'
                        class_id = class_id[0]
                        gt_cls.append(class_id)
                        masks.append(instance_ids == instance_id)

                gt_cls = torch.stack(gt_cls)
                masks = torch.stack(masks, dim=0)
                new_targets.append(
                    {
                        "labels": gt_cls,
                        "masks": masks,
                    }
                )
        else:
            raise NotImplementedError

        return new_targets
