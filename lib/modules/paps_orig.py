"""
PaPs Implementation
Author: Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""
from typing import List, Tuple, Dict, Optional

import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch_scatter import scatter_max

from timm.models.layers import trunc_normal_

from losses import PaPsLoss


class PaPs(nn.Module):
    def __init__(self, config, **kwargs):
        """
        Implementation of the Parcel-as-Points Module (PaPs) for panoptic segmentation of agricultural
        parcels from satellite image time series.
        Args:
            encoder (nn.Module): Backbone encoding network. The encoder is expected to return
            a feature map at the same resolution as the input images and a list of feature maps
            of lower resolution.
            num_classes (int): Number of classes (including stuff and void classes).
            shape_size (int): S hyperparameter defining the shape of the local patch.
            mask_conv (bool): If False no residual CNN is applied after combination of
            the predicted shape and the cropped saliency (default True)
            min_confidence (float): Cut-off confidence level for the pseudo NMS (predicted instances with
            lower condidence will not be included in the panoptic prediction).
            min_remain (float): Hyperparameter of the pseudo-NMS that defines the fraction of a candidate instance mask
            that needs to be new to be included in the final panoptic prediction (default  0.5).
            mask_threshold (float): Binary threshold for instance masks (default 0.4)

        """
        super(PaPs, self).__init__()

        spec_dict = {k.lower(): v for k, v in dict(config.PAPS_HEAD).items()}
        spec_dict.update(**kwargs)
        loss_center_weight = spec_dict['loss_center_weight']
        loss_size_weight = spec_dict['loss_size_weight']
        loss_shape_weight = spec_dict['loss_shape_weight']
        loss_class_weight = spec_dict['loss_class_weight']
        binary_threshold = spec_dict['binary_threshold']
        center_loss_alpha = spec_dict['center_loss_alpha']
        center_loss_beta = spec_dict['center_loss_beta']
        focal_loss_gamma = spec_dict['focal_loss_gamma']
        mask_conv = spec_dict['mask_conv']
        enc_dim = spec_dict['enc_dim']
        stack_dim = spec_dict['stack_dim']
        ignore_index = config.LOSS.IGNORE_INDEX

        self.shape_size = spec_dict['shape_size']
        self.min_confidence = spec_dict['min_confidence']
        self.min_remain = spec_dict['min_remain']
        self.mask_threshold = spec_dict['mask_threshold']
        self.num_classes = config.DATASET.NUM_CLASSES

        self.center_extractor = CenterExtractor()

        self.heatmap_conv = ConvLayer(
            nkernels=[enc_dim, 32, 1], last_relu=False, k=3, p=1,
            padding_mode="reflect"
        )

        self.saliency_conv = ConvLayer(
            nkernels=[enc_dim, 32, 1], last_relu=False, k=3, p=1,
            padding_mode="reflect"
        )

        self.shape_mlp = nn.Sequential(
            nn.Linear(stack_dim, stack_dim // 2),
            nn.BatchNorm1d(stack_dim // 2),
            nn.ReLU(),
            nn.Linear(stack_dim // 2, self.shape_size ** 2),
        )

        self.size_mlp = nn.Sequential(
            nn.Linear(stack_dim, stack_dim // 2),
            nn.BatchNorm1d(stack_dim // 2),
            nn.ReLU(),
            nn.Linear(stack_dim // 2, stack_dim // 4),
            nn.BatchNorm1d(stack_dim // 4),
            nn.ReLU(),
            nn.Linear(stack_dim // 4, 2),
            nn.Softplus(),
        )

        self.class_mlp = nn.Sequential(
            nn.Linear(stack_dim, stack_dim // 2),
            nn.BatchNorm1d(stack_dim // 2),
            nn.ReLU(),
            nn.Linear(stack_dim // 2, stack_dim // 4),
            nn.Linear(stack_dim // 4, self.num_classes),
        )

        if mask_conv:
            self.mask_cnn = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
                nn.GroupNorm(num_channels=16, num_groups=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1),
            )
        else:
            self.mask_cnn = None

        self.loss_decode = PaPsLoss(
            l_center=loss_center_weight,
            l_size=loss_size_weight,
            l_shape=loss_shape_weight,
            l_class=loss_class_weight,
            alpha=center_loss_alpha,
            beta=center_loss_beta,
            gamma=focal_loss_gamma,
            ignore_index=ignore_index,
            binary_threshold=binary_threshold
        )

        self.apply(self._reset_parameters)
        self._init_head()

    def _reset_parameters(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=0.02, a=-0.02, b=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)

    def _init_head(self, r=0.01):
        # for focal loss
        for layer in [self.heatmap_conv, self.saliency_conv]:
            m = getattr(layer.conv, str(len(layer.conv) - 1))
            nn.init.constant_(m.bias, -1.0 * math.log((1 - r) / r))
        for layer in [self.shape_mlp, self.class_mlp]:
            m = getattr(layer, str(len(layer) - 1))
            nn.init.constant_(m.bias, -1.0 * math.log((1 - r) / r))

    def losses(self, outputs: Dict[str, Tensor], targets, **kwargs) -> Dict[str, Tensor]:
        return self.loss_decode(outputs, targets)

    def forward(
        self,
        x: Tensor,
        feat_maps: List[Tensor],
        zones: Optional[Tensor] = None,
        pseudo_nms: bool = True,
        **kwargs,
    ) -> Dict[str, Tensor]:

        # Global Predictions
        heatmap = self.heatmap_conv(x).sigmoid()
        saliency = self.saliency_conv(x)

        center_mask, _ = self.center_extractor(
            heatmap,
            zones=zones,
        )  # (B, H, W) mask of N detected centers

        # Retrieve info of detected centers
        B, H, W = center_mask.shape
        center_batch, center_h, center_w = torch.where(center_mask)
        center_positions = torch.stack([center_h, center_w], dim=1)

        # Construct multi-level feature stack for centers
        stack = []
        for i, m in enumerate(feat_maps):
            # Assumes resolution is divided by 2 at each level
            h_mask = center_h.div(2 ** i, rounding_mode='floor')
            w_mask = center_w.div(2 ** i, rounding_mode='floor')
            m = m.permute(0, 2, 3, 1)
            stack.append(m[center_batch, h_mask.long(), w_mask.long()])
        stack = torch.cat(stack, dim=1)

        # Center-level predictions
        size = self.size_mlp(stack)
        sem = self.class_mlp(stack)
        shapes = self.shape_mlp(stack)
        shapes = shapes.view((-1, 1, self.shape_size, self.shape_size))
        # (N,1,S,S) instance shapes

        centerness = heatmap[center_mask[:, None, :, :]]
        confidence = centerness

        # Instance Boxes Assembling
        # Minimal box size of 1px
        # Combine clamped sizes and center positions to obtain box coordinates
        clamp_size = size.detach().round().long().clamp_min(min=1)
        half_size = clamp_size.div(2, rounding_mode='floor')
        remainder_size = clamp_size % 2
        start_hw = center_positions - half_size
        stop_hw = center_positions + half_size + remainder_size

        instance_boxes = torch.cat([start_hw, stop_hw], dim=1)
        instance_boxes.clamp_(min=0, max=H)
        instance_boxes = instance_boxes[:, [1, 0, 3, 2]]  # h,w,h,w to x,y,x,y

        valid_start = (-start_hw).clamp_(
            min=0
        )  # if h=-5 crop the shape mask before the 5th pixel
        valid_stop = (stop_hw - start_hw) - (stop_hw - H).clamp_(
            min=0
        )  # crop if h_stop > H

        # Instance Masks Assembling
        instance_masks = []
        for i, s in enumerate(shapes.split(1, dim=0)):
            h, w = clamp_size[i]  # Box size
            w_start, h_start, w_stop, h_stop = instance_boxes[i]  # Box coordinates
            h_start_valid, w_start_valid = valid_start[i]  # Part of the Box that lies
            h_stop_valid, w_stop_valid = valid_stop[i]  # within the image's extent

            ## Resample local shape mask
            pred_mask = (
                F.interpolate(s, size=(h.item(), w.item()), mode="bilinear")
            ).squeeze(0)
            pred_mask = pred_mask[
                :, h_start_valid:h_stop_valid, w_start_valid:w_stop_valid
            ]

            ## Crop saliency
            crop_saliency = saliency[center_batch[i], :, h_start:h_stop, w_start:w_stop]

            ## Combine both
            if self.mask_cnn is None:
                pred_mask = torch.sigmoid(pred_mask) * torch.sigmoid(crop_saliency)
            else:
                pred_mask = pred_mask + crop_saliency
                pred_mask = torch.sigmoid(pred_mask) * torch.sigmoid(
                    self.mask_cnn(pred_mask.unsqueeze(0)).squeeze(0)
                )
            instance_masks.append(pred_mask.squeeze(0))

        # PSEUDO-NMS
        if pseudo_nms:
            panoptic_instance = []
            panoptic_semantic = []
            for b in range(B):  # iterate over elements of batch
                panoptic_mask = torch.zeros((H, W), device=center_mask.device)
                semantic_mask = torch.zeros((self.num_classes, H, W), device=center_mask.device)

                # get indices of centers in this batch element
                candidates = torch.where(center_batch == b)[0]
                for n, (c, idx) in enumerate(
                    zip(*torch.sort(confidence[candidates], descending=True))
                ):
                    if c < self.min_confidence:
                        break
                    else:
                        new_mask = torch.zeros((H, W), device=center_mask.device)
                        pred_mask_bin = (
                            instance_masks[candidates[idx]] > self.mask_threshold
                        ).float()

                        if pred_mask_bin.sum() > 0:
                            xtl, ytl, xbr, ybr = instance_boxes[candidates[idx]]
                            new_mask[ytl:ybr, xtl:xbr] = pred_mask_bin

                            if ((new_mask != 0) * (panoptic_mask != 0)).any():
                                n_total = (new_mask != 0).sum()
                                non_overlaping_mask = (new_mask != 0) * (
                                    panoptic_mask == 0
                                )
                                n_new = non_overlaping_mask.sum().float()
                                if n_new / n_total > self.min_remain:
                                    panoptic_mask[non_overlaping_mask] = n + 1
                                    semantic_mask[:, non_overlaping_mask] = sem[
                                        candidates[idx]
                                    ][:, None]
                            else:
                                panoptic_mask[(new_mask != 0)] = n + 1
                                semantic_mask[:, (new_mask != 0)] = sem[
                                    candidates[idx]
                                ][:, None]
                panoptic_instance.append(panoptic_mask)
                panoptic_semantic.append(semantic_mask)
            panoptic_instance = torch.stack(panoptic_instance, dim=0)
            panoptic_semantic = torch.stack(panoptic_semantic, dim=0)
        else:
            panoptic_instance = None
            panoptic_semantic = None

        predictions = dict(
            center_mask=center_mask,
            saliency=saliency,
            heatmap=heatmap,
            semantic=sem,
            size=size,
            confidence=confidence,
            centerness=centerness,
            instance_masks=instance_masks,
            instance_boxes=instance_boxes,
            pano_instance=panoptic_instance,
            pano_semantic=panoptic_semantic,
        )

        return predictions


class CenterExtractor(nn.Module):

    def __init__(self):
        """
        Module for local maxima extraction
        """
        super(CenterExtractor, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(
            self,
            x: Tensor,
            zones: Optional[Tensor] = None,
        ):
        """
        Args:
            x (tensor): Centerness heatmap
            zones (tensor, Optional): Tensor that defines the mapping between each pixel position and
            the "closest" center during training (see paper paragraph Centerpoint detection).
            If provided, the highest local maxima in each zone is kept. As a result at most one
            prediction is made per ground truth object.
            If not provided, all local maxima are returned.
        """
        if self.training:
            assert zones is not None # auxiliary supervisory signal
            x = x.flatten(1)
            hw_shape = zones.shape[-2:]
            num_elem = x.shape[1]
            _, argmax = scatter_max(x, zones.flatten(1).long(), dim=1)
            masks = []
            for idx in argmax:
                mask = torch.zeros(num_elem, device=x.device)
                mask[idx[idx != num_elem]] = 1 # filter out non-present group index
                masks.append(mask.view(hw_shape))
            centermask = torch.stack(masks, dim=0).bool() # [bsz, H, W]
        else:
            centermask = x == self.pool(x)
            no_valley = x > x.mean()
            centermask = (centermask * no_valley).squeeze(dim=1)

        n_centers = int(centermask.sum().detach().cpu())
        while n_centers == 0:
            centermask = torch.empty_like(centermask).random_(2).bool()
            n_centers = int(centermask.sum().detach().cpu())

        return centermask, n_centers


class ConvLayer(nn.Module):
    def __init__(
        self,
        nkernels,
        norm="batch",
        k=3,
        s=1,
        p=1,
        n_groups=4,
        last_relu=True,
        padding_mode="reflect",
    ):
        super(ConvLayer, self).__init__()
        layers = []
        if norm == "batch":
            nl = nn.BatchNorm2d
        elif norm == "instance":
            nl = nn.InstanceNorm2d
        elif norm == "group":
            nl = lambda num_feats: nn.GroupNorm(
                num_channels=num_feats,
                num_groups=n_groups,
            )
        else:
            nl = None
        for i in range(len(nkernels) - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=nkernels[i],
                    out_channels=nkernels[i + 1],
                    kernel_size=k,
                    padding=p,
                    stride=s,
                    padding_mode=padding_mode,
                )
            )
            if nl is not None:
                layers.append(nl(nkernels[i + 1]))

            if last_relu:
                layers.append(nn.ReLU())
            elif i < len(nkernels) - 2:
                layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, input):
        return self.conv(input)
