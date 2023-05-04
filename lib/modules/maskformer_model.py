"""
Mostly copy-paste from Mask2Former
https://github.com/facebookresearch/Mask2Former
Modified by X.Cai
"""
from typing import List, Tuple, Dict, Optional

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from timm.models.layers import trunc_normal_

from .swin import SwinTransformer
from .pvt import PVT2
from .fpn import FPN
from .maskformer_decoder import MultiScaleMaskedTransformerDecoder
from losses import MaskFormerLoss


class MaskFormer(nn.Module):

    def __init__(self, config, **kwargs):
        super(MaskFormer, self).__init__()

        spec_dict = {k.lower(): v for k, v in dict(config.MASKFORMER).items()}
        spec_dict.update(**kwargs) # overwrite
        backbone_type = spec_dict['backbone_type']
        pixel_decoder_type = spec_dict['pixel_decoder_type']
        # MaskFormer Decoder
        hidden_dim = spec_dict['hidden_dim']
        mask_dim = spec_dict['mask_dim']
        dec_layers = spec_dict['dec_layers']
        num_heads = spec_dict['num_heads']
        dim_ffd = spec_dict['dim_ffd']
        num_queries = spec_dict['num_queries']
        num_feature_levels = spec_dict['num_feature_levels']
        pre_norm = spec_dict['pre_norm']
        enforce_input_project = spec_dict['enforce_input_project']
        object_mask_threshold = spec_dict['object_mask_threshold']
        overlap_threshold = spec_dict['overlap_threshold']
        thing_ids = config.DATASET.THING_IDS # now only valid for PASTIS
        task_type = config.DATASET.TASK_TYPE
        # MaskFormer Loss
        mask_weight = spec_dict['mask_weight']
        dice_weight = spec_dict['dice_weight']
        cls_weight = spec_dict['cls_weight']
        num_points = spec_dict['num_points']
        oversample_ratio = spec_dict['oversample_ratio']
        importance_sample_ratio = spec_dict['importance_sample_ratio']

        in_channels = kwargs['in_channels']
        num_classes = config.DATASET.NUM_CLASSES
        ignore_index = config.LOSS.IGNORE_INDEX

        if backbone_type == 'swin':
            self.backbone = SwinTransformer(
                config,
                in_channels=in_channels,
            )
        elif backbone_type == 'pvt':
            self.backbone = PVT2(
                config,
                in_channels=in_channels,
            )
        else:
            raise NotImplementedError

        if pixel_decoder_type == 'fpn':
            self.pixel_decoder = FPN(
                feature_channels=self.backbone.num_features,
                conv_dim=hidden_dim,
                mask_dim=mask_dim,
                norm='GN',
                maskformer_num_feature_levels=num_feature_levels,
            )
        else:
            raise NotImplementedError

        self.maskformer_decoder = MultiScaleMaskedTransformerDecoder(
            [hidden_dim] * num_feature_levels,
            hidden_dim,
            num_queries,
            num_heads,
            dim_ffd,
            dec_layers,
            mask_dim,
            num_classes,
            task_type=task_type,
            thing_ids=thing_ids,
            object_mask_threshold=object_mask_threshold,
            overlap_threshold=overlap_threshold,
            num_feature_levels=num_feature_levels,
            pre_norm=pre_norm,
            enforce_input_project=enforce_input_project,
        )

        self.loss_decode = MaskFormerLoss(
            num_classes,
            dec_layers,
            task_type=task_type,
            mask_weight=mask_weight,
            dice_weight=dice_weight,
            cls_weight=cls_weight,
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            ignore_index=ignore_index,
        )

    def losses(self, outputs, targets, **kwargs):
        losses = self.loss_decode(outputs, targets)
        return losses

    def forward(self, x, tgt_size) -> Dict[str, Tensor]:

        feat_maps: List[Tensor] = self.backbone(x)
        mask_features, multi_scale_features = self.pixel_decoder(feat_maps)
        outputs = self.maskformer_decoder(multi_scale_features, mask_features, tgt_size)

        return outputs
