from typing import List, Tuple, Dict, Optional

import math
from collections import OrderedDict

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from timm.models.layers import trunc_normal_

from modules import Exchanger, UNet, PaPs, MaskFormer
from layers import TemporalPositionalEncoding
from .cls import masked_mean


class PanopticSegmentor(nn.Module):
    def __init__(self, config, **kwargs):
        super(PanopticSegmentor, self).__init__()

        self.mode = kwargs['mode']
        spec_dict = {k.lower(): v for k, v in dict(config.PANOPTIC_SEGMENTOR).items()}
        modality = config.DATASET.MODALITY[0]
        pos_encode_type = spec_dict['pos_encode_type']
        with_gdd_pos = spec_dict['with_gdd_pos']
        pe_dim = spec_dict['pe_dim']
        pe_t = spec_dict['pe_t']
        max_temp_len = spec_dict['max_temp_len']
        space_encoder_type = spec_dict['space_encoder_type']
        in_dim = config.DATASET.INPUT_DIM[0]
        num_classes = config.DATASET.NUM_CLASSES
        ignore_index = config.LOSS.IGNORE_INDEX
        loss_type = config.LOSS.TYPE

        self.temp_pos_encode = TemporalPositionalEncoding(
            pos_encode_type,
            pe_dim,
            T=pe_t,
            with_gdd_pos=with_gdd_pos,
            max_len=max_temp_len,
        )

        self.temp_encoder = Exchanger(config, **{'in_channels': in_dim, 'pe_dim': pe_dim})
        temp_out_dim = self.temp_encoder.out_dim

        if space_encoder_type == 'unet':
            self.space_encoder = UNet(config, **{'in_channels': temp_out_dim})
            space_enc_out_dims = self.space_encoder.out_dims
            self.paps_head = PaPs(
                config,
                **{
                    'enc_dim': space_enc_out_dims[0],
                    'stack_dim': sum(space_enc_out_dims),
                }
            )
        elif space_encoder_type == 'maskformer':
            self.space_encoder = MaskFormer(
                config,
                in_channels=temp_out_dim,
            )
        else:
            raise NotImplementedError

        self.modality = modality
        self.space_encoder_type = space_encoder_type
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    def parse_inputs(self, inputs: Dict[str, Tensor], modality: str):

        data = inputs[f'data_{modality}'][:, :-1, ...].float()
        img_mask = inputs[f'data_{modality}'][:, -1, ...].bool() # [B, T, H, W]
        date_pos = inputs[f'date_positions_{modality}'][:, 0, ...].long()
        temporal_mask = inputs[f'date_positions_{modality}'][:, 1, ...].bool()
        #gdd = inputs[f'gdd_{modality}'][:, 0, ...].long()
        labels = inputs['label'][:, :-1, ...].float() # [B, K, H, W]

        return data, img_mask, date_pos, temporal_mask, labels

    def losses(self, outputs: Dict[str, Tensor], targets: Tensor, **kwargs):
        if self.space_encoder_type == 'maskformer':
            return self.space_encoder.losses(outputs, targets)
        elif hasattr(self, 'paps_head'):
            return self.paps_head.losses(outputs, targets)
        else:
            raise NotImplementedError

    def forward(
            self,
            inputs: Dict[str, Tensor],
            output_aux: bool = False,
            **kwargs
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:

        outputs = {}

        data, img_mask, date_pos, temporal_mask, labels = self.parse_inputs(inputs, self.modality)

        B, _, T, H, W = data.shape
        num_patches = H * W # patch size 1
        x = data.reshape(B, -1, T, num_patches).contiguous().permute(0, 3, 2, 1).contiguous()\
            .view(B*num_patches, T, -1)

        img_mask = img_mask.view(B, T, num_patches).transpose(1, 2).contiguous().view(B*num_patches, T)

        temp_pos = self.temp_pos_encode(
            date_pos,
            key_padding_mask=temporal_mask
        )
        temp_pos = temp_pos.repeat_interleave(num_patches, dim=0).transpose(0, 1)

        x = self.temp_encoder(x, temp_pos, img_mask)[0]
        x = masked_mean(x, img_mask).contiguous().view(B, num_patches, -1).transpose(1, 2)\
            .contiguous().view(B, -1, H, W)

        if self.space_encoder_type == 'maskformer':
            out: Dict[str, Tensor] = self.space_encoder(x, (H, W))
            outputs.update(**out)
        else:
            space_outs: List[Tensor] = self.space_encoder(x)
            if self.space_encoder_type == 'unet':
                high_res_level = space_outs[-1]
                feat_maps = space_outs[::-1]
            else:
                raise NotImplementedError

            zones = labels[:, 2, ...] if self.training else None
            outputs = self.paps_head(high_res_level, feat_maps, zones=zones)

        outputs['labels'] = labels

        if self.mode != 'test':
            losses = self.losses(outputs, labels)
        else:
            losses = {}

        return outputs, losses
