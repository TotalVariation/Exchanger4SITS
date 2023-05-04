from typing import List, Tuple, Dict, Optional

import math
from collections import OrderedDict

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from timm.models.layers import trunc_normal_

from modules import Exchanger, UNet, MaskFormer
from layers import TemporalPositionalEncoding
from datasets.transforms import pad_if_smaller
from losses import FocalCELoss
from .cls import masked_mean


class Segmentor(nn.Module):
    def __init__(self, config, **kwargs):
        super(Segmentor, self).__init__()

        self.mode = kwargs['mode']
        spec_dict = {k.lower(): v for k, v in dict(config.SEGMENTOR).items()}
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
            last_dim = self.space_encoder.out_dims[0]
            self.cls_head = nn.Conv2d(last_dim, num_classes, 1)
            if loss_type == 'crossentropy':
                self.ce_loss = nn.CrossEntropyLoss(
                    ignore_index=ignore_index,
                    reduction='mean',
                    label_smoothing=config.LOSS.SMOOTH_FACTOR
                )
            elif loss_type == 'focal_ce':
                gamma = config.LOSS.FOCAL[1]
                self.ce_loss = FocalCELoss(gamma=gamma, size_average=True, ignore_index=ignore_index)
                r = 0.01
                nn.init.constant_(self.cls_head.bias, -1.0 * math.log((1 - r) / r))
            else:
                raise NotImplementedError
        elif space_encoder_type == 'maskformer':
            # use DETR-like set prediction objective
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
        labels = inputs['label'][:, 0, ...].long() # [B, H, W]
        spatial_pad_masks = inputs['label'][:, -1, ...].bool()
        labels = labels.masked_fill(spatial_pad_masks, self.ignore_index)

        return data, img_mask, date_pos, temporal_mask, labels

    def losses(self, outputs: Dict[str, Tensor], targets: Tensor, **kwargs):
        if self.space_encoder_type == 'maskformer':
            losses = self.space_encoder.losses(outputs, targets)
        else:
            preds = outputs['preds']
            if preds.shape[-2:] != targets.shape[-2:]:
                preds = F.interpolate(preds, size=targets.shape[-2:], mode='bilinear', align_corners=False)
            ce_loss = self.ce_loss(preds, targets)
            losses = {'loss_ce': ce_loss}

        return losses

    def forward(
            self,
            inputs: Dict[str, Tensor],
            **kwargs
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:

        outputs = {}

        data, img_mask, date_pos, temporal_mask, labels = self.parse_inputs(inputs, self.modality)

        B, _, T, H, W = data.shape
        num_patches = H * W # patch size 1
        x = data.reshape(B, -1, T, num_patches).contiguous().permute(0, 3, 2, 1)\
            .contiguous().view(B * num_patches, T, -1)

        img_mask = img_mask.view(B, T, num_patches).transpose(1, 2).contiguous().view(B * num_patches, T)

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
            feats: List[Tensor] = self.space_encoder(x)
            outputs['preds'] = self.cls_head(feats[-1] if self.space_encoder_type == 'unet' else feats)

        outputs['labels'] = labels

        if self.mode != 'test':
            losses = self.losses(outputs, labels)
        else:
            losses = {}

        return outputs, losses

    def multi_crop_inference(self, inputs: Dict[str, Tensor], crop_size):
        # split orig input into multiple patches and stitch the preds
        # workaround for CUDA OOM
        # not work for MaskFormer
        x = inputs.pop(f'data_{self.modality}')
        B = x.shape[0]
        orig_h, orig_w = x.shape[-2:]
        assert B == 1, f'only supports batchsize 1'
        assert crop_size[0] < orig_h and crop_size[1] < orig_w, \
            f'crop size should be smaller than input size {orig_h, orig_w}, but got {crop_size}'

        stride_h = int(crop_size[0] * 2. / 3.)
        stride_w = int(crop_size[1] * 2. / 3.)
        nrows = int(np.ceil((orig_h - crop_size[0]) / stride_h)) + 1
        ncols = int(np.ceil((orig_w - crop_size[1]) / stride_w)) + 1
        final_pred = torch.zeros((B, self.num_classes, orig_h, orig_w)).to(x)
        count = torch.zeros((B, 1, orig_h, orig_w)).to(x)

        for i in range(nrows):
            for j in range(ncols):
                h0 = i * stride_h
                w0 = j * stride_w
                h1 = min(h0 + crop_size[0], orig_h)
                w1 = min(w0 + crop_size[1], orig_w)
                patch = x[..., h0:h1, w0:w1]
                if patch.shape[-2] < crop_size[0] or patch.shape[-1] < crop_size[1]:
                    img = patch[:, :-1, ...]
                    mask = patch[:, -1, ...].unsqueeze(dim=1)
                    img = pad_if_smaller(img, crop_size, fill=0.)
                    mask = pad_if_smaller(mask, crop_size, fill=1)
                    patch = torch.cat([img, mask], dim=1)
                inputs[f'data_{self.modality}'] = patch
                outputs, _ = self.forward(inputs)
                preds = outputs['preds']
                if preds.shape[-2:] != crop_size:
                    preds = F.interpolate(preds, size=crop_size, mode='bilinear', align_corners=False)
                final_pred[..., h0:h1, w0:w1] += preds[..., :h1-h0, :w1-w0]
                count[..., h0:h1, w0:w1] += 1

        final_pred = final_pred / count
        labels = outputs['labels']
        return {'preds': final_pred, 'labels': labels}

