from typing import List, Tuple, Dict, Optional

import math
from collections import OrderedDict

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from timm.models.layers import trunc_normal_

from layers import TemporalPositionalEncoding, Projector, ClfLayer
from modules import Exchanger, TempSpatTransformer


class Classifier(nn.Module):
    def __init__(self, config, **kwargs):
        super(Classifier, self).__init__()

        self.mode = kwargs['mode']
        spec_dict = {k.lower(): v for k, v in dict(config.CLS_HEAD).items()}
        modality = config.DATASET.MODALITY[0]
        temp_encoder_type = spec_dict['temp_encoder_type']
        pos_encode_type = spec_dict['pos_encode_type']
        with_gdd_pos = spec_dict['with_gdd_pos']
        pe_dim = spec_dict['pe_dim']
        pe_t = spec_dict['pe_t']
        max_temp_len = spec_dict['max_temp_len']
        proj_hid_nlayers = spec_dict['proj_hid_nlayers']
        proj_hid_dim = spec_dict['proj_hid_dim']
        proj_bot_dim = spec_dict['proj_bot_dim']
        proj_norm_type = spec_dict['proj_norm_type']
        proj_act_type = spec_dict['proj_act_type']
        tau = spec_dict['tau']
        ignore_index = config.LOSS.IGNORE_INDEX
        in_dim = config.DATASET.INPUT_DIM[0]
        num_classes = config.DATASET.NUM_CLASSES
        kth_cluster = kwargs.get('kth_cluster', None)

        self.temp_pos_encode = TemporalPositionalEncoding(
            pos_encode_type,
            pe_dim,
            T=pe_t,
            with_gdd_pos=with_gdd_pos,
            max_len=max_temp_len,
        )

        if temp_encoder_type == 'exchanger':
            self.temp_encoder = Exchanger(
                config,
                **{'in_channels': in_dim,
                   'pe_dim': pe_dim,
                   }
            )
        elif temp_encoder_type == 'self_attn':
            self.temp_encoder = TempSpatTransformer(config, **{'in_channels': in_dim})
        else:
            raise NotImplementedError

        temp_out_dim = self.temp_encoder.out_dim

        projector = Projector(
            feat_dim=temp_out_dim,
            input_l2_norm=True,
            hidden_layers=proj_hid_nlayers,
            hidden_dim=proj_hid_dim,
            bottleneck_dim=proj_bot_dim,
            norm_type=proj_norm_type,
            act_type=proj_act_type,
        )
        cls_layer = ClfLayer(proj_bot_dim, num_classes, tau=tau)
        self.cls_head = nn.Sequential(*[projector, cls_layer])

        self.cls_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction='mean',
            label_smoothing=config.LOSS.SMOOTH_FACTOR
        )

        self.modality = modality
        self.temp_encoder_type = temp_encoder_type
        self.kth_cluster = kth_cluster

    def parse_inputs(self, inputs: Dict[str, Tensor], modality: str):

        data = inputs[f'data_{modality}'][:, :-1, ...].float()
        img_mask = inputs[f'data_{modality}'][:, -1, ...].bool() # [B, T, N]
        date_pos = inputs[f'date_positions_{modality}'][:, 0, ...].long()
        temporal_mask = inputs[f'date_positions_{modality}'][:, 1, ...].bool()
        gdd = inputs[f'gdd_{modality}'][:, 0, ...].long()
        labels = inputs['label'].long()

        return data, img_mask, date_pos, gdd, temporal_mask, labels

    def losses(self, outputs, targets, **kwargs):
        assert 'pred_logits' in outputs
        predictions = outputs['pred_logits']
        loss = self.cls_loss(predictions, targets)
        losses = {'loss_ce': loss}

        return losses

    def forward(
            self,
            inputs: Dict[str, Tensor],
            eps: float = 1e-6,
            return_attn: bool = False,
            **kwargs
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:

        outputs = {}

        data, img_mask, date_pos, gdd, temporal_mask, labels = self.parse_inputs(inputs, self.modality)

        B, _, T, N = data.shape

        x = data.permute(0, 3, 2, 1).contiguous().view(B * N, T, -1)
        img_mask = img_mask.transpose(1, 2).contiguous().view(B * N, T)

        temp_pos = self.temp_pos_encode(
            date_pos,
            gdd=gdd,
            key_padding_mask=temporal_mask,
        )
        temp_pos = temp_pos.repeat_interleave(N, dim=0).transpose(0, 1)

        if self.temp_encoder_type == 'exchanger':
            x, attn, latent_feats = self.temp_encoder(
                x, temp_pos, img_mask,
                return_attn=return_attn,
                kth_cluster=self.kth_cluster,
            )
            x = masked_mean(x, img_mask)
            x = x.view(B, N, -1).mean(dim=1)
            if latent_feats is not None:
                num_stages = latent_feats.shape[1]
                latent_feats = masked_mean(
                    latent_feats, # [B*N, num_stages, T, C]
                    img_mask.unsqueeze(dim=1).\
                    repeat(1, num_stages, 1)
                ).view(B, N, num_stages, -1).mean(dim=1)
        elif self.temp_encoder_type == 'self_attn':
            x = self.temp_encoder(x, temp_pos, N, img_mask)
        else:
            raise NotImplementedError

        logits = self.cls_head(x)

        outputs['pred_logits'] = logits
        outputs['labels'] = labels

        if return_attn:
            outputs['attn'] = attn
            outputs['latent_feats'] = latent_feats

        if self.mode != 'test':
            losses = self.losses(outputs, labels)
        else:
            losses = {}

        return outputs, losses


def masked_mean(x, mask, eps=1e-6):
    """
    x shape: [B, T, C]
    mask shape: [B, T]
    """
    rmask = ~(mask.bool()) # for non-padded elements
    n_elements = rmask.sum(dim=-1, keepdim=True).float()
    masked_mean = (x * rmask.unsqueeze(dim=-1)).sum(dim=-2) / (n_elements + eps)
    return masked_mean
