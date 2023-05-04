from typing import List, Tuple, Dict, Optional

import math
from collections import OrderedDict

import torch
from torch import nn, Tensor

from modules import PixelSetEncoder, LTAE
from layers import LinearLayer, TemporalPositionalEncoding, Projector, ClfLayer


class PSELTAE(nn.Module):
    def __init__(self, config, **kwargs):
        super(PSELTAE, self).__init__()

        self.mode = kwargs['mode']
        spec_dict = {k.lower(): v for k, v in dict(config.PSELTAE).items()}
        proj_hid_nlayers = spec_dict['proj_hid_nlayers']
        proj_hid_dim = spec_dict['proj_hid_dim']
        proj_bot_dim = spec_dict['proj_bot_dim']
        proj_norm_type = spec_dict['proj_norm_type']
        proj_act_type = spec_dict['proj_act_type']
        tau = spec_dict['tau']
        ignore_index = config.LOSS.IGNORE_INDEX
        self.modality = config.DATASET.MODALITY[0]
        in_dim = config.DATASET.INPUT_DIM[0]
        num_classes = config.DATASET.NUM_CLASSES

        spatial_encoder = PixelSetEncoder(config, **{'input_dim': in_dim})
        self.add_module(f'spatial_encoder_{self.modality}', spatial_encoder)

        kwargs['in_channels'] = spatial_encoder.last_dim
        self.temp_decoder = LTAE(config, **kwargs) # LTAE is a light-weight transformer decoder

        last_dim = self.temp_decoder.last_dim
        projector = Projector(
            feat_dim=last_dim,
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

    def losses(self, outputs, targets, **kwargs):
        assert 'pred_logits' in outputs
        predictions = outputs['pred_logits']
        loss = self.cls_loss(predictions, targets)
        losses = {'loss_ce': loss}

        return losses

    def parse_inputs(self, inputs: Dict[str, Tensor], modality: str):

        data = inputs[f'data_{modality}'][:, :-1, ...].float()
        img_mask = inputs[f'data_{modality}'][:, -1, ...].bool() # [B, T, N]
        date_pos = inputs[f'date_positions_{modality}'][:, 0, ...].long()
        temporal_mask = inputs[f'date_positions_{modality}'][:, 1, ...].bool()
        gdd = inputs[f'gdd_{modality}'][:, 0, ...].long()
        labels = inputs['label'].long()

        return data, img_mask, date_pos, gdd, temporal_mask, labels

    def forward(
            self,
            inputs: Dict[str, Tensor],
            return_attn: bool = False,
            eps: float = 1e-6,
            **kwargs
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:

        outputs = {}

        data, img_mask, date_pos, gdd, temporal_mask, labels = self.parse_inputs(inputs, self.modality)

        spatial_encoder = self.__getattr__(f'spatial_encoder_{self.modality}')
        out = spatial_encoder(data, img_mask, temporal_mask)

        out, attn = self.temp_decoder(
            out,
            temp_idx=date_pos,
            gdd=gdd,
            key_padding_mask=temporal_mask,
        )

        logits = self.cls_head(out)

        outputs['pred_logits'] = logits
        outputs['labels'] = labels

        if return_attn:
            outputs['attn'] = attn.transpose(-2, -1)

        if self.mode != 'test':
            losses = self.losses(outputs, labels)
        else:
            losses = {}

        return outputs, losses

