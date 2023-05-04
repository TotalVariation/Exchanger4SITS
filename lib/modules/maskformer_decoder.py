"""
Mostly Copy-Paste from https://github.com/facebookresearch/Mask2Former
"""

from typing import List, Dict, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from timm.models.layers import trunc_normal_

from layers import PositionEmbeddingSine, MLP


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MultiScaleMaskedTransformerDecoder(nn.Module):

    def __init__(
            self,
            in_channels,
            hidden_dim,
            num_queries,
            nheads,
            dim_ffd,
            dec_layers,
            mask_dim,
            num_classes,
            task_type='maskformer_pano',
            thing_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                       11, 12, 13, 14, 15, 16, 17, 18], # PASTIS
            object_mask_threshold=0.8,
            overlap_threshold=0.8,
            num_feature_levels=3,
            pre_norm=False,
            enforce_input_project=False,
            **kwargs
    ):
        super().__init__()

        # positional encoding
        self.pe_layer = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.num_feature_levels = num_feature_levels
        self.task_type = task_type
        self.thing_ids = thing_ids
        self.object_mask_threshold = object_mask_threshold
        self.overlap_threshold = overlap_threshold

        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_ffd,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat_nwd = nn.Parameter(torch.empty(num_queries, hidden_dim))
        trunc_normal_(self.query_feat_nwd, std=.02, a=-.02, b=.02)
        # learnable query p.e.
        self.query_pos_nwd = nn.Parameter(torch.empty(num_queries, hidden_dim))
        trunc_normal_(self.query_pos_nwd, std=.02, a=-.02, b=.02)

        # level embedding (we always use 3 scales)
        # self.num_feature_levels = 3
        self.level_embed_nwd = nn.Parameter(torch.empty(self.num_feature_levels, hidden_dim))
        trunc_normal_(self.level_embed_nwd, std=.02, a=-.02, b=.02)
        self.input_proj = nn.ModuleList()
        for i in range(self.num_feature_levels):
            if in_channels[i] != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels[i], hidden_dim, kernel_size=1))
                nn.init.xavier_uniform_(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Identity())

        # output FFNs
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        self.task_type = task_type
        self.num_classes = num_classes

    def forward(
            self,
            x: List[Tensor],
            mask_features: Tensor,
            tgt_size: List[int],
            mask: Optional[Tensor] = None,
            **kwargs
    ):
        # x is a list of multi-scale feature from low-res to high-res
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        bs = x[0].shape[0]

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed_nwd[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        # QxNxC
        query_pos = self.query_pos_nwd.unsqueeze(1).expand(-1, bs, -1)
        output = self.query_feat_nwd.unsqueeze(1).expand(-1, bs, -1)

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index],
                query_pos=query_pos
            )

            output = self.transformer_self_attention_layers[i](
                output,
                tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_pos
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output, mask_features,
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        out_dict = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class,
                predictions_mask
            )
        }

        if not self.training:
            if self.task_type == 'sem_seg':
                out = self.semantic_inference(predictions_class[-1], predictions_mask[-1], tgt_size)
            elif self.task_type == 'maskformer_pano':
                out = self.panoptic_inference(predictions_class[-1], predictions_mask[-1], tgt_size)
            else:
                raise NotImplementedError
            out_dict.update(**out)

        return out_dict

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc, bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend
        # while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).expand(-1, self.num_heads, -1, -1)\
                     .flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]

    def semantic_inference(self, mask_cls, mask_pred, tgt_size):
        if mask_pred.shape[-2:] != tgt_size:
            mask_pred = F.interpolate(mask_pred, size=tgt_size, mode='bilinear', align_corners=False)
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1] # remove no object class when inference
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("bqc, bqhw -> bchw", mask_cls, mask_pred)
        out = {'preds': semseg.argmax(dim=1)}
        return out

    def panoptic_inference(self, mask_cls, mask_pred, tgt_size):

        if mask_pred.shape[-2:] != tgt_size:
            mask_pred = F.interpolate(mask_pred, size=tgt_size, mode='bilinear', align_corners=False)

        device = mask_cls.device
        panoptic_instance = []
        panoptic_semantic = []

        for ma_cls_per_img, ma_pred_per_img in zip(mask_cls, mask_pred):
            panoptic_seg = torch.zeros(tgt_size, device=device) # to be filled with instance ids
            semantic_seg = torch.zeros(tgt_size, device=device)

            scores, labels = F.softmax(ma_cls_per_img, dim=-1).max(-1)
            ma_pred_per_img = ma_pred_per_img.sigmoid()

            keep = labels.ne(self.num_classes) & (scores > self.object_mask_threshold)
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = ma_pred_per_img[keep]
            cur_mask_cls = ma_cls_per_img[keep]
            cur_mask_cls = cur_mask_cls[:, :-1]

            cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

            current_segment_id = 0

            if cur_masks.shape[0] != 0:
                # take argmax
                cur_mask_ids = cur_prob_masks.argmax(0)
                stuff_memory_list = {}
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()
                    isthing = pred_class in self.thing_ids
                    mask_area = (cur_mask_ids == k).sum().item()
                    original_area = (cur_masks[k] >= 0.5).sum().item()
                    mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        if mask_area / original_area < self.overlap_threshold:
                            continue

                    semantic_seg[mask] = pred_class

                    # TODO merge stuff regions
                    # stuff classes are currently not considered in PaPs
                    """
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1
                    """
                    if not isthing: # background class 0 in PASTIS
                        assert int(pred_class) == 0, \
                            f'currently stuff class is only background but got {pred_class}'
                        panoptic_seg[mask] = 0
                        continue

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

            panoptic_instance.append(panoptic_seg)
            panoptic_semantic.append(semantic_seg)

        panoptic_instance = torch.stack(panoptic_instance, dim=0) # [bsz, H, W]
        panoptic_semantic = torch.stack(panoptic_semantic, dim=0)
        out = {
            'pano_instance': panoptic_instance,
            'pano_semantic': panoptic_semantic,
        }
        return out
