from typing import List, Tuple, Dict, Optional

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from layers import LinearLayer
from modules import GPBlock


class Exchanger(nn.Module):
    def __init__(self, config, **kwargs):
        super(Exchanger, self).__init__()

        spec_dict = {k.lower(): v for k, v in dict(config.EXCHANGER).items()}
        embed_dims: List[int] = spec_dict['embed_dims']
        num_token_list = spec_dict['num_token_list']
        num_heads_list = spec_dict['num_heads_list']
        drop_path_rate = spec_dict['drop_path_rate']
        mlp_norm = spec_dict['mlp_norm']
        mlp_act = spec_dict['mlp_act']
        in_dim = kwargs['in_channels']
        pe_dim = kwargs['pe_dim']

        num_stages = len(num_token_list)
        assert len(embed_dims) == len(num_heads_list) == num_stages
        dpr = np.linspace(0, drop_path_rate, num_stages)

        self.mlp_layers = nn.ModuleList()
        self.temp_encoder_blocks = nn.ModuleList()

        for i in range(num_stages):

            self.mlp_layers.append(
                LinearLayer(
                    in_dim=in_dim if i == 0 else embed_dims[i-1],
                    out_dim=embed_dims[i],
                    norm_type=mlp_norm,
                    act_type=mlp_act,
                    bias=False
                )
            )

            self.temp_encoder_blocks.append(
                GPBlock(
                    config,
                    **{
                        'embed_dims': embed_dims[i],
                        'num_group_tokens': num_token_list[i],
                        'num_heads': num_heads_list[i],
                        'drop_path': dpr[i],
                        'pe_dim': pe_dim,
                     }
                )
            )

        self.num_stages = num_stages
        self.out_dim = embed_dims[-1]

    def forward(
            self,
            x: Tensor,
            temp_pos: Tensor,
            temp_mask: Optional[Tensor] = None,
            return_attn: bool = False,
            kth_cluster: Optional[int] = None,
            **kwargs
    ) -> Tensor:
        """
        Args Shape:
            x: [B, T, C]
            temp_pos: [T, B, C]
            temp_mask: [B, T]
        """
        attn_list = []
        feat_list = []

        for i in range(self.num_stages):
            x = self.mlp_layers[i](x)
            x = x.transpose(0, 1)
            x, attn = self.temp_encoder_blocks[i](x, temp_pos, temp_mask, kth_cluster=kth_cluster)
            if return_attn:
                attn_list.append(attn)
            x = x.transpose(0, 1)
            if return_attn:
                feat_list.append(x)

        if return_attn:
            return x, torch.stack(attn_list, dim=1), torch.stack(feat_list, dim=1)
        else:
            return x, None, None
