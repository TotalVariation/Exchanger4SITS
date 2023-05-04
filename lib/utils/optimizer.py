# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# ------------------------------------------------------------------------------


import torch
import torch.nn as nn

from .dist_helper import log


def get_optimizer(config, model, lr_scaler):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = config.LR.LR * lr_scaler
        if '_nwd' in key or key.endswith('.bias') or len(value.shape) == 1:
            weight_decay = 0.
        else:
            weight_decay = config.OPTIMIZER.WD

        log(f'Params: {key}, LR: {lr}, Weight_Decay: {weight_decay}')
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if config.OPTIMIZER.TYPE == 'sgd':
        optimizer = torch.optim.SGD(params,
                                    lr=config.LR.LR * lr_scaler,
                                    momentum=config.OPTIMIZER.MOMENTUM,
                                    weight_decay=config.OPTIMIZER.WD,
                                    )
    elif config.OPTIMIZER.TYPE == 'adam':
        optimizer = torch.optim.Adam(params,
                                     lr=config.LR.LR * lr_scaler,
                                     )
    elif config.OPTIMIZER.TYPE == 'adamw':
        optimizer = torch.optim.AdamW(params,
                                      lr=config.LR.LR * lr_scaler,
                                      weight_decay=config.OPTIMIZER.WD,
                                      )
    else:
        raise NotImplementedError

    return optimizer

