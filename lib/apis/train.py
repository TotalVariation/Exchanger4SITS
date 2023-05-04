# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import logging
import os
import time
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import dist_helper, recursive2device


def train(config, epoch, max_iters,
          trainloader, optimizer, lr_scheduler,
          model, writer_dict, device='cpu'):
    # Training
    model.train()

    metric_logger = dist_helper.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr',
        dist_helper.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    header = f'Epoch: [{epoch}]'

    for batch in metric_logger.log_every(trainloader, config.PRINT_FREQ, header):
        optimizer.zero_grad()

        batch = recursive2device(batch, device)
        outputs, losses = model(batch)

        metric_logger.update(**losses)

        tot_loss = 0.
        for l in losses.values():
            tot_loss += l

        tot_loss.backward()

        optimizer.step()
        lr_scheduler.step()

        metric_logger.update(TotalLoss=tot_loss)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    dist_helper.log(metric_logger)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train_loss', metric_logger.TotalLoss.global_avg, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1

