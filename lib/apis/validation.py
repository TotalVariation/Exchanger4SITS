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
from metrics import Precision_Recall_Fscore, IoU, PanopticMeter


def validate(config, testloader, model, writer_dict, device='cpu'):

    model.eval()
    metric_logger = dist_helper.MetricLogger(delimiter="  ")
    header = 'Validation:'

    task_type = config.DATASET.TASK_TYPE
    num_classes = config.DATASET.NUM_CLASSES
    ignore_index = config.LOSS.IGNORE_INDEX

    if task_type == 'cls':
        eval_metric = Precision_Recall_Fscore(num_classes, cm_device='cpu', ignore_index=ignore_index)
    elif task_type == 'sem_seg':
        eval_metric = IoU(num_classes, cm_device='cpu', ignore_index=ignore_index)
    elif task_type in ('paps_pano', 'maskformer_pano'):
        eval_metric = PanopticMeter(
            num_classes,
            background_label=0, # FIXME ignore stuff classes
            void_label=ignore_index,
            device=device,
            target_type=task_type
        )
    else:
        raise NotImplementedError

    with torch.no_grad():
        for batch in metric_logger.log_every(testloader, config.PRINT_FREQ, header):

            batch = recursive2device(batch, device)
            outputs, losses = model(batch)

            if task_type == 'cls':
                preds = outputs['pred_logits'] # [b, c]
                preds = preds.softmax(dim=1).argmax(dim=1)
            elif task_type == 'sem_seg':
                preds = outputs['preds'] # [b, c, h, w]
                if preds.ndim == 4:
                    if preds.shape[-2:] != outputs['labels'].shape[-2:]:
                        preds = F.interpolate(
                            preds, size=outputs['labels'].shape[-2:], mode='bilinear', align_corners=False
                        )
                    preds = preds.softmax(dim=1).argmax(dim=1)
            elif task_type in ('paps_pano', 'maskformer_pano'):
                preds = outputs
            else:
                raise NotImplementedError

            metric_logger.update(**losses)
            tot_loss = 0.
            for l in losses.values():
                tot_loss += l
            metric_logger.update(TotalLoss=tot_loss)

            eval_metric.add(preds, outputs['labels'])

        metric_logger.synchronize_between_processes()
        dist_helper.log(metric_logger)
        scores = eval_metric.value()

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', metric_logger.TotalLoss.global_avg, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return metric_logger.TotalLoss.global_avg, scores
