# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import logging
import os
import time
from tqdm import tqdm
import itertools

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import dist_helper, recursive2device, remove_spectral_norm, remove_weight_norm
from metrics import Precision_Recall_Fscore, IoU, PanopticMeter


def testeval(config, testloader, model, saved_path, device='cpu'):
    model.eval()

    model.apply(remove_spectral_norm)
    model.apply(remove_weight_norm)

    metric_logger = dist_helper.MetricLogger(delimiter="  ")
    header = 'Test:'

    task_type = config.DATASET.TASK_TYPE
    num_classes = config.DATASET.NUM_CLASSES
    ignore_index = config.LOSS.IGNORE_INDEX
    return_attn = config.TEST.RETURN_ATTN
    if return_attn:
        assert dist_helper.get_world_size() == 1, f'only supports non-parallel evaluation.'
        assert saved_path is not None, f'requires specifiying save path'
        attn_list = []
        parcel_id_list = []
        y_true_list = []
        y_pred_list = []
        latent_feats = []

    if task_type == 'cls':
        eval_metric = Precision_Recall_Fscore(num_classes, cm_device='cpu', ignore_index=ignore_index)
    elif task_type == 'sem_seg':
        eval_metric = IoU(num_classes, cm_device='cpu', ignore_index=ignore_index)
    elif task_type in ('paps_pano', 'maskformer_pano'):
        eval_metric = PanopticMeter(
            num_classes,
            background_label=0,
            void_label=ignore_index,
            device=device,
            target_type=task_type
        )
    else:
        raise NotImplementedError

    with torch.no_grad():

        for batch in metric_logger.log_every(testloader, 1, header):

            batch = recursive2device(batch, device)

            if task_type == 'cls':
                outputs = model(batch, return_attn=return_attn)[0]
                preds = outputs['pred_logits'] # [b, c]
                preds = preds.softmax(dim=1).argmax(dim=1)
            elif task_type == 'sem_seg':
                if config.TEST.MULTI_CROP_TEST:
                    # workaround for CUDA OOM crop & stitch not test augmentation
                    crop_size = config.TEST.CROP_SIZE
                    if dist_helper.get_world_size() > 1:
                        outputs = model.module.multi_crop_inference(batch, crop_size)
                    else:
                        outputs = model.multi_crop_inference(batch, crop_size)
                else:
                    outputs = model(batch)[0]
                preds = outputs['preds'] # [b, c, h, w] or [b, h, w]
                if preds.ndim == 4:
                    if preds.shape[-2:] != outputs['labels'].shape[-2:]:
                        preds = F.interpolate(
                            preds, size=outputs['labels'].shape[-2:],
                            mode='bilinear', align_corners=False
                        )
                    preds = preds.softmax(dim=1).argmax(dim=1)
            elif task_type in ('paps_pano', 'maskformer_pano'):
                outputs = model(batch)[0]
                preds = outputs
            else:
                raise NotImplementedError

            eval_metric.add(preds, outputs['labels'])
            if return_attn:
                y_true_list.append(outputs['labels'])
                y_pred_list.append(preds)
                attn_list.append(outputs['attn'].unbind(dim=0)) # [bsz, num_stages, seq_len, num_clusters] for variable length input
                latent_feats.append(outputs['latent_feats'].unbind(dim=0))
                parcel_id_list.append(batch['field_id'])

        metric_logger.synchronize_between_processes()
        dist_helper.log(metric_logger)
        #scores = eval_metric.value()

    if return_attn:
        y_true = torch.cat(y_true_list, dim=0).cpu().numpy()
        y_pred = torch.cat(y_pred_list, dim=0).cpu().numpy()
        parcel_ids = torch.cat(parcel_id_list, dim=0).cpu().numpy()
        attn_list = [attn.cpu().numpy() for attn in itertools.chain.from_iterable(attn_list)]
        feat_list = [feat.cpu().numpy() for feat in itertools.chain.from_iterable(latent_feats)]

        output_list = []
        for y_t, y_p, pid, attn, feat in zip(y_true, y_pred, parcel_ids, attn_list, feat_list):
            output_list.append(
                {
                    'y_true': int(y_t),
                    'y_pred': int(y_p),
                    'parcel_id': int(pid),
                    'attn_weights': attn.tolist(),
                    'latent_feats': feat.tolist(),
                }
            )
        output_df = pd.DataFrame.from_dict(output_list)
        output_df.to_json(saved_path)
        logging.info(f"Test results have been saved to location: {saved_path}")

    return eval_metric

