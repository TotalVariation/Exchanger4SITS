# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path
import random
import collections
import pickle
from functools import partial

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from fvcore.nn import FlopCountAnalysis, flop_count_table

import _init_paths
import datasets
import models
from config import config, update_config
from apis import testeval
from utils import create_logger, pad_collate, pad_collate_split_stack, dist_helper


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Crop Classification Network')

    parser.add_argument('--cfg',
                        help='experiment configuration file name',
                        required=True,
                        type=str
                        )
    parser.add_argument('--model',
                        help='the path to the model to be tested',
                        required=True,
                        type=str
                        )
    parser.add_argument('--save-name', default=None,
                        help='the name to the saved file',
                        type=str
                        )
    parser.add_argument('--val', action='store_true', default=False,
                        help='test on valition dataset',
                        )
    parser.add_argument('--n-fold', type=int, default=None,
                        choices=[1, 2, 3, 4, 5], # 5-Fold cross validation
                        help='N-Fold for cross validation')
    parser.add_argument('--kth-cluster', type=int, default=None,
                        help='only keep kth-cluster for evaluation'
                        )
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed')
    parser.add_argument('--workers', type=int, default=16,
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    # distributed parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    if args.world_size > 1:
        dist_helper.init_distributed_mode(args)
    else:
        args.distributed = False

    device = torch.device(args.device)

    if args.seed is not None:
        seed = args.seed + dist_helper.get_rank()
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        cudnn.deterministic = True

    # cudnn related setting
    cudnn.benchmark = True

    verbose = True if dist_helper.is_main_process() else False

    n_fold = args.n_fold - 1 if args.n_fold is not None else config.DATASET.N_FOLD
    if verbose:
        print("create logger ...")
        logger, final_output_dir, _ = create_logger(
            config,
            args.cfg,
            phase='test',
            n_fold=n_fold,
        )
        logger.info(pprint.pformat(args))
        logger.info(config)

    # prepare data
    dataset_type = config.DATASET.DATASET
    if dataset_type == 'TimeMatch':
        test_list = config.DATASET.TEST_TILES
        test_ds = []
        for i in range(len(test_list)):
            ds = eval('datasets.' + config.DATASET.READER)(
                config,
                dataset_splits=None,
                **test_list[i],
                mode='test')
            test_ds.append(ds)
        test_dataset = torch.utils.data.ConcatDataset(test_ds)
    elif dataset_type in ('PASTIS-R', 'PASTIS-R_PixelSet'):
        fold_sequence = [
            [[1, 2, 3], [4], [5]],
            [[2, 3, 4], [5], [1]],
            [[3, 4, 5], [1], [2]],
            [[4, 5, 1], [2], [3]],
            [[5, 1, 2], [3], [4]],
        ]
        test_fold = fold_sequence[n_fold][-1] if not args.val else fold_sequence[n_fold][-2]
        test_dataset = eval('datasets.' + config.DATASET.READER)(config, nfolds=test_fold, mode='test')
    elif dataset_type in ('T31TFM', 'T31TFM_PixelSet', 'MTLCC'):
        test_dataset = eval('datasets.' + config.DATASET.READER)(config, mode='test')
    else:
        raise NotImplementedError

    # build model
    model = eval('models.'+ config.MODEL.NAME)(config, mode='test', kth_cluster=args.kth_cluster)

    if verbose:
        model.eval()
        logger.info(model)
        tot_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.
        logger.info(f">>> total trainable params: {tot_params:.2f}Mb")
        # provide the summary of model
        #dummy_sample = pad_collate([test_dataset[0]])
        #logger.info(summary(model, inputs=dummy_sample))
        #flops = FlopCountAnalysis(model, dummy_sample) FIXME trigger errors with CenterExtractor in PaPs
        #logger.info(flop_count_table(flops))

    if args.distributed:
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=dist_helper.get_world_size(),
            rank=dist_helper.get_rank(),
            shuffle=False,
            drop_last=False)
    else:
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        drop_last=False,
        sampler=test_sampler,
        collate_fn=partial(
            pad_collate,
            pad_value=config.DATASET.PAD_VALUE
        ) if not config.TEST.MULTI_CROP_TEST else \
        partial(
            pad_collate_split_stack, # workaround for CUDA OOM when using Mask2Former
            split_size=config.TEST.CROP_SIZE[0],
            pad_value=config.DATASET.PAD_VALUE
        ),
        num_workers=args.workers,
        pin_memory=True,
    )

    # load the trained model to all ranks
    assert os.path.isfile(args.model), f'the path of the test model "{args.model}" is not valid!'
    checkpoint = torch.load(args.model, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    if verbose:
        logger.info(f'=> Loading model from {args.model}')
        logger.info(f"=> metrics of the model is {checkpoint['metrics']}")
        for k, _ in state_dict.items():
            logger.info(f'=> Loading {k} from the trained model')

    model = model.to(device)

    start = timeit.default_timer()

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        #model_without_ddp = model.module

    saved_path = os.path.join(final_output_dir, str(args.save_name) + '.json') if verbose else None
    eval_metric = testeval(config, testloader, model, saved_path, device=device)
    scores = eval_metric.value()

    if hasattr(eval_metric, 'conf_metric'):
        conf_matrix = eval_metric.conf_metric.value()
        conf_matrix = dist_helper.reduce_across_processes(conf_matrix)
        if torch.is_tensor(conf_matrix):
            conf_matrix = conf_matrix.cpu().numpy()

        if verbose:
            with open(os.path.join(final_output_dir, 'conf_mat.pkl'), 'wb') as file:
                pickle.dump(conf_matrix, file)

    end = timeit.default_timer()
    if verbose:
        scores_msg = []
        for k, v in scores.items():
            if isinstance(v, np.ndarray):
                v = np.array2string(v.flatten(), precision=4, separator=', ')
            elif isinstance(v, collections.abc.Sequence):
                v = '[' + ', '.join(map(lambda x: f'{x:.4f}', v)) + ']'
            elif isinstance(v, float):
                v = f'{v:.4f}'
            scores_msg.append(f'{k}={v}')

        msg = '\nMetrics Y: \n' + ', '.join(scores_msg)
        logger.info(msg)

        logger.info(f'Seconds: {end-start}')
        #logger.info(f'Hours: {(end-start)/3600:.2f}')
        logger.info('Done!')

    if args.distributed:
        dist_helper.cleanup()

if __name__ == '__main__':
    main()
