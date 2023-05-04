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
import datetime
from pathlib import Path
import json
import random
from functools import partial
import collections

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from fvcore.nn import FlopCountAnalysis, flop_count_table

import _init_paths
import datasets
import models
from config import config, update_config
from apis import train, validate
from utils import create_logger, get_optimizer, get_lr_scheduler, pad_collate, dist_helper


def parse_args():
    parser = argparse.ArgumentParser(description='Train Crop Classification Network')

    parser.add_argument('--cfg',
                        help='experiment configuration file name',
                        required=True,
                        type=str)
    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume training from a checkpoint')
    parser.add_argument('--fine-tune', action='store_true', default=False,
                        help='fine tune from a pretrained model')
    parser.add_argument('--n-fold', type=int, default=None,
                        help='n fold for cross validation')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed')
    parser.add_argument('--workers', type=int, default=16,
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    # distributed training parameters
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
        np.random.seed(seed)
        random.seed(seed)
        cudnn.deterministic = True

    # cudnn related setting
    cudnn.benchmark = True

    n_fold = args.n_fold - 1 if args.n_fold is not None else config.DATASET.N_FOLD
    fold_sequence = [
        [[1, 2, 3], [4], [5]],
        [[2, 3, 4], [5], [1]],
        [[3, 4, 5], [1], [2]],
        [[4, 5, 1], [2], [3]],
        [[5, 1, 2], [3], [4]],
    ]

    # Distributed Training: print logs on the first worker
    verbose = True if dist_helper.is_main_process() else False

    if verbose:
        print("create logger ...")
        logger, final_output_dir, tb_log_dir = create_logger(
            config,
            args.cfg,
            phase=f'train_{n_fold}',
            n_fold=n_fold,
        )
        logger.info(pprint.pformat(args))
        logger.info(config)

    # write Tensorboard logs on the first worker
    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    } if dist_helper.is_main_process() else None

    # prepare data
    train_fold, val_fold, _ = fold_sequence[n_fold]
    train_dataset = eval('datasets.' + config.DATASET.READER)(
        config,
        nfolds=train_fold,
        mode='train'
    )

    val_dataset = eval('datasets.' + config.DATASET.READER)(
        config,
        nfolds=val_fold,
        mode='val'
    )

    # build model
    model = eval('models.' + config.MODEL.NAME)(config, mode='train_val')

    if verbose:
        model.eval()
        logger.info(model)
        tot_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.
        logger.info(f">>> total params: {tot_params:.2f}Mb")
        # provide the summary of model
        #dummy_sample = pad_collate([train_dataset[0]])
        #logger.info(summary(model, inputs=dummy_sample))
        #flops = FlopCountAnalysis(model, dummy_sample)
        #logger.info(flop_count_table(flops))

    if args.distributed:
        # Distributed Training: use DistributedSampler to partition data among workers. Manually
        # specify `num_replicas` and `rank`.
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist_helper.get_world_size(),
            rank=dist_helper.get_rank(),
            shuffle=True,
            drop_last=True,
        )

        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=dist_helper.get_world_size(),
            rank=dist_helper.get_rank(),
            shuffle=False,
            drop_last=True)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    print(f'The length of trainloader on each GPU is: {len(train_sampler)}')
    print(f'The length of valloader on each GPU is: {len(val_sampler)}')

    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=False,
        drop_last=True,
        sampler=train_sampler,
        collate_fn=partial(pad_collate, pad_value=config.DATASET.PAD_VALUE),
        **kwargs
    )

    valloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=False,
        drop_last=True,
        sampler=val_sampler,
        collate_fn=partial(pad_collate, pad_value=config.DATASET.PAD_VALUE),
        **kwargs
    )


    # Load weights from pre-trained models
    if args.fine_tune:
        assert os.path.isfile(config.MODEL.PRETRAINED), f'the path of pretrained model {config.MODEL.PRETRAINED}" is not valid!!'
        model_state_file = config.MODEL.PRETRAINED
        dist_helper.log(f'=> Loading model from {model_state_file}')
        pretrained_dict = torch.load(model_state_file, map_location='cpu')['state_dict']
        model_dict = model.state_dict()
        # check loaded parameters size/shape compatibility
        for k in pretrained_dict:
            if k in model_dict:
                if pretrained_dict[k].shape != model_dict[k].shape:
                    dist_helper.log(f'=> Skip loading parameter {k}, required shape {model_dict[k].shape}, loaded shape {pretrained_dict[k].shape}.')
                    pretrained_dict[k] = model_dict[k]
            else:
                dist_helper.log(f'=> Drop parameter {k}.')
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict.keys()}
        if verbose:
            for k in pretrained_dict.keys():
                logger.info(f'=> Loading {k} from pretrained model')
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    best = 0.
    last_epoch = 0
    # Load weights from checkpoint
    if args.resume:
        model_state_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location='cpu')
            best = checkpoint['best']
            last_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if verbose:
                logger.info(f"=> loaded checkpoint (epoch {last_epoch})")

    model = model.to(device)

    lr_scaler = dist_helper.get_world_size()
    # optimizer
    optimizer = get_optimizer(config, model, lr_scaler)

    epoch_iters = int(train_dataset.__len__() /
                      config.TRAIN.BATCH_SIZE_PER_GPU / dist_helper.get_world_size())
    end_epoch = config.TRAIN.END_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters

    dist_helper.log(f'epoch_iters: {epoch_iters}, max_iters: {num_iters}')

    # learning rate scheduler
    lr_scheduler_dict = {
        'optimizer': optimizer,
        'milestones': [int(s * num_iters) for s in config.LR.LR_STEP],
        'gamma': config.LR.LR_FACTOR,
        'max_iters': num_iters,
        'last_epoch': last_epoch,
        'epoch_iters': epoch_iters,
        'warmup_iters': int(config.LR.WARMUP_ITERS_RATIO * num_iters),
        'warmup_factor': config.LR.WARMUP_FACTOR,
    }
    lr_scheduler = get_lr_scheduler(config.LR.LR_SCHEDULER, **lr_scheduler_dict)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    start_time = time.time()
    task_type = config.DATASET.TASK_TYPE

    for epoch in range(last_epoch, end_epoch):

        if args.distributed:
            train_sampler.set_epoch(epoch)

        train(config, epoch, num_iters,
              trainloader, optimizer, lr_scheduler,
              model, writer_dict, device)

        valid_loss, scores = validate(config, valloader, model, writer_dict, device)

        # save checkpoints and print logs in the main process.
        if dist_helper.get_rank() == 0:
            if task_type == 'cls':
                best_cur = scores['F_score']
            elif task_type == 'sem_seg':
                best_cur = scores['mIoU']
            elif task_type in ('paps_pano', 'maskformer_pano'):
                best_cur = scores['PQ']
            else:
                raise NotImplementedError

            scores_msg = []
            for k, v in scores.items():
                if isinstance(v, np.ndarray):
                    v = np.array2string(v.flatten(), precision=4, separator=', ')
                elif isinstance(v, collections.abc.Sequence):
                    v = '[' + ', '.join(map(lambda x: f'{x:.4f}', v)) + ']'
                elif isinstance(v, float):
                    v = f'{v:.4f}'
                scores_msg.append(f'{k}={v}')

            msg = f'\nValidation Loss: {valid_loss:.4f}, ' + '\nMetrics: \n' + ', '.join(scores_msg)
            logger.info(msg)

            logger.info('=> saving checkpoint to {}'.format(
                final_output_dir + '/checkpoint.pth.tar'))
            torch.save(
                {'epoch': epoch + 1,
                 'best': best,
                 'val_loss': valid_loss,
                 'state_dict': model_without_ddp.state_dict(),
                 'optimizer': optimizer.state_dict()},
                os.path.join(final_output_dir, 'checkpoint.pth.tar')
            )

            writer_dict['writer'].add_scalars(
                "metrics",
                dict([(k, v) for k, v in scores.items() if np.isscalar(v)]),
                global_step=epoch
            )

            if best < best_cur:
                best = best_cur
                torch.save({'epoch': epoch + 1,
                            'best': best,
                            'val_loss': valid_loss,
                            'metrics': scores,
                            'state_dict': model_without_ddp.state_dict()},
                           os.path.join(final_output_dir, 'best.pth'))

            if epoch == end_epoch - 1:
                torch.save({'metrics': scores,
                            'state_dict': model_without_ddp.state_dict()},
                           os.path.join(final_output_dir, 'final_state.pth'))

                writer_dict['writer'].close()
                end_time = time.time() - start_time
                tot_time = str(datetime.timedelta(seconds=int(end_time)))
                logger.info(f'Training Time: {tot_time}')
                logger.info('Done!')

    if dist_helper.get_rank() == 0:
        model_state_file = os.path.join(final_output_dir, 'best.pth')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location='cpu')
            epoch = checkpoint['epoch']
            val_loss = checkpoint['val_loss']
            val_metrics = checkpoint['metrics']

        scores_msg = f"\nEpoch: {epoch}\n"
        scores_msg += "Validation Metrics: \n"
        scores_msg += ", ".join([f"{k}={v:.4f}" for (k, v) in val_metrics.items() if np.isscalar(v)])
        msg = f'\nValidation Loss: {val_loss:.4f}, ' + scores_msg
        logger.info(msg)

    if args.distributed:
        dist_helper.cleanup()

if __name__ == '__main__':
    main()
