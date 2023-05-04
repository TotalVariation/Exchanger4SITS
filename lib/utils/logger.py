# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# ------------------------------------------------------------------------------
import os
import logging
import time
from pathlib import Path


def create_logger(cfg, cfg_name, phase='train', n_fold=None):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name
    if n_fold is not None:
        final_output_dir = final_output_dir / f'Fold_{n_fold}'

    print('=> creating {}'.format(final_output_dir))
    if phase == "train":
        final_output_dir.mkdir(parents=True, exist_ok=False)
    else:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
            (cfg_name + '_' + str(n_fold) + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    if 'train' in phase:
        tensorboard_log_dir.mkdir(parents=True, exist_ok=False)

    return logger, str(final_output_dir), str(tensorboard_log_dir)
