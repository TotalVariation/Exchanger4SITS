# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = '/home/cai-x/crop_cls/proj_001/Exchanger4SITS/output'
_C.LOG_DIR = '/home/cai-x/crop_cls/proj_001/Exchanger4SITS/log'
_C.PRINT_FREQ = 5

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'Classifier'
_C.MODEL.PRETRAINED = './'

# loss
_C.LOSS = CN()
_C.LOSS.TYPE = 'crossentropy'
_C.LOSS.SMOOTH_FACTOR = 0.0
_C.LOSS.FOCAL = [0.25, 2.0]
_C.LOSS.IGNORE_INDEX = -1

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = '/home/cai-x/data/PASTIS-R_PixelSet'
_C.DATASET.READER = 'PASTISPixelSetReader'
_C.DATASET.DATASET = 'PASTIS-R_PixelSet'
_C.DATASET.MODALITY = ['S2']
_C.DATASET.INPUT_DIM = [10]
_C.DATASET.NUM_CLASSES = 18
_C.DATASET.N_FOLD = 0
_C.DATASET.NBINS = 16
_C.DATASET.TEMP_DROP_RATE = [[0.2, 0.4]]
_C.DATASET.PAD_VALUE = 0
_C.DATASET.TASK_TYPE = 'cls'
_C.DATASET.TRAIN_TILES = [{'year': 2017, 'country': 'austria', 'tile': '33UVP'},
                         {'year': 2017, 'country': 'denmark', 'tile': '32VNH'},
                          {'year': 2017, 'country': 'france', 'tile': '30TXT'}]
_C.DATASET.TEST_TILES = [{'year': 2017, 'country': 'france', 'tile': '31TCJ'}]
_C.DATASET.THING_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
_C.DATASET.Z_NORM = True

# training
_C.TRAIN = CN()
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 50
_C.TRAIN.BATCH_SIZE_PER_GPU = 128
_C.TRAIN.RANDOM_CROP = True
_C.TRAIN.CROP_SIZE = (32, 32)

# classification head specification
_C.CLS_HEAD = CN()
_C.CLS_HEAD.TEMP_ENCODER_TYPE = 'exchanger'
_C.CLS_HEAD.POS_ENCODE_TYPE = 'default'
_C.CLS_HEAD.WITH_GDD_POS = False
_C.CLS_HEAD.PE_DIM = 128
_C.CLS_HEAD.PE_T = 1000
_C.CLS_HEAD.MAX_TEMP_LEN = 1000
_C.CLS_HEAD.PROJ_HID_NLAYERS = 2
_C.CLS_HEAD.PROJ_HID_DIM = 256
_C.CLS_HEAD.PROJ_BOT_DIM = 64
_C.CLS_HEAD.PROJ_NORM_TYPE = 'batchnorm'
_C.CLS_HEAD.PROJ_ACT_TYPE = 'gelu'
_C.CLS_HEAD.TAU = 0.1

# exchanger specification
_C.EXCHANGER = CN()
_C.EXCHANGER.EMBED_DIMS = [128, 128]
_C.EXCHANGER.NUM_TOKEN_LIST = [8, 8]
_C.EXCHANGER.NUM_HEADS_LIST = [8, 8]
_C.EXCHANGER.DROP_PATH_RATE = 0.1
_C.EXCHANGER.MLP_NORM = 'batchnorm'
_C.EXCHANGER.MLP_ACT = 'gelu'

# pseltae head specification
_C.PSELTAE = CN()
_C.PSELTAE.NORM_TYPE = 'batchnorm'
_C.PSELTAE.ACT_TYPE = 'gelu'
_C.PSELTAE.PROJ_HID_NLAYERS = 2
_C.PSELTAE.PROJ_HID_DIM = 256
_C.PSELTAE.PROJ_BOT_DIM = 64
_C.PSELTAE.PROJ_NORM_TYPE = 'batchnorm'
_C.PSELTAE.PROJ_ACT_TYPE = 'gelu'
_C.PSELTAE.TAU = 0.1

# pse specification
_C.PSE = CN()
_C.PSE.MLP1_DIMS = [32, 64]
_C.PSE.MLP2_DIMS = [128]
_C.PSE.NORM_TYPE = 'batchnorm'
_C.PSE.ACT_TYPE = 'gelu'

# ltae specification
_C.LTAE = CN()
_C.LTAE.D_MODEL = 256
_C.LTAE.D_K = 8
_C.LTAE.N_HEAD = 16
_C.LTAE.NORM_TYPE = 'batchnorm'
_C.LTAE.ACT_TYPE = 'gelu'
_C.LTAE.DROPOUT = 0.1
_C.LTAE.MLP_DIMS = [128]
_C.LTAE.WITH_POS_ENC = True
_C.LTAE.POS_ENC_TYPE = 'default'
_C.LTAE.WITH_GDD_POS = False
_C.LTAE.PE_T = 1000
_C.LTAE.MAX_TEMP_LEN = 1000

# PVT2 specification
_C.PVT2 = CN()
_C.PVT2.DEPTHS = [2, 2, 2, 2]
_C.PVT2.EMBED_DIMS = [64, 128, 320, 512]
_C.PVT2.NUM_HEADS = [1, 2, 5, 8]
_C.PVT2.SR_RATIOS = [8, 4, 2, 1]
_C.PVT2.MLP_RATIOS = [8, 8, 4, 4]
_C.PVT2.QKV_BIAS = True
_C.PVT2.LINEAR = False
_C.PVT2.DROP_RATE = 0.1
_C.PVT2.ATTN_DROP_RATE = 0.
_C.PVT2.DROP_PATH_RATE = 0.1
_C.PVT2.NORM_TYPE = 'layernorm'

# SWIN specification
_C.SWIN = CN()
_C.SWIN.PRETRAIN_IMG_SIZE = 224
_C.SWIN.PATCH_SIZE = 1
_C.SWIN.EMBED_DIM = 96
_C.SWIN.DEPTHS = [2, 2, 6, 2]
_C.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.SWIN.WINDOW_SIZE = 7
_C.SWIN.MLP_RATIO = 4.0
_C.SWIN.QKV_BIAS = True
_C.SWIN.QK_SCALE = None
_C.SWIN.DROP_RATE = 0.1
_C.SWIN.ATTN_DROP_RATE = 0.0
_C.SWIN.DROP_PATH_RATE = 0.2
_C.SWIN.NORM_LAYER_TYPE = 'layernorm'
_C.SWIN.APE = False
_C.SWIN.PATCH_NORM = True
_C.SWIN.OUT_INDICES = [0, 1, 2, 3]
_C.SWIN.FROZEN_STAGES = -1
_C.SWIN.USE_CHECKPOINT = False

# Unet specification
_C.UNET = CN()
_C.UNET.BASE_CHANNELS = 64
_C.UNET.NUM_STAGES = 4
_C.UNET.STRIDES = [1, 1, 1, 1]
_C.UNET.ENC_NUM_CONVS = [2, 2, 2, 2]
_C.UNET.DEC_NUM_CONVS = [2, 2, 2]
_C.UNET.DOWNSAMPLES = [True, True, True]
_C.UNET.ENC_DILATIONS = [1, 1, 1, 1]
_C.UNET.DEC_DILATIONS = [1, 1, 1]
_C.UNET.NORM_TYPE = 'bn'
_C.UNET.ACT_TYPE = 'gelu'
_C.UNET.UPSAMPLE_TYPE = 'interp'

# TSViT cls specification
_C.TEMP_SPAT_TRANSFORMER = CN()
_C.TEMP_SPAT_TRANSFORMER.EMBED_DIMS = 128
_C.TEMP_SPAT_TRANSFORMER.TEMPORAL_DEPTH = 4
_C.TEMP_SPAT_TRANSFORMER.SPATIAL_DEPTH = 2
_C.TEMP_SPAT_TRANSFORMER.NUM_TOKENS = 8
_C.TEMP_SPAT_TRANSFORMER.NUM_HEADS = 8
_C.TEMP_SPAT_TRANSFORMER.ATTN_DROP = 0.
_C.TEMP_SPAT_TRANSFORMER.DROP = 0.1
_C.TEMP_SPAT_TRANSFORMER.DROP_PATH = 0.1
_C.TEMP_SPAT_TRANSFORMER.FFN_RATIO = 4.0
_C.TEMP_SPAT_TRANSFORMER.ACT_TYPE = 'gelu'
_C.TEMP_SPAT_TRANSFORMER.NORM_TYPE = 'layernorm'
_C.TEMP_SPAT_TRANSFORMER.QKV_BIAS = True
_C.TEMP_SPAT_TRANSFORMER.UNTIED_POS_ENCODE = True
_C.TEMP_SPAT_TRANSFORMER.USE_SPACE_TRANSFORMER = True

# GPBlock specification
_C.GPBLOCK = CN()
_C.GPBLOCK.EMBED_DIMS = 128
_C.GPBLOCK.NUM_GROUP_TOKENS = 8
_C.GPBLOCK.NUM_HEADS = 8
_C.GPBLOCK.ACT_TYPE = 'gelu'
_C.GPBLOCK.NORM_TYPE = 'layernorm'
_C.GPBLOCK.FFN_RATIO = 4.0
_C.GPBLOCK.QKV_BIAS = True
_C.GPBLOCK.MIXER_DEPTH = 1
_C.GPBLOCK.MIXER_TOKEN_EXPANSION = 0.5
_C.GPBLOCK.MIXER_CHANNEL_EXPANSION = 4.0
_C.GPBLOCK.DROP = 0.1
_C.GPBLOCK.ATTN_DROP = 0.
_C.GPBLOCK.DROP_PATH = 0.1
_C.GPBLOCK.UNTIED_POS_ENCODE = True
_C.GPBLOCK.ADD_POS_TOKEN = True

# MaskFormer specification
_C.MASKFORMER = CN()
_C.MASKFORMER.BACKBONE_TYPE = 'pvt'
_C.MASKFORMER.PIXEL_DECODER_TYPE = 'fpn'
_C.MASKFORMER.HIDDEN_DIM = 256
_C.MASKFORMER.MASK_DIM = 256
_C.MASKFORMER.DEC_LAYERS = 3
_C.MASKFORMER.NUM_HEADS = 8
_C.MASKFORMER.DIM_FFD = 1024
_C.MASKFORMER.NUM_QUERIES = 100
_C.MASKFORMER.NUM_FEATURE_LEVELS = 3
_C.MASKFORMER.PRE_NORM = False
_C.MASKFORMER.ENFORCE_INPUT_PROJECT = False
_C.MASKFORMER.OBJECT_MASK_THRESHOLD = 0.7
_C.MASKFORMER.OVERLAP_THRESHOLD = 0.7
_C.MASKFORMER.MASK_WEIGHT = 5.0
_C.MASKFORMER.DICE_WEIGHT = 5.0
_C.MASKFORMER.CLS_WEIGHT = 2.0
_C.MASKFORMER.NUM_POINTS = 128
_C.MASKFORMER.OVERSAMPLE_RATIO = 3.0
_C.MASKFORMER.IMPORTANCE_SAMPLE_RATIO = 0.75


# Segmentor specification
_C.SEGMENTOR = CN()
_C.SEGMENTOR.POS_ENCODE_TYPE = 'default'
_C.SEGMENTOR.WITH_GDD_POS = False
_C.SEGMENTOR.PE_DIM = 128
_C.SEGMENTOR.PE_T = 1000
_C.SEGMENTOR.MAX_TEMP_LEN = 1000
_C.SEGMENTOR.SPACE_ENCODER_TYPE = 'unet'


# PaPs
_C.PAPS_HEAD = CN()
_C.PAPS_HEAD.LOSS_CENTER_WEIGHT = 1.0
_C.PAPS_HEAD.LOSS_SIZE_WEIGHT = 1.0
_C.PAPS_HEAD.LOSS_SHAPE_WEIGHT = 1.0
_C.PAPS_HEAD.LOSS_CLASS_WEIGHT = 1.0
_C.PAPS_HEAD.BINARY_THRESHOLD = 0.4
_C.PAPS_HEAD.CENTER_LOSS_ALPHA = 2.0
_C.PAPS_HEAD.CENTER_LOSS_BETA = 4.0
_C.PAPS_HEAD.FOCAL_LOSS_GAMMA = 2.0
_C.PAPS_HEAD.SHAPE_SIZE = 16
_C.PAPS_HEAD.MIN_CONFIDENCE = 0.2
_C.PAPS_HEAD.MIN_REMAIN = 0.5
_C.PAPS_HEAD.MASK_THRESHOLD = 0.4
_C.PAPS_HEAD.MASK_CONV = True


# panoptic segmentor specification
_C.PANOPTIC_SEGMENTOR = CN()
_C.PANOPTIC_SEGMENTOR.POS_ENCODE_TYPE = 'default'
_C.PANOPTIC_SEGMENTOR.WITH_GDD_POS = False
_C.PANOPTIC_SEGMENTOR.PE_DIM = 128
_C.PANOPTIC_SEGMENTOR.PE_T = 1000
_C.PANOPTIC_SEGMENTOR.MAX_TEMP_LEN = 1000
_C.PANOPTIC_SEGMENTOR.SPACE_ENCODER_TYPE = 'unet'


# learning rate
_C.LR = CN()
_C.LR.LR_FACTOR = 0.1
_C.LR.LR_STEP = [0.7, 0.9]
_C.LR.LR = 0.0001
_C.LR.LR_SCHEDULER = 'step'
_C.LR.WARMUP_ITERS_RATIO = 0.1
_C.LR.WARMUP_FACTOR = 1e-03

# optimizer
_C.OPTIMIZER = CN()
_C.OPTIMIZER.TYPE = 'adamw'
_C.OPTIMIZER.MOMENTUM = 0.9
_C.OPTIMIZER.WD = 1e-04

# testing
_C.TEST = CN()
_C.TEST.MULTI_CROP_TEST = True
_C.TEST.BATCH_SIZE_PER_GPU = 1
_C.TEST.MODEL_FILE = './'
_C.TEST.CROP_SIZE = (32, 32)
_C.TEST.RETURN_ATTN = False

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False


def update_config(cfg, args):
    cfg.defrost()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
