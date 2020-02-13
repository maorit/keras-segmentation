from pathlib import Path

import numpy as np

INPUT_HEIGHT = 512  # 经测试,训练集中图片的最小尺寸为height=240,width=292
INPUT_WIDTH = 512

N_CLASSES = 21
TRAIN_BATCH_SIZE = 16
VAL_BATCH_SIZE = 16

TRAIN_IDX_PATH = Path(r'data/VOCdevkit/VOC2007/ImageSets/Segmentation/train.txt')
VAL_IDX_PATH = Path(r'data/VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt')

IMG_DIR = Path(r'data/VOCdevkit/VOC2007/JPEGImages/')
SEG_DIR = Path(r'data/VOCdevkit/VOC2007/SegmentationClass/')
LOG_DIR = Path(r'logs/')

PALETTE = np.array(
    [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
     [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
     [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]], dtype='uint8')

IMAGE_ORDERING = 'channels_last'
