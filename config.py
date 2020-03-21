from pathlib import Path

import numpy as np

# 模型参数
INPUT_HEIGHT = 512  # 模型输入尺寸INPUT_HEIGHT
INPUT_WIDTH = 512  # 模型输出尺寸INPUT_WIDTH
N_CLASSES = 21  # 分类数
IMAGE_ORDERING = 'channels_last'  # 图片通道顺序

# 训练参数
TRAIN_IDX_PATH = Path(r'data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt').absolute()  # 记录训练集image_id的文件
VAL_IDX_PATH = Path(r'data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt').absolute()  # 记录验证集image_id的文件
IMG_DIR = Path(r'data/VOCdevkit/VOC2012/JPEGImages/').absolute()  # 图片文件目录
SEG_DIR = Path(r'data/VOCdevkit/VOC2012/SegmentationClass/').absolute()  # 分割结果文件目录
LOG_DIR = r'logs/'  # 日志目录
TRAIN_BATCH_SIZE = 4  # 训练集batchsize
VAL_BATCH_SIZE = 4  # 验证集batchsize

# VOC数据集
PALETTE = np.array(
    [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
     [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
     [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]], dtype='uint8')  # 颜色版:不同类别对应的颜色
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']  # 类别名: 不同类别对应的类名
