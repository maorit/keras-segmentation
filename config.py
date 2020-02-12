import numpy as np

TARGET_HEIGHT = 240  # 经测试,训练集中图片的最小尺寸为height=240,width=292
TARGET_WIDTH = 292
NCLASSES = 21
batch_size = 16

train_idx_path = r'data/VOCdevkit/VOC2007/ImageSets/Segmentation/train.txt'
val_idx_path = r'data/VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt'

img_dir = r'data/VOCdevkit/VOC2007/JPEGImages/'
mask_dir = r'data/VOCdevkit/VOC2007/SegmentationClass/'

log_dir = r'logs/'

palette = np.array(
    [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
     [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
     [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]], dtype='uint8')
