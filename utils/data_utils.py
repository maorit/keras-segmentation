import itertools
import random
from pathlib import Path

import cv2
import numpy as np

from config import PALETTE, TRAIN_IDX_PATH, VAL_IDX_PATH, IMG_DIR, SEG_DIR, IMAGE_ORDERING


def _get_image_array(image_path: Path, width: int, height: int, data_format=IMAGE_ORDERING) -> np.ndarray:
    """
    读取原图片,生成输入给模型的图片数据
    :param image_path: 输入图片的路径,pathlib.Path对象
    :param width: 生成数据的宽度
    :param height: 生成数据的高度
    :param data_format: 生成数据维度的顺序,one of "channels_last" or "channels_first"
    :return: 输入给模型的图片数据
    """
    img = cv2.imread(str(image_path.absolute()), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (width, height))
    img = img / 255.0

    if data_format == 'channels_first':
        img = np.rollaxis(img, 2, 0)

    return img


def _get_seg_array(seg_image_path: Path, n_classes: int, width: int, height: int) -> np.ndarray:
    """
    读取分类结果图片,生成输入给模型的分割结果数据
    :param seg_image_path: 分割结果图片的路径,pathlib.Path对象
    :param n_classes: 分割目标类别数
    :param width: 生成数据的宽度
    :param height: 生成数据的高度
    :return: 输入给模型的分割结果数据
    """

    '''
    cv2有两个比较坑的点:
    1. 别的都是行先序,但是它是列先序,所以resize时候注意先width,后height
    2. 默认读入彩色图片为bgr顺序,然而实际常用的是rgb顺序
    '''
    seg_img = cv2.imread(str(seg_image_path.absolute()), cv2.IMREAD_COLOR)
    seg_img = cv2.resize(seg_img, (width, height), interpolation=cv2.INTER_NEAREST)
    seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)

    seg_labels = np.zeros((height, width, n_classes))
    for c in range(n_classes):
        seg_labels[:, :, c] = (seg_img == PALETTE[c].reshape(1, 1, 3)).all(axis=2).astype(np.float)
    seg_labels = seg_labels.reshape(width * height, n_classes)

    return seg_labels


def _generate_image_ids(stage: str) -> itertools.cycle:
    """
    获取所有图片的编号(id)
    :param stage: 阶段,one of 'train' or 'val'
    :return: 所有图片的编号
    """

    # 获取保存id的文件
    if stage == 'train':
        idx_path = TRAIN_IDX_PATH
    elif stage == 'val':
        idx_path = VAL_IDX_PATH
    else:
        raise ValueError('传入的参数必须为"train"或"val"之一')

    # 读取文件
    with open(idx_path, 'r') as f:
        idxs = f.readlines()

    # 将列表转为生成器
    random.shuffle(idxs)
    idx_generator = map(str.strip, idxs)
    idx_generator = itertools.cycle(idx_generator)

    return idx_generator


def generate_input_data(stage: str, batch_size: int, n_classes: int, input_width: int, input_height: int,
                        output_width: int, output_height: int):
    """
    生成一批用于训练或验证的数据
    :param stage: 阶段,one of 'train' or 'val'
    :param batch_size: 每批数据的样本个数
    :return: 一批用于训练或验证的数据
    """

    # 获取用于分类的id列表
    ids = _generate_image_ids(stage=stage)

    while True:
        # 生成batch_size个图片和分割数据
        img_arrays = []
        seg_arrays = []
        for _ in range(batch_size):
            # 下一个样本id
            id = next(ids)

            # 生成图片数据
            img_path = Path.joinpath(IMG_DIR, f'{id}.jpg')
            img_data = _get_image_array(img_path, width=input_width, height=input_height)
            img_arrays.append(img_data)

            # 生成对应的分割数据
            seg_path = Path.joinpath(SEG_DIR, f'{id}.png')
            seg_data = _get_seg_array(seg_path, n_classes=n_classes, width=output_width, height=output_height)
            seg_arrays.append(seg_data)

        yield (np.array(img_arrays), np.array(seg_arrays))
