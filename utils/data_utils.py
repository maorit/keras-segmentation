import itertools
import random
from pathlib import Path

import numpy as np
from PIL import Image

from config import TRAIN_IDX_PATH, VAL_IDX_PATH, IMG_DIR, SEG_DIR, IMAGE_ORDERING


def _load_image(image_path, width: int, height: int, data_format=IMAGE_ORDERING) -> np.ndarray:
    """
    读取原图片,生成输入给模型的图片数据
    :param image_path: 输入图片的路径,pathlib.Path对象或字符串
    :param width: 生成数据的宽度
    :param height: 生成数据的高度
    :param data_format: 生成数据维度的顺序,one of "channels_last" or "channels_first"
    :return: 输入给模型的图片数据
    """
    # 读取图像文件并正规化
    img = Image.open(image_path)
    img = img.resize((width, height))
    img = np.array(img)
    # img = img / 255.0
    # 调整对应的通道顺序
    if data_format == 'channels_first':
        img = np.rollaxis(img, 2, 0)

    return img


def _load_label(seg_image_path: Path, n_classes: int, width: int, height: int) -> np.ndarray:
    """
    读取分类结果图片,生成输入给模型的分割结果数据
    :param seg_image_path: 分割结果图片的路径,pathlib.Path对象
    :param n_classes: 分割目标类别数
    :param width: 生成数据的宽度
    :param height: 生成数据的高度
    :return: 输入给模型的分割结果数据
    """

    # 读取标签数据并转为numpy数组
    seg_img = Image.open(seg_image_path)
    seg_img = seg_img.resize((width, height), Image.NEAREST)
    seg_img = np.array(seg_img, dtype=np.int32)
    # 转为one-hot编码
    seg_array = np.zeros((height, width, n_classes))
    seg_img[seg_img == 255] = 0
    for c in range(n_classes):
        seg_array[:, :, c] = (seg_img == c)
    # 转为向量
    seg_array = seg_array.reshape(width * height, n_classes)

    return seg_array


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
    # 读取文件,获取未经处理的id列表
    with open(idx_path, 'r') as f:
        idxs = f.read().splitlines()
    # 处理id列表并将其转为生成器
    random.shuffle(idxs)
    idx_generator = iter(idxs)
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

    # 获取用于分类的id生成器
    ids = _generate_image_ids(stage=stage)
    # 循环产生每批次传给网络的输入数据
    while True:
        # 清空图像和标签数据列表
        img_arrays = []
        seg_arrays = []
        # 循环产生每个图像和标签数据
        for _ in range(batch_size):
            # 获取下一个样本id
            id = next(ids)
            # 生成图片数据
            img_path = Path.joinpath(IMG_DIR, f'{id}.jpg')
            img_data = _load_image(img_path, width=input_width, height=input_height)
            img_arrays.append(img_data)
            # 生成对应的标签数据
            seg_path = Path.joinpath(SEG_DIR, f'{id}.png')
            seg_data = _load_label(seg_path, n_classes=n_classes, width=output_width, height=output_height)
            seg_arrays.append(seg_data)

        # 生成一批次的数据
        yield (np.array(img_arrays), np.array(seg_arrays))
