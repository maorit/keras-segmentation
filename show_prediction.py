import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import N_CLASSES, INPUT_WIDTH, INPUT_HEIGHT, PALETTE, SEG_DIR, IMG_DIR
from model.fcn import fcn_32
from utils.data_utils import _load_image
from utils.model_utils import get_pretrained_model


def predict(model, img_id):
    """
    进行预测并获取输出
    :param model: 预训练的模型
    :param img_id: 图片id,str
    :return: 模型输出,np.array,(1,INPUT_WIDTH*INPUT_HEIGHT,21)
    """
    test_data = _load_image(image_path=os.path.join(IMG_DIR, f'{img_id}.jpg'), width=INPUT_WIDTH, height=INPUT_HEIGHT)
    test_data = np.array([test_data])
    output_data = model.predict(test_data)
    return output_data


def clazznum2image(output_data, img_id):
    """
    将神经网络输出转为图片
    :param output_data: 神经网络输出,np.array,(1,INPUT_WIDTH*INPUT_HEIGHT,21)
    :param img_id: 图片id,str
    :return: PIL.Image对象，尺寸为target_shape
    """
    # 调整output形状至(INPUT_WIDTH, INPUT_HEIGHT)
    output_data = output_data[0]
    output_data = output_data.argmax(axis=1)
    output_data.resize(INPUT_WIDTH, INPUT_HEIGHT)
    # 根据数组,生成PIL.Image对象
    output_img = np.zeros((INPUT_WIDTH, INPUT_HEIGHT, 3))
    for c in range(N_CLASSES):
        output_img[:, :, 0] += (output_data == c).astype(np.uint8) * PALETTE[c, 0]  # r
        output_img[:, :, 1] += (output_data == c).astype(np.uint8) * PALETTE[c, 1]  # g
        output_img[:, :, 2] += (output_data == c).astype(np.uint8) * PALETTE[c, 2]  # b
    output_img = Image.fromarray(output_img.astype(np.uint8))
    # 将图片缩放到原图片大小
    original_img = Image.open(os.path.join(IMG_DIR, f'{img_id}.jpg'))
    output_img = output_img.resize(original_img.size)
    return output_img


if __name__ == '__main__':
    # 参数设置
    model_fn = fcn_32
    img_id = '2009_003849'
    hyper_permutation = 'fcn32_FocalLoss_PositiveNegativeBoth_OnesBalance_lr1e4_gamma5.h5'
    # hyper_permutation = 'fcn32_CrossEntropy_null_null_1e4.h5'
    # 展示真实分割
    true_seg = Image.open(os.path.join(SEG_DIR, f'{img_id}.png'))
    plt.imshow(true_seg)
    plt.show()
    # 展示预测分割
    model = get_pretrained_model(model_fn=model_fn, hyper_permutation=hyper_permutation)
    output_data = predict(model=model, img_id=img_id)
    output_img = clazznum2image(output_data=output_data, img_id=img_id)
    # output_img.save(f'{hyper_permutation}----{img_id}.png')
    plt.imshow(output_img)
    plt.title(f'{hyper_permutation}----{img_id}')
    plt.show()
