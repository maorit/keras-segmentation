import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import N_CLASSES, INPUT_WIDTH, INPUT_HEIGHT, PALETTE
from model.fcn import fcn_32
from utils.data_utils import _load_image


def get_model(model_fn, hyper_permutation):
    """
    获取模型并加载训练好的权重
    :param model_fn: 生成模型的函数
    :param hyper_permutation: 超参数组合(权重文件名),str
    :return: 加载了预训练权重的模型
    """
    model = model_fn(n_classes=N_CLASSES, input_width=INPUT_WIDTH, input_height=INPUT_HEIGHT)
    model.load_weights(f"E:\\keras-segmentation\\logs\\{hyper_permutation}")
    return model


def predict(model, img_id):
    """
    进行预测并获取输出
    :param model: 预训练的模型
    :param img_id: 图片id,str
    :return: 模型输出,np.array,(1,INPUT_WIDTH*INPUT_HEIGHT,21)
    """
    test_data = _load_image(image_path=f'data\\VOCdevkit\\VOC2012\\JPEGImages\\{img_id}.jpg', width=INPUT_WIDTH,
                            height=INPUT_HEIGHT)
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
    original_img = Image.open(f'data\\VOCdevkit\\VOC2012\\JPEGImages\\{img_id}.jpg')
    output_img = output_img.resize(original_img.size)
    return output_img


if __name__ == '__main__':
    # 参数设置
    model_fn = fcn_32
    img_id = '2007_000129'
    hyper_permutation = 'fcn32_FocalLoss_PositiveNegativeBoth_OnesBalance_lr1e4_gamma2.h5'
    # 展示真实分割
    true_seg = Image.open(f'data\\VOCdevkit\\VOC2012\\SegmentationClass\\{img_id}.png')
    plt.imshow(true_seg)
    plt.show()
    # 展示预测分割
    model = get_model(model_fn=model_fn, hyper_permutation=hyper_permutation)
    output_data = predict(model=model, img_id=img_id)
    output_img = clazznum2image(output_data=output_data, img_id=img_id)
    plt.imshow(output_img)
    plt.title(f'{hyper_permutation}----{img_id}')
    plt.show()
