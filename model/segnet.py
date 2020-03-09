from keras import Model
from keras.layers import Conv2D, Dropout, Conv2DTranspose, Add, Reshape, Activation

from config import IMAGE_ORDERING
from model.vgg16 import vgg_encoder
from utils.model_utils import _crop_to_size, _crop_to_same


def segnet(n_classes, input_height, input_width, encoder=vgg_encoder):
    """
    根据给定的encoder,构造SegNet模型
    :param n_classes: 分类数
    :param input_height: 输入尺寸input_height
    :param input_width: 输入尺寸input_width
    :param encoder: 编码器
    :return: SegNet模型
    """

    img_input, [h1, h2, h3, h4, h5] = encoder(input_height=input_height, input_width=input_width)

