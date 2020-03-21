from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dropout, Conv2DTranspose, Add, Reshape, Activation

from config import IMAGE_ORDERING
from model.vgg16 import vgg_encoder
from utils.model_utils import _crop_to_size, _crop_to_same


def fcn_32(n_classes, input_height, input_width, encoder=vgg_encoder):
    """
    根据给定的encoder,构造FCN32模型
    :param n_classes: 分类数
    :param input_height: 输入尺寸input_height
    :param input_width: 输入尺寸input_width
    :param encoder: 编码器
    :return: FCN32模型
    """

    img_input, [f1, f2, f3, f4, f5] = encoder(input_height=input_height, input_width=input_width)

    o = f5
    o = Conv2D(256, (7, 7), activation='relu', padding='same', data_format=IMAGE_ORDERING)(o)
    o = Dropout(0.5)(o)
    o = Conv2D(256, (1, 1), activation='relu', padding='same', data_format=IMAGE_ORDERING)(o)
    o = Dropout(0.5)(o)
    o = Conv2D(n_classes, (1, 1), kernel_initializer='he_normal', data_format=IMAGE_ORDERING)(o)
    o = Conv2DTranspose(n_classes, kernel_size=(64, 64), strides=(32, 32), use_bias=False, data_format=IMAGE_ORDERING)(
        o)
    o = _crop_to_size(img_input, o, input_height, input_width, target_height=input_height, target_width=input_width)
    o = Reshape((-1, n_classes))(o)
    o = Activation('softmax')(o)

    return Model(img_input, o)


def fcn_16(n_classes, input_height, input_width, encoder=vgg_encoder):
    """
    根据给定的encoder,构造FCN16模型
    :param n_classes: 分类数
    :param input_height: 输入尺寸input_height
    :param input_width: 输入尺寸input_width
    :param encoder: 编码器
    :return: FCN16模型
    """

    img_input, [f1, f2, f3, f4, f5] = encoder(input_height=input_height, input_width=input_width)
    # 提取第5层feature map
    o = f5
    o = Conv2D(256, (7, 7), activation='relu', padding='same', data_format=IMAGE_ORDERING)(o)
    o = Dropout(0.5)(o)
    o = Conv2D(256, (1, 1), activation='relu', padding='same', data_format=IMAGE_ORDERING)(o)
    o = Dropout(0.5)(o)
    o = Conv2D(21, (1, 1), activation='linear', padding='same', data_format=IMAGE_ORDERING)(o)
    o = Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False, data_format=IMAGE_ORDERING)(o)
    # 提取第4层feature map并汇总
    o1 = f4
    o1 = Conv2D(21, (1, 1), activation='linear', padding='same', data_format=IMAGE_ORDERING)(o1)
    o, o1 = _crop_to_same(img_input, o, o1, input_height, input_width)
    o = Add()([o, o1])
    # 上采样并整理输出tensor形状
    o = Conv2DTranspose(n_classes, kernel_size=(32, 32), strides=(16, 16), use_bias=False, data_format=IMAGE_ORDERING)(
        o)
    o = _crop_to_size(img_input, o, input_height, input_width, target_height=input_height, target_width=input_width)
    o = Reshape((-1, n_classes))(o)
    o = Activation('softmax')(o)

    return Model(img_input, o)


def fcn_8(n_classes, input_height, input_width, encoder=vgg_encoder):
    """
    根据给定的encoder,构造FCN8模型
    :param n_classes: 分类数
    :param input_height: 输入尺寸input_height
    :param input_width: 输入尺寸input_width
    :param encoder: 编码器
    :return: FCN8模型
    """

    img_input, [f1, f2, f3, f4, f5] = encoder(input_height=input_height, input_width=input_width)
    # 提取第5层feature map
    o = f5
    o = Conv2D(256, (7, 7), activation='relu', padding='same', data_format=IMAGE_ORDERING)(o)
    o = Dropout(0.5)(o)
    o = Conv2D(256, (1, 1), activation='relu', padding='same', data_format=IMAGE_ORDERING)(o)
    o = Dropout(0.5)(o)
    o = Conv2D(21, (1, 1), activation='linear', padding='same', data_format=IMAGE_ORDERING)(o)
    o = Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False, data_format=IMAGE_ORDERING)(o)
    # 提取第4层feature map并汇总
    o1 = f4
    o1 = Conv2D(21, (1, 1), activation='linear', padding='same', data_format=IMAGE_ORDERING)(o1)
    o, o1 = _crop_to_same(img_input, o, o1, input_height, input_width)
    o = Add()([o, o1])
    o = Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False, data_format=IMAGE_ORDERING)(o)
    # 提取第三层feature map并汇总
    o1 = f3
    o1 = Conv2D(n_classes, (1, 1), data_format=IMAGE_ORDERING)(o1)
    o, o1 = _crop_to_same(img_input, o, o1, input_height, input_width)
    o = Add()([o, o1])
    # 上采样并整理输出tensor形状
    o = Conv2DTranspose(n_classes, kernel_size=(16, 16), strides=(8, 8), use_bias=False, data_format=IMAGE_ORDERING)(o)
    o = _crop_to_size(img_input, o, input_height, input_width, target_height=input_height, target_width=input_width)
    o = Reshape((-1, n_classes))(o)
    o = Activation('softmax')(o)

    return Model(img_input, o)
