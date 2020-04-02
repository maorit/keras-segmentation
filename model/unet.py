from tensorflow.keras import Model
from tensorflow.keras.backend import concatenate
from tensorflow.keras.layers import Conv2D, Reshape, Activation, BatchNormalization, UpSampling2D

from config import IMAGE_ORDERING
from model.vgg16 import vgg_encoder
from utils.model_crop_utils import _crop_to_size

if IMAGE_ORDERING == 'channels_first':
    MERGE_AXIS = 1
elif IMAGE_ORDERING == 'channels_last':
    MERGE_AXIS = -1


def unet(n_classes, input_height, input_width, encoder=vgg_encoder):
    """
    根据给定的encoder,构造unet模型
    :param n_classes: 分类数
    :param input_height: 输入尺寸input_height
    :param input_width: 输入尺寸input_width
    :param encoder: 编码器
    :return: SegNet模型
    """

    img_input, [f1, f2, f3, f4, f5] = encoder(input_height=input_height, input_width=input_width)

    o = f5
    o = Conv2D(512, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o)
    o = BatchNormalization()(o)
    o = UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(o)
    o = concatenate([o, f4], axis=MERGE_AXIS)

    o = Conv2D(256, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o)
    o = BatchNormalization()(o)
    o = UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(o)
    o = concatenate([o, f3], axis=MERGE_AXIS)

    o = Conv2D(128, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o)
    o = BatchNormalization()(o)
    o = UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(o)
    o = concatenate([o, f2], axis=MERGE_AXIS)

    o = Conv2D(64, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o)
    o = BatchNormalization()(o)
    o = UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(o)
    o = concatenate([o, f1], axis=MERGE_AXIS)

    o = Conv2D(64, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o)
    o = BatchNormalization()(o)
    o = UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(o)

    o = Conv2D(n_classes, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o)
    o = _crop_to_size(img_input, o, input_height, input_width, target_height=input_height, target_width=input_width)
    o = Reshape((-1, n_classes))(o)
    o = Activation('softmax')(o)

    return Model(img_input, o)
