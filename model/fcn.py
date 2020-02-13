from keras.layers import Conv2D, Dropout, Conv2DTranspose

from config import IMAGE_ORDERING
from utils.model_utils import get_segmentation_model
from model.vgg16 import vgg_encoder


def fcn_32(n_classes, encoder=vgg_encoder, input_height=416, input_width=608):
    img_input, levels = encoder(input_height=input_height, input_width=input_width)
    [f1, f2, f3, f4, f5] = levels

    o = f5

    o = Conv2D(4096, (7, 7), activation='relu', padding='same', data_format=IMAGE_ORDERING)(o)
    o = Dropout(0.5)(o)
    o = (Conv2D(4096, (1, 1), activation='relu', padding='same', data_format=IMAGE_ORDERING))(o)
    o = Dropout(0.5)(o)
    o = (Conv2D(n_classes, (1, 1), kernel_initializer='he_normal', data_format=IMAGE_ORDERING))(o)
    o = Conv2DTranspose(n_classes, kernel_size=(64, 64), strides=(32, 32), use_bias=False, data_format=IMAGE_ORDERING)(
        o)

    model = get_segmentation_model(img_input, o)
    model.model_name = "fcn_32"
    return model
