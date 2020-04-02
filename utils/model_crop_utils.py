from tensorflow.keras import Model
from tensorflow.keras.layers import Cropping2D

from config import IMAGE_ORDERING


def _get_output_shape(img_input, output, input_height, input_width):
    """
    计算以img_input为输入层,输入层尺寸为input_height, input_width的模型的output层输出尺寸
    :return: 输出层尺寸(output_height, output_width)
    """
    _, output_height, output_width, _ = Model(img_input, output).compute_output_shape(
        input_shape=(1, input_height, input_width, 3))
    return (output_height, output_width)


def _crop_to_same(img_input, output1, output2, input_height, input_width):
    """
    将output1和output2两层裁剪到相同尺寸
    :param img_input: 模型的输入层
    :param output1: 待裁剪的层1
    :param output2: 待裁剪的层1
    :param input_height: 输入层尺寸
    :param input_width: 输入层尺寸
    :return: 裁剪好后的output1和output2两层
    """
    # 计算两层的输出尺寸
    output_height1, output_width1 = _get_output_shape(img_input, output1, input_height, input_width)
    output_height2, output_width2 = _get_output_shape(img_input, output2, input_height, input_width)
    # 对比两个维度上的尺寸差值
    cx = abs(output_width2 - output_width1)
    cy = abs(output_height2 - output_height1)
    # 裁剪
    if output_width1 > output_width2:
        output1 = Cropping2D(cropping=((0, 0), (0, cx)), data_format=IMAGE_ORDERING)(output1)
    else:
        output2 = Cropping2D(cropping=((0, 0), (0, cx)), data_format=IMAGE_ORDERING)(output2)
    if output_height1 > output_height2:
        output1 = Cropping2D(cropping=((0, cy), (0, 0)), data_format=IMAGE_ORDERING)(output1)
    else:
        output2 = Cropping2D(cropping=((0, cy), (0, 0)), data_format=IMAGE_ORDERING)(output2)

    return output1, output2


def _crop_to_size(img_input, output, input_height, input_width, target_height, target_width):
    """
    将output层裁剪到(target_height,target_width)尺寸
    :param img_input: 输入层
    :param output: 待裁剪的层
    :param input_height: 输入层尺寸input_height
    :param input_width: 输入层尺寸input_width
    :param target_height: 裁剪目标尺寸target_height
    :param target_width: 裁剪目标尺寸target_width
    :return: 裁剪好后的output层
    """

    assert target_height <= input_height
    assert target_width <= input_width

    # 计算输出尺寸
    output_height, output_width = _get_output_shape(img_input, output, input_height, input_width)
    # 对比两个维度上的尺寸差值
    cy = abs(output_height - target_height)
    cx = abs(output_width - target_width)
    # 裁剪
    if cx > 0 or cy > 0:
        output = Cropping2D(cropping=((0, cy), (0, cx)), data_format=IMAGE_ORDERING)(output)

    return output
