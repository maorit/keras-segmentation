from keras.applications.vgg16 import VGG16


def vgg_encoder(input_height, input_width):
    """
    下载预训练好的VGG16模型并返回其特征层
    :param input_height: 输入尺寸input_height
    :param input_width: 输入尺寸input_width
    :return: VGG16模型
    """

    # 下载vgg16模型并锁定各层的权值
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(input_height, input_width, 3))
    for layer in vgg_model.layers:
        layer.trainable = False
    # 提取特征层
    img_input = vgg_model.get_layer(index=0).output
    f1 = vgg_model.get_layer('block1_pool').output
    f2 = vgg_model.get_layer('block2_pool').output
    f3 = vgg_model.get_layer('block3_pool').output
    f4 = vgg_model.get_layer('block4_pool').output
    f5 = vgg_model.get_layer('block5_pool').output
    # 返回模型的输入层和特征层
    return img_input, [f1, f2, f3, f4, f5]
