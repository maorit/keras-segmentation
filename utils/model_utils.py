from config import N_CLASSES, INPUT_WIDTH, INPUT_HEIGHT


def get_pretrained_model(model_fn, hyper_permutation):
    """
    获取模型并加载训练好的权重
    :param model_fn: 生成模型的函数
    :param hyper_permutation: 超参数组合(权重文件名),str
    :return: 加载了预训练权重的模型
    """
    model = model_fn(n_classes=N_CLASSES, input_width=INPUT_WIDTH, input_height=INPUT_HEIGHT)
    model.load_weights(f"E:\\keras-segmentation\\logs\\{hyper_permutation}")
    return model
