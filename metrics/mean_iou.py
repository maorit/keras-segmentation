import keras.backend as K


def mean_iou(n_classes):
    def _iou(y_true, y_pred, clazz):
        """
        计算第clazz类的iou
        """
        y_true = K.cast(K.equal(K.argmax(y_true, axis=-1), clazz), K.floatx())
        y_pred = K.cast(K.equal(K.argmax(y_pred, axis=-1), clazz), K.floatx())
        intersection = K.sum(y_true * y_pred)
        union = K.sum(y_true) + K.sum(y_pred) - intersection
        return K.switch(K.equal(union, 0), 1.0, intersection / union)

    def _mean_iou(y_true, y_pred):
        """
        计算平均iou
        """
        mean_iou = K.variable(0)
        for clazz in range(n_classes):
            mean_iou = mean_iou + _iou(y_true, y_pred, clazz)
        return mean_iou / n_classes

    return _mean_iou
