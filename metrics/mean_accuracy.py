import keras.backend as K


def mean_acc(n_classes):
    def _acc(y_true, y_pred, clazz):
        """
        计算第clazz类的accuracy
        """
        epsilon = 1e-7
        y_true = K.cast(K.equal(K.argmax(y_true, axis=-1), clazz), K.floatx())
        y_pred = K.cast(K.equal(K.argmax(y_pred, axis=-1), clazz), K.floatx())
        acc = K.sum(y_true * y_pred) / (K.sum(y_true) + epsilon)

        return acc

    def _mean_acc(y_true, y_pred):
        """
        计算平均acc
        """
        mean_acc = K.variable(0)
        for clazz in range(n_classes):
            mean_acc = mean_acc + _acc(y_true, y_pred, clazz)
        return mean_acc / n_classes

    return _mean_acc
