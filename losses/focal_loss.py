import tensorflow as tf


def focal_loss(gamma=2., alpha=4., positive_only=True):
    """
    :param gamma: 聚焦系数
    :param alpha: 加权系数
    :param positive_only: 是否只考虑正例
    :return:
    """
    gamma = tf.constant(gamma, dtype=tf.float32)
    alpha = tf.constant(alpha, dtype=tf.float32)
    epsilon = tf.keras.backend.epsilon()

    def focal_loss_positive_and_negative(y_true, y_pred):
        """
        既考虑正例损失，又考虑负例损失
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        y_t = (y_true * y_pred) + (1 - y_true) * (1 - y_pred)
        ce = -tf.math.log(y_t)
        weight = (1 - y_t) ** gamma
        fl = alpha * ce * weight

        return tf.reduce_mean(tf.reduce_sum(fl, axis=-1))

    def focal_loss_positive_only(y_true, y_pred):
        """
        只考虑正例损失，不考虑负例损失
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        y_t = (y_true * y_pred) + (1 - y_true) * (1 - y_pred)
        ce = -tf.math.log(y_t)
        weight = (1 - y_t) ** gamma
        fl = alpha * ce * weight

        return tf.reduce_mean(tf.boolean_mask(fl, tf.cast(y_true, tf.bool)))

    if positive_only:
        return focal_loss_positive_only
    else:
        return focal_loss_positive_and_negative
