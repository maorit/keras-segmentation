import tensorflow as tf

"""
https://blog.csdn.net/u011583927/article/details/90716942
"""


def focal_loss(gamma=2., alpha=4.):
    gamma = tf.constant(gamma, dtype=tf.float32)
    alpha = tf.constant(alpha, dtype=tf.float32)
    epsilon = tf.keras.backend.epsilon()

    def focal_loss_positive_and_negative(y_true, y_pred):
        """
        既惩罚正例,又惩罚负例
        """

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        y_t = (y_true * y_pred) + (1 - y_true) * (1 - y_pred)
        ce = -tf.math.log(y_t)
        weight = (1 - y_t) ** gamma
        fl = alpha * ce * weight

        return tf.reduce_mean(tf.reduce_sum(fl, axis=-1))

    return focal_loss_positive_and_negative
