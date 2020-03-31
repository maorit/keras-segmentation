import numpy as np
import tensorflow as tf

class_freq = np.array(
    [0.6959531209240221, 0.11719239841807973, 0.06354487492487981, 0.1147020975748698, 0.11007367647611178,
     0.0969464093789287, 0.31713197170159757, 0.15578770637512207, 0.28628443943635196, 0.10850246532543285,
     0.18360745906829834, 0.22228599176174257, 0.1948848755891658, 0.1909648670869715, 0.19605655434690875,
     0.1515207592718202, 0.11043455542587652, 0.1966079833015563, 0.2172339654737903, 0.2676844309611493,
     0.151970507150673])


def focal_loss(gamma=2., beta=0, positive_only=True):
    """
    :param gamma: 聚焦系数
    :param beta: 加权系数
    :param positive_only: 是否只考虑正例
    :return: 对应的focal loss函数
    """
    gamma = tf.constant(gamma, dtype=tf.float32)
    alpha = tf.constant(class_freq ** beta, dtype=tf.float32)
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
