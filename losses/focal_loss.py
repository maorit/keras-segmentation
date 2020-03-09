"""
## Focal loss

Focal loss 的表达式:

$$
FL(p_t) = -\alpha_t(1-p_t)^\gamma \log(p_t)
$$

其中


$$
p_t = \left \{
    \begin{aligned}
    &p 	 &		&\text{if} \; y=1  \\
    &1-p &		&\text{otherwise,}
    \end{aligned}
\right.
$$

原[论文地址](https://arxiv.org/pdf/1708.02002.pdf)

[参考实现](https://blog.csdn.net/u011583927/article/details/90716942)

"""

import tensorflow as tf


def focal_loss(gamma=2., alpha=4.):
    gamma = float(gamma)
    alpha = tf.constant(alpha, dtype=tf.float32)

    def focal_loss_fixed(y_true, y_pred):
        """
        这个loss只惩罚正例
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)

    return focal_loss_fixed
