import numpy as np
import tensorflow as tf

gamma = tf.constant(2, dtype=tf.float32)
alpha = tf.constant(np.ones(3), dtype=tf.float32)
epsilon = tf.keras.backend.epsilon()

y_true = tf.constant([[0, 1, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0]])
y_pred = tf.constant([[0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7]])

y_true = tf.cast(y_true, tf.float32)
y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)

y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
ce = -tf.math.log(y_t)
weight = tf.pow(tf.subtract(1., y_t), gamma)

fl = tf.multiply(ce * weight)
