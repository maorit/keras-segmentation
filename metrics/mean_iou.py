import tensorflow as tf

def mean_iou(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    return tf.metrics.mean_iou


#
# def mean_iou(n_classes):
#     def _iou(y_true, y_pred, clazz):
#         """
#         计算第clazz类的iou
#         """
#         y_true = tf.cast(tf.equal(tf.argmax(y_true, axis=-1), clazz), tf.float32)
#         y_pred = tf.cast(tf.equal(tf.argmax(y_pred, axis=-1), clazz), tf.float32)
#         intersection = tf.reduce_sum(y_true * y_pred)
#         union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
#         return intersection / (union + 1e-10)
#
#     def _mean_iou(y_true, y_pred):
#         """
#         计算平均iou
#         """
#         mean_iou = 0
#         for clazz in range(n_classes):
#             mean_iou = mean_iou + _iou(y_true, y_pred, clazz)
#         return mean_iou / n_classes
#
#     return _mean_iou
