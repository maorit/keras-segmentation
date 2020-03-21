import tensorflow as tf


def acc_of_clazz0(y_true, y_pred):
    """
    计算第clazz类的accuracy
    """
    y_true = tf.equal(tf.argmax(y_true, axis=-1), 0)
    y_pred = tf.cast(tf.equal(tf.argmax(y_pred, axis=-1), 0), tf.float32)

    acc = tf.boolean_mask(y_pred, y_true)
    return acc


def acc_of_clazz1(y_true, y_pred):
    """
    计算第clazz类的accuracy
    """
    y_true = tf.equal(tf.argmax(y_true, axis=-1), 1)
    y_pred = tf.cast(tf.equal(tf.argmax(y_pred, axis=-1), 1), tf.float32)

    acc = tf.boolean_mask(y_pred, y_true)
    return acc


def acc_of_clazz2(y_true, y_pred):
    """
    计算第clazz类的accuracy
    """
    y_true = tf.equal(tf.argmax(y_true, axis=-1), 2)
    y_pred = tf.cast(tf.equal(tf.argmax(y_pred, axis=-1), 2), tf.float32)

    acc = tf.boolean_mask(y_pred, y_true)
    return acc


def acc_of_clazz3(y_true, y_pred):
    """
    计算第clazz类的accuracy
    """
    y_true = tf.equal(tf.argmax(y_true, axis=-1), 3)
    y_pred = tf.cast(tf.equal(tf.argmax(y_pred, axis=-1), 3), tf.float32)

    acc = tf.boolean_mask(y_pred, y_true)
    return acc


def acc_of_clazz4(y_true, y_pred):
    """
    计算第clazz类的accuracy
    """
    y_true = tf.equal(tf.argmax(y_true, axis=-1), 4)
    y_pred = tf.cast(tf.equal(tf.argmax(y_pred, axis=-1), 4), tf.float32)

    acc = tf.boolean_mask(y_pred, y_true)
    return acc


def acc_of_clazz5(y_true, y_pred):
    """
    计算第clazz类的accuracy
    """
    y_true = tf.equal(tf.argmax(y_true, axis=-1), 5)
    y_pred = tf.cast(tf.equal(tf.argmax(y_pred, axis=-1), 5), tf.float32)

    acc = tf.boolean_mask(y_pred, y_true)
    return acc


def acc_of_clazz6(y_true, y_pred):
    """
    计算第clazz类的accuracy
    """
    y_true = tf.equal(tf.argmax(y_true, axis=-1), 6)
    y_pred = tf.cast(tf.equal(tf.argmax(y_pred, axis=-1), 6), tf.float32)

    acc = tf.boolean_mask(y_pred, y_true)
    return acc


def acc_of_clazz7(y_true, y_pred):
    """
    计算第clazz类的accuracy
    """
    y_true = tf.equal(tf.argmax(y_true, axis=-1), 7)
    y_pred = tf.cast(tf.equal(tf.argmax(y_pred, axis=-1), 7), tf.float32)

    acc = tf.boolean_mask(y_pred, y_true)
    return acc


def acc_of_clazz8(y_true, y_pred):
    """
    计算第clazz类的accuracy
    """
    y_true = tf.equal(tf.argmax(y_true, axis=-1), 8)
    y_pred = tf.cast(tf.equal(tf.argmax(y_pred, axis=-1), 8), tf.float32)

    acc = tf.boolean_mask(y_pred, y_true)
    return acc


def acc_of_clazz9(y_true, y_pred):
    """
    计算第clazz类的accuracy
    """
    y_true = tf.equal(tf.argmax(y_true, axis=-1), 9)
    y_pred = tf.cast(tf.equal(tf.argmax(y_pred, axis=-1), 9), tf.float32)

    acc = tf.boolean_mask(y_pred, y_true)
    return acc


def acc_of_clazz10(y_true, y_pred):
    """
    计算第clazz类的accuracy
    """
    y_true = tf.equal(tf.argmax(y_true, axis=-1), 10)
    y_pred = tf.cast(tf.equal(tf.argmax(y_pred, axis=-1), 10), tf.float32)

    acc = tf.boolean_mask(y_pred, y_true)
    return acc


def acc_of_clazz11(y_true, y_pred):
    """
    计算第clazz类的accuracy
    """
    y_true = tf.equal(tf.argmax(y_true, axis=-1), 11)
    y_pred = tf.cast(tf.equal(tf.argmax(y_pred, axis=-1), 11), tf.float32)

    acc = tf.boolean_mask(y_pred, y_true)
    return acc


def acc_of_clazz12(y_true, y_pred):
    """
    计算第clazz类的accuracy
    """
    y_true = tf.equal(tf.argmax(y_true, axis=-1), 12)
    y_pred = tf.cast(tf.equal(tf.argmax(y_pred, axis=-1), 12), tf.float32)

    acc = tf.boolean_mask(y_pred, y_true)
    return acc


def acc_of_clazz13(y_true, y_pred):
    """
    计算第clazz类的accuracy
    """
    y_true = tf.equal(tf.argmax(y_true, axis=-1), 13)
    y_pred = tf.cast(tf.equal(tf.argmax(y_pred, axis=-1), 13), tf.float32)

    acc = tf.boolean_mask(y_pred, y_true)
    return acc


def acc_of_clazz14(y_true, y_pred):
    """
    计算第clazz类的accuracy
    """
    y_true = tf.equal(tf.argmax(y_true, axis=-1), 14)
    y_pred = tf.cast(tf.equal(tf.argmax(y_pred, axis=-1), 14), tf.float32)

    acc = tf.boolean_mask(y_pred, y_true)
    return acc


def acc_of_clazz15(y_true, y_pred):
    """
    计算第clazz类的accuracy
    """
    y_true = tf.equal(tf.argmax(y_true, axis=-1), 15)
    y_pred = tf.cast(tf.equal(tf.argmax(y_pred, axis=-1), 15), tf.float32)

    acc = tf.boolean_mask(y_pred, y_true)
    return acc


def acc_of_clazz16(y_true, y_pred):
    """
    计算第clazz类的accuracy
    """
    y_true = tf.equal(tf.argmax(y_true, axis=-1), 16)
    y_pred = tf.cast(tf.equal(tf.argmax(y_pred, axis=-1), 16), tf.float32)

    acc = tf.boolean_mask(y_pred, y_true)
    return acc


def acc_of_clazz17(y_true, y_pred):
    """
    计算第clazz类的accuracy
    """
    y_true = tf.equal(tf.argmax(y_true, axis=-1), 17)
    y_pred = tf.cast(tf.equal(tf.argmax(y_pred, axis=-1), 17), tf.float32)

    acc = tf.boolean_mask(y_pred, y_true)
    return acc


def acc_of_clazz18(y_true, y_pred):
    """
    计算第clazz类的accuracy
    """
    y_true = tf.equal(tf.argmax(y_true, axis=-1), 18)
    y_pred = tf.cast(tf.equal(tf.argmax(y_pred, axis=-1), 18), tf.float32)

    acc = tf.boolean_mask(y_pred, y_true)
    return acc


def acc_of_clazz19(y_true, y_pred):
    """
    计算第clazz类的accuracy
    """
    y_true = tf.equal(tf.argmax(y_true, axis=-1), 19)
    y_pred = tf.cast(tf.equal(tf.argmax(y_pred, axis=-1), 19), tf.float32)

    acc = tf.boolean_mask(y_pred, y_true)
    return acc


def acc_of_clazz20(y_true, y_pred):
    """
    计算第clazz类的accuracy
    """
    y_true = tf.equal(tf.argmax(y_true, axis=-1), 20)
    y_pred = tf.cast(tf.equal(tf.argmax(y_pred, axis=-1), 20), tf.float32)

    acc = tf.boolean_mask(y_pred, y_true)
    return acc
