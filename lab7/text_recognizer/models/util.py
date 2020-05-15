from tensorflow.python.keras import backend as K
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import tensorflow as tf


def sparse_categorical_crossentropy_ignoring_padding(
    y_true: tf.Tensor, y_pred: tf.Tensor, padding_label: int
) -> tf.Tensor:
    """Sparse categorical crossentropy that ignores padding."""
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, padding_label), tf.float32)
    loss = tf.multiply(loss, mask)
    return tf.reduce_mean(loss)


def sparse_categorical_accuracy_ignoring_padding(y_true, y_pred, padding_label):
    """Sparse categorical accuracy that ignores padding in y_true."""
    # From https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/python/keras/metrics.py#L2974-L2989
    y_pred_rank = ops.convert_to_tensor(y_pred).shape.ndims
    y_true_rank = ops.convert_to_tensor(y_true).shape.ndims

    # If the shape of y_true is (num_samples, 1), squeeze to (num_samples,)
    if (
        (y_true_rank is not None)
        and (y_pred_rank is not None)
        and (len(K.int_shape(y_true)) == len(K.int_shape(y_pred)))
    ):
        y_true = array_ops.squeeze(y_true, [-1])

    y_pred = math_ops.argmax(y_pred, axis=-1)

    # If the predicted output and actual output types don't match, force cast them to match.
    if K.dtype(y_pred) != K.dtype(y_true):
        y_pred = math_ops.cast(y_pred, K.dtype(y_true))

    true_mask = tf.cast(tf.not_equal(y_true, padding_label), K.dtype(y_true))
    y_true_masked = tf.multiply(y_true, true_mask)
    y_pred_masked = tf.multiply(y_pred, true_mask)

    return math_ops.cast(math_ops.equal(y_true_masked, y_pred_masked), K.floatx())


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate scheduler from the Attention is all you need paper."""

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
