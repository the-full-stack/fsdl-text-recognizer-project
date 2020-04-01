"""Define ctc_decode function."""
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.ops import ctc_ops, sparse_ops  # pylint: disable=no-name-in-module


def ctc_decode(y_pred, input_length, max_output_length):
    """
    Cut down from https://github.com/keras-team/keras/blob/master/keras/backend/tensorflow_backend.py#L4170

    Decodes the output of a softmax.
    Uses greedy (best path) search.

    # Arguments
        y_pred: tensor `(samples, time_steps, num_categories)`
            containing the prediction, or output of the softmax.
        input_length: tensor `(samples, )` containing the sequence length for
            each batch item in `y_pred`.
        max_output_length: int giving the max output sequence length

    # Returns
        List: list of one element that contains the decoded sequence.
    """
    y_pred = tf.math.log(tf.transpose(y_pred, perm=[1, 0, 2]) + K.epsilon())
    input_length = tf.cast((tf.squeeze(input_length, axis=-1)), tf.int32)

    (decoded, _) = ctc_ops.ctc_greedy_decoder(inputs=y_pred, sequence_length=input_length)

    sparse = decoded[0]
    decoded_dense = sparse_ops.sparse_to_dense(sparse.indices, sparse.dense_shape, sparse.values, default_value=-1)

    # Unfortunately, decoded_dense will be of different number of columns, depending on the decodings.
    # For use in `predict()`, we need to get it all in one standard shape, so let's pad if necessary.
    max_length = max_output_length + 2  # giving 2 extra characters for CTC leeway
    cols = tf.shape(decoded_dense)[-1]

    def pad():
        return tf.pad(decoded_dense, [[0, 0], [0, max_length - cols]], constant_values=-1)

    def noop():
        return decoded_dense

    return tf.cond(tf.less(cols, max_length), pad, noop)
