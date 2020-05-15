# Code originally from https://colab.research.google.com/github/tensorflow/examples/blob/master/community/en/transformer_chatbot.ipynb
import tensorflow as tf


def create_padding_mask(x, padding_label=0):
    mask = tf.cast(tf.math.equal(x, padding_label), tf.float32)
    # (batch_size, 1, 1, sequence length)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(x, padding_label=0):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x, padding_label)
    return tf.maximum(look_ahead_mask, padding_mask)
