"""Transformer Keras model."""
# Code originally from https://colab.research.google.com/github/tensorflow/examples/blob/master/community/en/transformer_chatbot.ipynb
from typing import Tuple

import tensorflow as tf

from text_recognizer.networks.transformer.decoder import decoder
from text_recognizer.networks.transformer.masking import create_look_ahead_mask
from text_recognizer.networks.transformer.positional_encoding import PositionalEncoding
from text_recognizer.networks.line_cnn_all_conv import line_cnn_for_transformer


def run_transformer_inference(
    model: tf.keras.models.Model, image: tf.Tensor, max_length: int, start_label: int, end_label: int
) -> tf.Tensor:
    output = tf.expand_dims([start_label], 0)

    for i in range(max_length):
        predictions = model(inputs=[image, output], training=False)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if tf.equal(predicted_id, end_label):
            break

        # concatenated the predicted_id to the output which is given to the decoder as input
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def transformer(
    image_shape: Tuple[int, ...],
    window_width: int,
    window_stride: int,
    vocab_size: int,
    num_layers: int,
    units: int,
    d_model: int,
    d_enc_outputs: int,
    num_heads: int,
    dropout: float,
    padding_label: int = 0,
) -> tf.keras.Model:
    image_inputs = tf.keras.Input(shape=image_shape, name="image_inputs")
    line_cnn_outputs = line_cnn_for_transformer(image_shape, window_width, window_stride)(image_inputs)

    # TODO: line_cnn_outputs *= tf.math.sqrt(tf.cast(d_enc_outputs, tf.float32)) ?
    position_encoded_line_cnn_outputs = PositionalEncoding(d_enc_outputs)(line_cnn_outputs)

    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

    # mask the future tokens for decoder inputs at the 1st attention block
    look_ahead_mask = tf.keras.layers.Lambda(
        lambda x: create_look_ahead_mask(x, padding_label=padding_label),
        output_shape=(1, None, None),
        name="look_ahead_mask",
    )(dec_inputs)

    dec_outputs = decoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        d_enc_outputs=d_enc_outputs,
        num_heads=num_heads,
        dropout=dropout,
    )(inputs=[dec_inputs, position_encoded_line_cnn_outputs, look_ahead_mask])

    outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

    return tf.keras.Model(inputs=[image_inputs, dec_inputs], outputs=outputs)
