# Code originally from https://colab.research.google.com/github/tensorflow/examples/blob/master/community/en/transformer_chatbot.ipynb
import tensorflow as tf

from text_recognizer.networks.transformer.positional_encoding import PositionalEncoding
from text_recognizer.networks.transformer.attention import MultiHeadAttention


def decoder_layer(units, d_model, d_enc_outputs, num_heads, dropout, name="decoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_enc_outputs), name="encoder_outputs")
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name="look_ahead_mask")

    attention1 = MultiHeadAttention(d_model, num_heads, name="attention_1")(
        inputs={"query": inputs, "key": inputs, "value": inputs, "mask": look_ahead_mask}
    )
    attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention1 + inputs)

    attention2 = MultiHeadAttention(d_model, num_heads, name="attention_2")(
        inputs={"query": attention1, "key": enc_outputs, "value": enc_outputs}
    )
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention2 + attention1)

    outputs = tf.keras.layers.Dense(units=units, activation="relu")(attention2)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs + attention2)

    return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask], outputs=outputs, name=name)


def decoder(vocab_size, num_layers, units, d_model, d_enc_outputs, num_heads, dropout, name="decoder"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_enc_outputs), name="encoder_outputs")
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name="look_ahead_mask")

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = decoder_layer(
            units=units,
            d_model=d_model,
            d_enc_outputs=d_enc_outputs,
            num_heads=num_heads,
            dropout=dropout,
            name="decoder_layer_{}".format(i),
        )(inputs=[outputs, enc_outputs, look_ahead_mask])
    return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask], outputs=outputs, name=name)
