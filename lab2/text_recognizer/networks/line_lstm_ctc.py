"""LSTM with CTC for handwritten text recognition within a line."""
from tensorflow.keras.layers import Dense, Input, Reshape, TimeDistributed, Lambda, LSTM
from tensorflow.keras.models import Model as KerasModel
import tensorflow.keras.backend as K

from text_recognizer.networks.lenet import lenet
from text_recognizer.networks.misc import slide_window
from text_recognizer.networks.ctc import ctc_decode


def line_lstm_ctc(input_shape, output_shape, window_width=28, window_stride=14):  # pylint: disable=too-many-locals
    image_height, image_width = input_shape
    output_length, num_classes = output_shape

    num_windows = int((image_width - window_width) / window_stride) + 1
    if num_windows < output_length:
        raise ValueError(f"Window width/stride need to generate >= {output_length} windows (currently {num_windows})")

    image_input = Input(shape=input_shape, name="image")
    y_true = Input(shape=(output_length,), name="y_true")
    input_length = Input(shape=(1,), name="input_length")
    label_length = Input(shape=(1,), name="label_length")

    # Your code should use slide_window and extract image patches from image_input.
    # Pass a convolutional model over each image patch to generate a feature vector per window.
    # Pass these features through one or more LSTM layers.
    # Convert the lstm outputs to softmax outputs.
    # Note that lstms expect a input of shape (num_batch_size, num_timesteps, feature_length).

    # Your code below (Lab 3)
    image_reshaped = Reshape((image_height, image_width, 1))(image_input)
    # (image_height, image_width, 1)

    image_patches = Lambda(slide_window, arguments={"window_width": window_width, "window_stride": window_stride})(
        image_reshaped
    )
    # (num_windows, image_height, window_width, 1)

    # Make a LeNet and get rid of the last two layers (softmax and dropout)
    convnet = lenet((image_height, window_width, 1), (num_classes,))
    convnet = KerasModel(inputs=convnet.inputs, outputs=convnet.layers[-2].output)
    convnet_outputs = TimeDistributed(convnet)(image_patches)
    # (num_windows, 128)

    lstm_output = LSTM(128, return_sequences=True)(convnet_outputs)
    # (num_windows, 128)

    softmax_output = Dense(num_classes, activation="softmax", name="softmax_output")(lstm_output)
    # (num_windows, num_classes)
    # Your code above (Lab 3)

    input_length_processed = Lambda(
        lambda x, num_windows=None: x * num_windows, arguments={"num_windows": num_windows}
    )(input_length)

    ctc_loss_output = Lambda(lambda x: K.ctc_batch_cost(x[0], x[1], x[2], x[3]), name="ctc_loss")(
        [y_true, softmax_output, input_length_processed, label_length]
    )

    ctc_decoded_output = Lambda(lambda x: ctc_decode(x[0], x[1], output_length), name="ctc_decoded")(
        [softmax_output, input_length_processed]
    )

    model = KerasModel(
        inputs=[image_input, y_true, input_length, label_length], outputs=[ctc_loss_output, ctc_decoded_output],
    )
    return model
