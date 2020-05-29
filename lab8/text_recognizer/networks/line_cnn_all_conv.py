"""CNN-based model for recognizing handwritten text."""
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Reshape, Lambda, Permute
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model as KerasModel


def line_cnn_all_conv(
    input_shape: Tuple[int, ...], output_shape: Tuple[int, ...], window_width: float = 28, window_stride: float = 14,
) -> KerasModel:
    image_height, image_width = input_shape
    output_length, num_classes = output_shape
    # Current shape is: (image_height, image_width, 1)

    model = Sequential()
    model.add(Reshape((image_height, image_width, 1), input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Dropout(0.2))
    # Current shape is: (image_height // 2, image_width // 2, 64)

    # So far, this is the same as LeNet. At this point, LeNet would flatten and Dense 128.
    # Instead, we are going to use a Conv2D to slide over these outputs with window_width and window_stride,
    # and output softmax activations of shape (output_length, num_classes)./

    # Because of MaxPooling, everything is divided by 2
    new_height = image_height // 2
    new_width = image_width // 2
    new_window_width = window_width // 2
    new_window_stride = window_stride // 2

    # Your code below (Lab 2)
    model.add(
        Conv2D(128, (new_height, new_window_width), (new_height, new_window_stride), activation="relu", padding="same")
    )
    model.add(Dropout(0.2))
    # Your code above (Lab 2)
    # Shape is now (1, num_windows, 128)
    num_windows = new_width // new_window_stride

    model.add(Permute((2, 1, 3)))  # We could instead do a Reshape((num_windows, 1, 128))
    # Shape is now (num_windows, 1, 128)

    final_classifier_width = num_windows // output_length
    model.add(
        Conv2D(
            num_classes, (final_classifier_width, 1), (final_classifier_width, 1), activation="softmax", padding="same"
        )
    )
    # Shape is now (output_length, 1, num_classes)

    model.add(Lambda(lambda x: tf.squeeze(x, 2)))  # We could instead do a Reshape((output_length, num_classes))
    # Shape is now (output_length, num_classes)

    # Since we floor'd the calculation of width, we might have too many items in the sequence. Take only output_length.
    # model.add(Lambda(lambda x: x[:, :output_length, :]))
    return model
