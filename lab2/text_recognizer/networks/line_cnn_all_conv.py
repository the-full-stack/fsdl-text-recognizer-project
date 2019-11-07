"""CNN-based model for recognizing handwritten text."""
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Reshape, Lambda
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model as KerasModel


def line_cnn_all_conv(
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        window_width: float = 16,
        window_stride: float = 8) -> KerasModel:
    image_height, image_width = input_shape
    output_length, num_classes = output_shape

    model = Sequential()
    model.add(Reshape((image_height, image_width, 1), input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # So far, this is the same as LeNet. At this point, LeNet would flatten and Dense 128.
    # Instead, we are going to use a Conv2D to slide over these outputs with window_width and window_stride,
    # and output softmax activations of shape (output_length, num_classes)./
    # In your calculation of the necessary filter size,
    # remember that padding is set to 'valid' (by default) in the network above.

    # Your code below (Lab 2)
    # (image_height // 2 - 2, image_width // 2 - 2, 64)
    new_height = image_height // 2 - 2
    new_width = image_width // 2 - 2
    new_window_width = window_width // 2 - 2
    new_window_stride = window_stride // 2
    model.add(Conv2D(128, (new_height, new_window_width), (1, new_window_stride), activation='relu'))
    model.add(Dropout(0.2))
    # (1, num_windows, 128)

    num_windows = int((new_width - new_window_width) / new_window_stride) + 1

    model.add(Reshape((num_windows, 128, 1)))
    # Your code above (Lab 2)
    # (num_windows, 128, 1)

    width = int(num_windows / output_length)
    model.add(Conv2D(num_classes, (width, 128), (width, 1), activation='softmax'))
    # (image_width / width, 1, num_classes)

    model.add(Lambda(lambda x: tf.squeeze(x, 2)))
    # (max_length, num_classes)

    # Since we floor'd the calculation of width, we might have too many items in the sequence. Take only output_length.
    model.add(Lambda(lambda x: x[:, :output_length, :]))

    return model
