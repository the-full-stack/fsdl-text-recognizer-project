"""LeNet network."""
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D
from tensorflow.keras.models import Sequential, Model


def lenet(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> Model:
    """Return LeNet Keras model."""
    num_classes = output_shape[0]

    # Your code below (Lab 2)
    model = Sequential()
    if len(input_shape) < 3:
        model.add(Lambda(lambda x: tf.expand_dims(x, -1), input_shape=input_shape, name='expand_dims'))
        input_shape = (input_shape[0], input_shape[1], 1)
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape, padding="valid"))
    model.add(Conv2D(64, (3, 3), activation="relu", padding="valid"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="valid"))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation="softmax"))
    # Your code above (Lab 2)

    return model
