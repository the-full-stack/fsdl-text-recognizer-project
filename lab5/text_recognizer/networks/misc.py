"""Misc neural network functionality."""
import numpy as np
import tensorflow as tf


def slide_window(image: np.ndarray, window_width: int, window_stride: int) -> np.ndarray:
    """
    Parameters
    ----------
    image
        (image_height, image_width, 1) input

    Returns
    -------
    np.ndarray
        (num_windows, image_height, window_width, 1) output, where
        num_windows is floor((image_width - window_width) / window_stride) + 1
    """
    kernel = [1, 1, window_width, 1]
    strides = [1, 1, window_stride, 1]
    patches = tf.image.extract_patches(image, kernel, strides, [1, 1, 1, 1], "VALID")
    patches = tf.transpose(patches, (0, 2, 1, 3))
    patches = tf.expand_dims(patches, -1)
    return patches
