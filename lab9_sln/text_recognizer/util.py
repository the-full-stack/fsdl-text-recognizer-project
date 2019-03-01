"""Utility functions for text_recognizer module."""
from pathlib import Path
from typing import Union
from urllib.request import urlopen
import base64
import os

import numpy as np
import cv2


def read_image(image_uri: Union[Path, str], grayscale=False) -> np.array:
    """Read image_uri."""
    def read_image_from_filename(image_filename, imread_flag):
        return cv2.imread(str(image_filename), imread_flag)

    def read_image_from_url(image_url, imread_flag):
        url_response = urlopen(str(image_url))  # nosec
        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
        return cv2.imdecode(img_array, imread_flag)

    imread_flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    local_file = os.path.exists(image_uri)
    try:
        img = None
        if local_file:
            img = read_image_from_filename(image_uri, imread_flag)
        else:
            img = read_image_from_url(image_uri, imread_flag)
        assert img is not None
    except Exception as e:
        raise ValueError("Could not load image at {}: {}".format(image_uri, e))
    return img


def read_b64_image(b64_string, grayscale=False):
    """Load base64-encoded images."""
    imread_flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    try:
        _, b64_data = b64_string.split(',')
        return cv2.imdecode(np.frombuffer(base64.b64decode(b64_data), np.uint8), imread_flag)
    except Exception as e:
        raise ValueError("Could not load image from b64 {}: {}".format(b64_string, e))


def write_image(image: np.ndarray, filename: Union[Path, str]) -> None:
    cv2.imwrite(str(filename), image)

