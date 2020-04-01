"""Module for creating EMNIST Lines test support files."""
from pathlib import Path
import shutil

import numpy as np

from text_recognizer.datasets import EmnistLinesDataset
import text_recognizer.util as util


SUPPORT_DIRNAME = Path(__file__).parents[0].resolve() / "emnist_lines"


def create_emnist_lines_support_files():
    shutil.rmtree(SUPPORT_DIRNAME, ignore_errors=True)
    SUPPORT_DIRNAME.mkdir()

    dataset = EmnistLinesDataset()
    dataset.load_or_generate_data()

    for ind in [0, 1, 3]:
        image = dataset.x_test[ind]
        print(image.sum(), image.dtype)
        label = "".join(dataset.mapping[label] for label in np.argmax(dataset.y_test[ind], axis=-1).flatten()).strip(
            " _"
        )
        print(label)
        util.write_image(image, str(SUPPORT_DIRNAME / f"{label}.png"))


if __name__ == "__main__":
    create_emnist_lines_support_files()
