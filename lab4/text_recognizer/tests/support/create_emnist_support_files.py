"""Module for creating EMNIST test support files."""
from pathlib import Path
import shutil

import numpy as np

from text_recognizer.datasets import EmnistDataset
import text_recognizer.util as util

SUPPORT_DIRNAME = Path(__file__).parents[0].resolve() / "emnist"


def create_emnist_support_files():
    shutil.rmtree(SUPPORT_DIRNAME, ignore_errors=True)
    SUPPORT_DIRNAME.mkdir()

    dataset = EmnistDataset()
    dataset.load_or_generate_data()

    for ind in [5, 7, 9]:
        image = dataset.x_test[ind]
        label = dataset.mapping[np.argmax(dataset.y_test[ind])]
        print(ind, label)
        util.write_image(image, str(SUPPORT_DIRNAME / f"{label}.png"))


if __name__ == "__main__":
    create_emnist_support_files()
