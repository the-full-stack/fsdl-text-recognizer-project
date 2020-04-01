"""Define LineDetectorModel class."""
from typing import Callable, Dict, Tuple

import numpy as np

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from text_recognizer.datasets.iam_paragraphs_dataset import IamParagraphsDataset
from text_recognizer.models.base import Model
from text_recognizer.networks import fcn


_DATA_AUGMENTATION_PARAMS = {
    "width_shift_range": 0.06,
    "height_shift_range": 0.1,
    "horizontal_flip": True,
    "zoom_range": 0.1,
    "fill_mode": "constant",
    "cval": 0,
    "shear_range": 3,
}


class LineDetectorModel(Model):
    """Model to detect lines of text in an image."""

    def __init__(
        self,
        dataset_cls: type = IamParagraphsDataset,
        network_fn: Callable = fcn,
        dataset_args: Dict = None,
        network_args: Dict = None,
    ):
        """Define the default dataset and network values for this model."""
        super().__init__(dataset_cls, network_fn, dataset_args, network_args)

        self.data_augmentor = ImageDataGenerator(**_DATA_AUGMENTATION_PARAMS)
        self.batch_augment_fn = self.augment_batch

    def loss(self):  # pylint: disable=no-self-use
        return "categorical_crossentropy"

    def optimizer(self):  # pylint: disable=no-self-use
        return Adam(0.001 / 2)

    def metrics(self):  # pylint: disable=no-self-use
        return None

    def augment_batch(self, x_batch: np.ndarray, y_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform different random transformations on the whole batch of x, y samples."""
        x_augment, y_augment = zip(*[self._augment_sample(x, y) for x, y in zip(x_batch, y_batch)])
        return np.stack(x_augment, axis=0), np.stack(y_augment, axis=0)

    def _augment_sample(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform the same random image transformation on both x and y.
        x is a 2d image of shape self.image_shape, but self.data_augmentor needs the channel image too.
        """
        x_3d = np.expand_dims(x, axis=-1)
        transform_parameters = self.data_augmentor.get_random_transform(x_3d.shape)
        x_augment = self.data_augmentor.apply_transform(x_3d, transform_parameters)
        y_augment = self.data_augmentor.apply_transform(y, transform_parameters)
        return np.squeeze(x_augment, axis=-1), y_augment

    def predict_on_image(self, x: np.ndarray) -> np.ndarray:
        """Predict on a single input."""
        return self.network.predict(np.expand_dims(x, axis=0))[0]

    def evaluate(self, x: np.ndarray, y: np.ndarray, batch_size: int = 32, verbose: bool = False) -> float:
        """Evaluate the model."""
        # pylint: disable=unused-argument
        return self.network.evaluate(x, y, batch_size=batch_size)
