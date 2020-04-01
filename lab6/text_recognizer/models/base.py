"""Model class, to be extended by specific types of models."""
# pylint: disable=missing-function-docstring
from pathlib import Path
from typing import Callable, Dict, Optional

from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.optimizers import RMSprop
import numpy as np

from text_recognizer.datasets.dataset_sequence import DatasetSequence


DIRNAME = Path(__file__).parents[1].resolve() / "weights"


class Model:
    """Base class, to be subclassed by predictors for specific type of data."""

    def __init__(
        self,
        dataset_cls: type,
        network_fn: Callable[..., KerasModel],
        dataset_args: Dict = None,
        network_args: Dict = None,
    ):
        self.name = f"{self.__class__.__name__}_{dataset_cls.__name__}_{network_fn.__name__}"

        if dataset_args is None:
            dataset_args = {}
        self.data = dataset_cls(**dataset_args)

        if network_args is None:
            network_args = {}
        self.network = network_fn(self.data.input_shape, self.data.output_shape, **network_args)
        self.network.summary()

        self.batch_augment_fn: Optional[Callable] = None
        self.batch_format_fn: Optional[Callable] = None

    @property
    def image_shape(self):
        return self.data.input_shape

    @property
    def weights_filename(self) -> str:
        DIRNAME.mkdir(parents=True, exist_ok=True)
        return str(DIRNAME / f"{self.name}_weights.h5")

    def fit(
        self, dataset, batch_size: int = 32, epochs: int = 10, augment_val: bool = True, callbacks: list = None,
    ):
        if callbacks is None:
            callbacks = []

        self.network.compile(loss=self.loss(), optimizer=self.optimizer(), metrics=self.metrics())

        train_sequence = DatasetSequence(
            dataset.x_train,
            dataset.y_train,
            batch_size,
            augment_fn=self.batch_augment_fn,
            format_fn=self.batch_format_fn,
        )
        test_sequence = DatasetSequence(
            dataset.x_test,
            dataset.y_test,
            batch_size,
            augment_fn=self.batch_augment_fn if augment_val else None,
            format_fn=self.batch_format_fn,
        )

        self.network.fit(
            train_sequence,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=test_sequence,
            use_multiprocessing=False,
            workers=1,
            shuffle=True,
        )

    def evaluate(self, x: np.ndarray, y: np.ndarray, batch_size: int = 16, _verbose: bool = False):
        # pylint: disable=unused-argument
        sequence = DatasetSequence(x, y, batch_size=batch_size)  # Use a small batch size to use less memory
        preds = self.network.predict(sequence)
        return np.mean(np.argmax(preds, -1) == np.argmax(y, -1))

    def loss(self):  # pylint: disable=no-self-use
        return "categorical_crossentropy"

    def optimizer(self):  # pylint: disable=no-self-use
        return RMSprop()

    def metrics(self):  # pylint: disable=no-self-use
        return ["accuracy"]

    def load_weights(self):
        self.network.load_weights(self.weights_filename)

    def save_weights(self):
        self.network.save_weights(self.weights_filename)
