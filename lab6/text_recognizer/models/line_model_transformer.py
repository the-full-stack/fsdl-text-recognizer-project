"""Define LineModelTransformer class."""
from functools import partial
from typing import Any, Callable, Dict, Tuple

import editdistance
import numpy as np
import tensorflow as tf

from text_recognizer.datasets.emnist_lines_dataset import EmnistLinesDataset
from text_recognizer.datasets.dataset_sequence import DatasetSequence
from text_recognizer.models.base import Model
from text_recognizer.models.util import (
    CustomSchedule,
    sparse_categorical_accuracy_ignoring_padding,
    sparse_categorical_crossentropy_ignoring_padding,
)
from text_recognizer.networks import line_cnn_for_transformer
from text_recognizer.networks.transformer import run_transformer_inference


class LineModelTransformer(Model):
    """Model for predicting a string from an image of a handwritten line of text using the Transformer architecture."""

    def __init__(
        self,
        dataset_cls: type = EmnistLinesDataset,
        network_fn: Callable = line_cnn_for_transformer,
        dataset_args: Dict[str, Any] = None,
        network_args: Dict[str, Any] = None,
    ):
        """Define the default dataset and network values for this model."""
        if dataset_args is None:
            dataset_args = {}
        dataset_args["with_start_and_end_labels"] = True
        super().__init__(dataset_cls, network_fn, dataset_args, network_args)

    def loss(self):  # pylint: disable=no-self-use
        """Compute sparse categorical crossentropy loss on integer labels, ignoring padding."""
        return partial(sparse_categorical_crossentropy_ignoring_padding, padding_label=self.data.padding_label)

    def optimizer(self):  # pylint: disable=no-self-use
        learning_rate = CustomSchedule(d_model=32)
        return tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    def metrics(self):  # pylint: disable=no-self-use
        """Compute sparse categorical accuracy, ignoring padding in the true label."""
        return [partial(sparse_categorical_accuracy_ignoring_padding, padding_label=self.data.padding_label)]

    def evaluate(self, x, y, batch_size=16, verbose=True):
        """Evaluate model."""
        # TODO
        sequence = DatasetSequence(x, y)
        preds_raw = self.network.predict(sequence)
        trues = np.argmax(y, -1)
        preds = np.argmax(preds_raw, -1)
        pred_strings = ["".join(self.data.mapping.get(label, "") for label in pred).strip(" |_") for pred in preds]
        true_strings = ["".join(self.data.mapping.get(label, "") for label in true).strip(" |_") for true in trues]
        char_accuracies = [
            1 - editdistance.eval(true_string, pred_string) / len(true_string)
            for pred_string, true_string in zip(pred_strings, true_strings)
        ]
        if verbose:
            sorted_ind = np.argsort(char_accuracies)
            print("\nLeast accurate predictions:")
            for ind in sorted_ind[:5]:
                print(f"True: {true_strings[ind]}")
                print(f"Pred: {pred_strings[ind]}")
            print("\nMost accurate predictions:")
            for ind in sorted_ind[-5:]:
                print(f"True: {true_strings[ind]}")
                print(f"Pred: {pred_strings[ind]}")
            print("\nRandom predictions:")
            random_ind = np.random.randint(0, len(char_accuracies), 5)
            for ind in random_ind:  # pylint: disable=not-an-iterable
                print(f"True: {true_strings[ind]}")
                print(f"Pred: {pred_strings[ind]}")
        mean_accuracy = np.mean(char_accuracies)
        return mean_accuracy

    def predict_on_image(self, image: np.ndarray) -> Tuple[str, float]:
        """Predict on a single image."""
        if image.dtype == np.uint8:
            image = (image / 255).astype(np.float32)
        pred = run_inference(self.model, image, self.data.max_length, self.data.start_label, self.data.end_label)
        string = "".join([dataset.mapping[ind] for ind in pred.numpy()]).strip()
        return pred, 1.0  # NOTE: conference is always given as 1.0 for now...
