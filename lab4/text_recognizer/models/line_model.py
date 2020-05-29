"""Define LineModel class."""
from typing import Callable, Dict, Tuple

import editdistance
import numpy as np

from text_recognizer.datasets.emnist_lines_dataset import EmnistLinesDataset
from text_recognizer.datasets.dataset_sequence import DatasetSequence
from text_recognizer.models.base import Model
from text_recognizer.networks import line_cnn_all_conv


class LineModel(Model):
    """Model for predicting a string from an image of a handwritten line of text."""

    def __init__(
        self,
        dataset_cls: type = EmnistLinesDataset,
        network_fn: Callable = line_cnn_all_conv,
        dataset_args: Dict = None,
        network_args: Dict = None,
    ):
        """Define the default dataset and network values for this model."""
        super().__init__(dataset_cls, network_fn, dataset_args, network_args)

    def evaluate(self, x, y, batch_size=16, verbose=True):
        """Evaluate model."""
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
        """Predict on a single input."""
        if image.dtype == np.uint8:
            image = (image / 255).astype(np.float32)
        pred_raw = self.network.predict(np.expand_dims(image, 0), batch_size=1).squeeze()
        pred = "".join(self.data.mapping[label] for label in np.argmax(pred_raw, axis=-1).flatten()).strip()
        conf = np.min(np.max(pred_raw, axis=-1))  # The least confident of the predictions.
        return pred, conf
