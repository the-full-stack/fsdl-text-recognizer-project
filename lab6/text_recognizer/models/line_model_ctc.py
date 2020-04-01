"""Define LineModelCtc class and associated functions."""
from typing import Callable, Dict, Tuple

import editdistance
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model as KerasModel

from text_recognizer.datasets.dataset_sequence import DatasetSequence
from text_recognizer.datasets import EmnistLinesDataset
from text_recognizer.models.base import Model
from text_recognizer.networks.line_lstm_ctc import line_lstm_ctc


class LineModelCtc(Model):
    """Model for recognizing handwritten text in an image of a line, using CTC loss/decoding."""

    def __init__(
        self,
        dataset_cls: type = EmnistLinesDataset,
        network_fn: Callable = line_lstm_ctc,
        dataset_args: Dict = None,
        network_args: Dict = None,
    ):
        """Define the default dataset and network values for this model."""
        default_dataset_args: dict = {}
        if dataset_args is None:
            dataset_args = {}
        dataset_args = {**default_dataset_args, **dataset_args}

        default_network_args = {"window_width": 12, "window_stride": 5}
        if network_args is None:
            network_args = {}
        network_args = {**default_network_args, **network_args}
        super().__init__(dataset_cls, network_fn, dataset_args, network_args)
        self.batch_format_fn = format_batch_ctc

    def loss(self):
        """Simply pass through the loss that we computed in the network."""
        return {"ctc_loss": lambda y_true, y_pred: y_pred}

    def metrics(self):
        """
        Compute no metrics.

        TODO: We could probably pass in a custom character accuracy metric for 'ctc_decoded' output here.
        """
        return None

    def evaluate(self, x, y, batch_size: int = 16, verbose: bool = True) -> float:
        """Evaluate model."""
        test_sequence = DatasetSequence(x, y, batch_size, format_fn=self.batch_format_fn)

        # We can use the `ctc_decoded` layer that is part of our model here.
        decoding_model = KerasModel(inputs=self.network.input, outputs=self.network.get_layer("ctc_decoded").output)
        preds = decoding_model.predict(test_sequence)

        trues = np.argmax(y, -1)
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
        softmax_output_fn = KerasModel(
            inputs=[self.network.get_layer("image").input], outputs=[self.network.get_layer("softmax_output").output],
        )
        if image.dtype == np.uint8:
            image = (image / 255).astype(np.float32)

        # Get the prediction and confidence using softmax_output_fn, passing the right input into it.
        input_image = np.expand_dims(image, 0)
        softmax_output = softmax_output_fn.predict(input_image)

        input_length = [softmax_output.shape[1]]
        decoded, log_prob = K.ctc_decode(softmax_output, input_length, greedy=True)

        pred_raw = K.eval(decoded[0])[0]
        pred = "".join(self.data.mapping[label] for label in pred_raw).strip()

        neg_sum_logit = K.eval(log_prob)[0][0]
        conf = np.exp(-neg_sum_logit)
        # Your code above (Lab 3)

        return pred, conf


def format_batch_ctc(batch_x, batch_y):
    """
    Because CTC loss needs to be computed inside of the network, we include information about outputs in the inputs.
    """
    batch_size = batch_y.shape[0]
    y_true = np.argmax(batch_y, axis=-1)

    label_lengths = []
    for ind in range(batch_size):
        # Find all of the indices in the label that are blank
        empty_at = np.where(batch_y[ind, :, -1] == 1)[0]
        # Length of the label is the pos of the first blank, or the max length
        if empty_at.shape[0] > 0:
            label_lengths.append(empty_at[0])
        else:
            label_lengths.append(batch_y.shape[1])

    batch_inputs = {
        "image": batch_x,
        "y_true": y_true,
        "input_length": np.ones((batch_size, 1)),  # dummy, will be set to num_windows in network
        "label_length": np.array(label_lengths),
    }
    batch_outputs = {"ctc_loss": np.zeros(batch_size), "ctc_decoded": y_true}  # dummy
    return batch_inputs, batch_outputs
