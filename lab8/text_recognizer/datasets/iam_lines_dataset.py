"""
IamLinesDataset class.

We will use a processed version of this dataset, without including code that did the processing.
We will look at how to generate processed data from raw IAM data in the IamParagraphsDataset.
"""

from boltons.cacheutils import cachedproperty
import h5py
from tensorflow.keras.utils import to_categorical

from text_recognizer import util
from text_recognizer.datasets.dataset import Dataset, _parse_args
from text_recognizer.datasets.emnist_lines_dataset import EmnistLinesDataset


PROCESSED_DATA_DIRNAME = Dataset.data_dirname() / "processed" / "iam_lines"
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / "iam_lines.h5"
PROCESSED_DATA_URL = "https://s3-us-west-2.amazonaws.com/fsdl-public-assets/iam_lines.h5"


class IamLinesDataset(Dataset):
    """
    Note that we use cachedproperty because data takes time to load.

    Parameters
    ----------
    categorical_format
        If True, then y labels are given as one-hot vectors.
    with_start_and_end_tokens
        If True, start and end each sequence with special tokens
    subsample_fraction
        If given, subsample data.
    """

    def __init__(
        self,
        categorical_format: bool = False,
        with_start_and_end_labels: bool = False,
        subsample_fraction: float = None,
    ):
        self.categorical_format = categorical_format
        self.with_start_and_end_labels = with_start_and_end_labels
        self.mapping = EmnistLinesDataset().mapping
        self.inverse_mapping = {v: k for k, v in self.mapping.items()}
        self.num_classes = len(self.mapping)
        self.input_shape = (28, 952)
        self.output_shape = (97, self.num_classes)

        self.subsample_fraction = subsample_fraction
        self.x_train = None
        self.x_test = None
        self.y_train_int = None
        self.y_test_int = None

    def load_or_generate_data(self):
        """Load or generate dataset data."""
        if not PROCESSED_DATA_FILENAME.exists():
            PROCESSED_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
            print("Downloading IAM lines...")
            util.download_url(PROCESSED_DATA_URL, PROCESSED_DATA_FILENAME)
        with h5py.File(PROCESSED_DATA_FILENAME, "r") as f:
            self.x_train = f["x_train"][:]
            self.y_train_int = f["y_train"][:]
            self.x_test = f["x_test"][:]
            self.y_test_int = f["y_test"][:]
        self._subsample()

    def _subsample(self):
        """Only this fraction of data will be loaded."""
        if self.subsample_fraction is None:
            return
        num_train = int(self.x_train.shape[0] * self.subsample_fraction)
        num_test = int(self.x_test.shape[0] * self.subsample_fraction)
        self.x_train = self.x_train[:num_train]
        self.y_train_int = self.y_train_int[:num_train]
        self.x_test = self.x_test[:num_test]
        self.y_test_int = self.y_test_int[:num_test]

    @cachedproperty
    def y_train(self):
        return self.format_y_int(self.y_train_int)

    @cachedproperty
    def y_test(self):
        return self.format_y_int(self.y_test_int)

    def format_y_int(self, y):
        if self.with_start_and_end_labels:
            y = add_start_and_end_labels(y, self.padding_label, self.start_label, self.end_label)
        if self.categorical_format:
            y = to_categorical(y, self.num_classes)
        return y

    def __repr__(self):
        """Print info about the dataset."""
        return (
            "IAM Lines Dataset\n"  # pylint: disable=no-member
            f"Num classes: {self.num_classes}\n"
            f"Mapping: {self.mapping}\n"
            f"Train: {self.x_train.shape} {self.y_train.shape}\n"
            f"Test: {self.x_test.shape} {self.y_test.shape}\n"
        )


def main():
    """Load dataset and print info."""
    args = _parse_args()
    dataset = IamLinesDataset(subsample_fraction=args.subsample_fraction)
    dataset.load_or_generate_data()
    print(dataset)


if __name__ == "__main__":
    main()
