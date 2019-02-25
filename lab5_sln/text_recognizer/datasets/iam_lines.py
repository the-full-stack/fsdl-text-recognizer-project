"""IamLinesDataset class."""
import pathlib
from urllib.request import urlretrieve

from boltons.cacheutils import cachedproperty
import h5py
from tensorflow.keras.utils import to_categorical

from text_recognizer.datasets.base import Dataset
from text_recognizer.datasets.emnist import EmnistDataset


DATA_DIRNAME = pathlib.Path(__file__).parents[2].resolve() / 'data'
PROCESSED_DATA_DIRNAME = DATA_DIRNAME / 'processed' / 'iam_lines'
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / 'iam_lines.h5'
PROCESSED_DATA_URL = 'https://s3-us-west-2.amazonaws.com/fsdl-public-assets/iam_lines.h5'


class IamLinesDataset(Dataset):
    """
    "The IAM Lines dataset, first published at the ICDAR 1999, contains forms of unconstrained handwritten text,
    which were scanned at a resolution of 300dpi and saved as PNG images with 256 gray levels.
    From http://www.fki.inf.unibe.ch/databases/iam-handwriting-database

    The data split we will use is
    IAM lines Large Writer Independent Text Line Recognition Task (lwitlrt): 9,862 text lines.
        The validation set has been merged into the train set.
        The train set has 7,101 lines from 326 writers.
        The test set has 1,861 lines from 128 writers.
        The text lines of all data sets are mutually exclusive, thus each writer has contributed to one set only.

    Note that we use cachedproperty because data takes time to load.
    """
    def __init__(self):
        self.mapping = EmnistDataset().mapping
        self.inverse_mapping = {v: k for k, v in self.mapping.items()}
        self.num_classes = len(self.mapping)
        self.input_shape = (28, 952)
        self.output_shape = (97, self.num_classes)
        self.x_train = None
        self.x_test = None
        self.y_train_int = None
        self.y_test_int = None

    def load_or_generate_data(self):
        """Load or generate dataset data."""
        if not PROCESSED_DATA_FILENAME.exists():
            PROCESSED_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
            print('Downloading IAM lines...')
            urlretrieve(PROCESSED_DATA_URL, PROCESSED_DATA_FILENAME)  # nosec
        with h5py.File(PROCESSED_DATA_FILENAME, 'r') as f:
            self.x_train = f['x_train'][:]
            self.y_train_int = f['y_train'][:]
            self.x_test = f['x_test'][:]
            self.y_test_int = f['y_test'][:]

    @cachedproperty
    def y_train(self):
        """Return y_train"""
        return to_categorical(self.y_train_int, self.num_classes)

    @cachedproperty
    def y_test(self):
        """Return y_test"""
        return to_categorical(self.y_test_int, self.num_classes)

    def __repr__(self):
        """Print info about the dataset."""
        return (
            'IAM Lines Dataset\n'  # pylint: disable=no-member
            f'Num classes: {self.num_classes}\n'
            f'Mapping: {self.mapping}\n'
            f'Train: {self.x_train.shape} {self.y_train.shape}\n'
            f'Test: {self.x_test.shape} {self.y_test.shape}\n'
        )


def main():
    """Load dataset and print info."""
    dataset = IamLinesDataset()
    dataset.load_or_generate_data()
    print(dataset)


if __name__ == '__main__':
    main()

