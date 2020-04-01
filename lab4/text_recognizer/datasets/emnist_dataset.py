"""
EMNIST dataset. Downloads from NIST website and saves as .npz file if not already present.
"""
import json
import os
from pathlib import Path
import shutil
import zipfile

from boltons.cacheutils import cachedproperty
from tensorflow.keras.utils import to_categorical
import h5py
import numpy as np
import toml

from text_recognizer.datasets.dataset import _download_raw_dataset, Dataset, _parse_args

SAMPLE_TO_BALANCE = True  # If true, take at most the mean number of instances per class.

RAW_DATA_DIRNAME = Dataset.data_dirname() / "raw" / "emnist"
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml"

PROCESSED_DATA_DIRNAME = Dataset.data_dirname() / "processed" / "emnist"
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / "byclass.h5"

ESSENTIALS_FILENAME = Path(__file__).parents[0].resolve() / "emnist_essentials.json"


class EmnistDataset(Dataset):
    """
    "The EMNIST dataset is a set of handwritten character digits derived from the NIST Special Database 19
    and converted to a 28x28 pixel image format and dataset structure that directly matches the MNIST dataset."
    From https://www.nist.gov/itl/iad/image-group/emnist-dataset

    The data split we will use is
    EMNIST ByClass: 814,255 characters. 62 unbalanced classes.
    """

    def __init__(self, subsample_fraction: float = None):
        if not os.path.exists(ESSENTIALS_FILENAME):
            _download_and_process_emnist()
        with open(ESSENTIALS_FILENAME) as f:
            essentials = json.load(f)
        self.mapping = _augment_emnist_mapping(dict(essentials["mapping"]))
        self.inverse_mapping = {v: k for k, v in self.mapping.items()}
        self.num_classes = len(self.mapping)
        self.input_shape = essentials["input_shape"]
        self.output_shape = (self.num_classes,)

        self.subsample_fraction = subsample_fraction
        self.x_train = None
        self.y_train_int = None
        self.x_test = None
        self.y_test_int = None

    def load_or_generate_data(self):
        if not os.path.exists(PROCESSED_DATA_FILENAME):
            _download_and_process_emnist()
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
        return to_categorical(self.y_train_int, self.num_classes)

    @cachedproperty
    def y_test(self):
        return to_categorical(self.y_test_int, self.num_classes)

    def __repr__(self):
        return (
            "EMNIST Dataset\n"
            f"Num classes: {self.num_classes}\n"
            f"Mapping: {self.mapping}\n"
            f"Input shape: {self.input_shape}\n"
        )


def _download_and_process_emnist():
    metadata = toml.load(METADATA_FILENAME)
    curdir = os.getcwd()
    os.chdir(RAW_DATA_DIRNAME)
    _download_raw_dataset(metadata)
    _process_raw_dataset(metadata["filename"])
    os.chdir(curdir)


def _process_raw_dataset(filename: str):
    print("Unzipping EMNIST...")
    zip_file = zipfile.ZipFile(filename, "r")
    zip_file.extract("matlab/emnist-byclass.mat")

    from scipy.io import loadmat  # pylint: disable=import-outside-toplevel

    # NOTE: If importing at the top of module, would need to list scipy as prod dependency.

    print("Loading training data from .mat file")
    data = loadmat("matlab/emnist-byclass.mat")
    x_train = data["dataset"]["train"][0, 0]["images"][0, 0].reshape(-1, 28, 28).swapaxes(1, 2)
    y_train = data["dataset"]["train"][0, 0]["labels"][0, 0]
    x_test = data["dataset"]["test"][0, 0]["images"][0, 0].reshape(-1, 28, 28).swapaxes(1, 2)
    y_test = data["dataset"]["test"][0, 0]["labels"][0, 0]

    if SAMPLE_TO_BALANCE:
        print("Balancing classes to reduce amount of data")
        x_train, y_train = _sample_to_balance(x_train, y_train)
        x_test, y_test = _sample_to_balance(x_test, y_test)

    print("Saving to HDF5 in a compressed format...")
    PROCESSED_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
    with h5py.File(PROCESSED_DATA_FILENAME, "w") as f:
        f.create_dataset("x_train", data=x_train, dtype="u1", compression="lzf")
        f.create_dataset("y_train", data=y_train, dtype="u1", compression="lzf")
        f.create_dataset("x_test", data=x_test, dtype="u1", compression="lzf")
        f.create_dataset("y_test", data=y_test, dtype="u1", compression="lzf")

    print("Saving essential dataset parameters to text_recognizer/datasets...")
    mapping = {int(k): chr(v) for k, v in data["dataset"]["mapping"][0, 0]}
    essentials = {"mapping": list(mapping.items()), "input_shape": list(x_train.shape[1:])}
    with open(ESSENTIALS_FILENAME, "w") as f:
        json.dump(essentials, f)

    print("Cleaning up...")
    shutil.rmtree("matlab")


def _sample_to_balance(x, y):
    """Because the dataset is not balanced, we take at most the mean number of instances per class."""
    np.random.seed(42)
    num_to_sample = int(np.bincount(y.flatten()).mean())
    all_sampled_inds = []
    for label in np.unique(y.flatten()):
        inds = np.where(y == label)[0]
        sampled_inds = np.unique(np.random.choice(inds, num_to_sample))
        all_sampled_inds.append(sampled_inds)
    ind = np.concatenate(all_sampled_inds)
    x_sampled = x[ind]
    y_sampled = y[ind]
    return x_sampled, y_sampled


def _augment_emnist_mapping(mapping):
    """Augment the mapping with extra symbols."""
    # Extra symbols in IAM dataset
    extra_symbols = [
        " ",
        "!",
        '"',
        "#",
        "&",
        "'",
        "(",
        ")",
        "*",
        "+",
        ",",
        "-",
        ".",
        "/",
        ":",
        ";",
        "?",
    ]

    # padding symbol
    extra_symbols.append("_")

    max_key = max(mapping.keys())
    extra_mapping = {}
    for i, symbol in enumerate(extra_symbols):
        extra_mapping[max_key + 1 + i] = symbol

    return {**mapping, **extra_mapping}


def main():
    """Load EMNIST dataset and print info."""
    args = _parse_args()
    dataset = EmnistDataset(subsample_fraction=args.subsample_fraction)
    dataset.load_or_generate_data()

    print(dataset)
    print(dataset.x_train.shape, dataset.y_train.shape)  # pylint: disable=E1101
    print(dataset.x_test.shape, dataset.y_test.shape)  # pylint: disable=E1101


if __name__ == "__main__":
    main()
