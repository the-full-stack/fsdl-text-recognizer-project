"""MNIST DataModule."""
import argparse

from torch.utils.data import random_split
from torchvision.datasets import MNIST as TorchMNIST

from text_recognizer.data.base_data_module import BaseDataModule, load_and_print_info
import text_recognizer.metadata.mnist as metadata
from text_recognizer.stems.mnist import MNISTStem


class MNIST(BaseDataModule):
    """MNIST DataModule."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.data_dir = metadata.DOWNLOADED_DATA_DIRNAME
        self.transform = MNISTStem()
        self.input_dims = metadata.DIMS
        self.output_dims = metadata.OUTPUT_DIMS
        self.mapping = metadata.MAPPING

    def prepare_data(self, *args, **kwargs) -> None:
        """Download train and test MNIST data from PyTorch canonical source."""
        TorchMNIST(self.data_dir, train=True, download=True)
        TorchMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None) -> None:
        """Split into train, val, test, and set dims."""
        mnist_full = TorchMNIST(self.data_dir, train=True, transform=self.transform)
        self.data_train, self.data_val = random_split(mnist_full, [metadata.TRAIN_SIZE, metadata.VAL_SIZE])  # type: ignore
        self.data_test = TorchMNIST(self.data_dir, train=False, transform=self.transform)


if __name__ == "__main__":
    load_and_print_info(MNIST)
