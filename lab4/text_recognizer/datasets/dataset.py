"""Dataset class to be extended by dataset-specific classes."""
from pathlib import Path
import argparse
import os

from text_recognizer import util


class Dataset:
    """Simple abstract class for datasets."""

    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[3] / "data"

    def load_or_generate_data(self):
        pass


def _download_raw_dataset(metadata):
    if os.path.exists(metadata["filename"]):
        return
    print(f"Downloading raw dataset from {metadata['url']}...")
    util.download_url(metadata["url"], metadata["filename"])
    print("Computing SHA-256...")
    sha256 = util.compute_sha256(metadata["filename"])
    if sha256 != metadata["sha256"]:
        raise ValueError("Downloaded data file SHA-256 does not match that listed in metadata document.")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subsample_fraction", type=float, default=None, help="If given, is used as the fraction of data to expose.",
    )
    return parser.parse_args()
