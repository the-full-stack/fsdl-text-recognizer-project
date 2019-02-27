"""Dataset class to be extended by dataset-specific classes."""
from pathlib import Path
from typing import Union
from urllib.request import urlretrieve
import argparse
import hashlib
import os


class Dataset:
    """Simple abstract class for datasets."""
    @classmethod
    def data_dirname(cls):
        return Path(__file__).parents[2].resolve() / 'data'

    def load_or_generate_data(self):
        pass


def _download_raw_dataset(metadata):
    if os.path.exists(metadata['filename']):
        return
    print('Downloading raw dataset...')
    urlretrieve(metadata['url'], metadata['filename'])  # nosec
    print('Computing SHA-256...')
    sha256 = _compute_sha256(metadata['filename'])
    if sha256 != metadata['sha256']:
        raise ValueError('Downloaded data file SHA-256 does not match that listed in metadata document.')


def _compute_sha256(filename: Union[Path, str]):
    """Return SHA256 checksum of a file."""
    with open(filename, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample_fraction",
                        type=float,
                        default=None,
                        help="If given, is used as the fraction of data to expose.")
    return parser.parse_args()

