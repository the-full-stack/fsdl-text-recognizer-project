"""Dataset class to be extended by dataset-specific classes."""
import pathlib


class Dataset:
    """Simple abstract class for datasets."""
    @classmethod
    def data_dirname(cls):
        return pathlib.Path(__file__).parents[3].resolve() / 'data'

    def load_or_generate_data(self):
        pass

