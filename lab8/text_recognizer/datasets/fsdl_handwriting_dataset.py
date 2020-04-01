"""Class for loading our own FSDL Handwriting dataset, which encompasses both paragraphs and lines."""
import json

import numpy as np
import toml

from text_recognizer import util
from text_recognizer.datasets.dataset import Dataset


RAW_DATA_DIRNAME = Dataset.data_dirname() / "raw" / "fsdl_handwriting"
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml"
PAGES_DIRNAME = RAW_DATA_DIRNAME / "pages"


class FsdlHandwritingDataset(Dataset):
    """
    FSDL Handwriting dataset gathered in class.
    """

    def __init__(self):
        self.metadata = toml.load(METADATA_FILENAME)
        with open(RAW_DATA_DIRNAME / self.metadata["filename"]) as f:
            page_data = [json.loads(line) for line in f.readlines()]
        # NOTE: pylint bug https://github.com/PyCQA/pylint/issues/3164
        # pylint: disable=unnecessary-comprehension
        self.data_by_page_id = {
            id_: data for id_, data in (_extract_id_and_data(page_datum) for page_datum in page_data)
        }
        # pylint: enable=unnecessary-comprehension

    def load_or_generate_data(self):
        if len(self.page_filenames) < len(self.data_by_page_id):
            self._download_pages()

    @property
    def page_filenames(self):
        return list(PAGES_DIRNAME.glob("*.jpg"))

    def _download_pages(self):
        PAGES_DIRNAME.mkdir(exist_ok=True, parents=True)
        ids, urls = zip(*[(id_, data["url"]) for id_, data in self.data_by_page_id.items()])
        filenames = [PAGES_DIRNAME / id_ for id_ in ids]
        util.download_urls(urls, filenames)

    @property
    def line_regions_by_id(self):
        """Return a dict from name of IAM form to a list of (x1, x2, y1, y2) coordinates of all lines in it."""
        return {id_: data["regions"] for id_, data in self.data_by_page_id.items()}

    @property
    def line_strings_by_id(self):
        """Return a dict from name of image to a list of strings."""
        return {id_: data["strings"] for id_, data in self.data_by_page_id.items()}

    def __repr__(self):
        return "FSDH Handwriting Dataset\n" f"Num pages: {len(self.data_by_page_id)}\n"


def _extract_id_and_data(page_datum):
    """
    page_datum is of the form
        {
            'label': ['line'],
            'shape': 'rectangle',
            'points': [
                [0.1422924901185771, 0.18948824343015216],
                [0.875494071146245, 0.18948824343015216],
                [0.875494071146245, 0.25034578146611347],
                [0.1422924901185771, 0.25034578146611347]
            ],
            'notes': 'A MOVE to stop Mr. Gaitskiell from',
            'imageWidth': 1240,
            'imageHeight': 1771
        }
    """
    url = page_datum["content"]
    id_ = url.split("/")[-1]
    regions = []
    strings = []
    try:
        for annotation in page_datum["annotation"]:
            points = np.array(annotation["points"])
            x1, y1 = points.min(0)
            x2, y2 = points.max(0)
            regions.append(
                {
                    "x1": int(x1 * annotation["imageWidth"]),
                    "y1": int(y1 * annotation["imageHeight"]),
                    "x2": int(x2 * annotation["imageWidth"]),
                    "y2": int(y2 * annotation["imageHeight"]),
                }
            )
            strings.append(annotation["notes"])
    except Exception:  # pylint: disable=broad-except
        pass
    return id_, {"url": url, "regions": regions, "strings": strings}


def main():
    dataset = FsdlHandwritingDataset()
    dataset.load_or_generate_data()
    print(dataset)


if __name__ == "__main__":
    main()
