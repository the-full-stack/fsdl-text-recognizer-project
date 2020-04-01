"""Tests for ParagraphTextRecognizer class."""
import os
from pathlib import Path
import unittest
from text_recognizer.paragraph_text_recognizer import ParagraphTextRecognizer
import text_recognizer.util as util


SUPPORT_DIRNAME = Path(__file__).parents[0].resolve() / "support" / "iam_paragraphs"

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestParagraphTextRecognizer(unittest.TestCase):
    """Test that it can take non-square images of max dimension larger than 256px."""

    def test_filename(self):  # pylint: disable=R0201
        predictor = ParagraphTextRecognizer()
        num_text_lines_by_name = {"a01-000u-cropped": 7}
        for filename in (SUPPORT_DIRNAME).glob("*.jpg"):
            full_image = util.read_image(str(filename), grayscale=True)
            predicted_text, line_region_crops = predictor.predict(full_image)
            print(predicted_text)
            assert len(line_region_crops) == num_text_lines_by_name[filename.stem]
