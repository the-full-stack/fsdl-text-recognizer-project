"""Tests for ParagraphTextRecognizer class."""
import os
from pathlib import Path
import unittest
from text_recognizer.paragraph_text_recognizer import ParagraphTextRecognizer


SUPPORT_DIRNAME = Path(__file__).parents[0].resolve() / 'support' / 'iam_paragraphs'

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestEmnistLinePredictor(unittest.TestCase):
    def test_filename(self):  # pylint: disable=R0201
        predictor = ParagraphTextRecognizer()

        for filename in (SUPPORT_DIRNAME).glob('*.jpg'):
            predicted_text, line_region_crops = predictor.predict(str(filename))
            print(predicted_text)
            assert len(line_region_crops) == 8

