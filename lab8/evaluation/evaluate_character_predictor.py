"""Run validation test for CharacterPredictor."""
import os
from pathlib import Path
from time import time
import unittest

from text_recognizer.datasets import EmnistDataset
from text_recognizer.character_predictor import CharacterPredictor

os.environ["CUDA_VISIBLE_DEVICES"] = ""

SUPPORT_DIRNAME = Path(__file__).parents[0].resolve() / "support" / "emnist"


class TestEvaluateCharacterPredictor(unittest.TestCase):
    def test_evaluate(self):
        predictor = CharacterPredictor()
        dataset = EmnistDataset()
        dataset.load_or_generate_data()
        t = time()
        metric = predictor.evaluate(dataset)
        time_taken = time() - t
        print(f"acc: {metric}, time_taken: {time_taken}")
        self.assertGreater(metric, 0.6)
        self.assertLess(time_taken, 10)
