import pathlib
from time import time
import unittest

from text_recognizer.datasets.emnist import EmnistDataset
from text_recognizer.character_predictor import CharacterPredictor


SUPPORT_DIRNAME = pathlib.Path(__file__).parents[0].resolve() / 'support' / 'emnist'


class TestEvaluateCharacterPredictor(unittest.TestCase):
    def test_evaluate(self):
        predictor = CharacterPredictor()
        dataset = EmnistDataset()
        dataset.load_or_generate_data()
        t = time()
        metric = predictor.evaluate(dataset)
        time_taken = time() - t
        print(f'acc: {metric}, time_taken: {time_taken}')
        self.assertGreater(metric, 0.7)
        self.assertLess(time_taken, 10)

