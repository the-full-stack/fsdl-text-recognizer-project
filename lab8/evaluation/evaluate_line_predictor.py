import pathlib
from time import time
import unittest

from text_recognizer.datasets import EmnistLinesDataset, IamLinesDataset
from text_recognizer.line_predictor import LinePredictor


SUPPORT_DIRNAME = pathlib.Path(__file__).parents[0].resolve() / 'support' / 'emnist_lines'


class TestEvaluateLinePredictor(unittest.TestCase):
    def test_evaluate(self):
        # predictor = LinePredictor(IamLinesDataset)
        # dataset = IamLinesDataset()

        predictor = LinePredictor(EmnistLinesDataset)
        dataset = EmnistLinesDataset()

        dataset.load_or_generate_data()

        t = time()
        metric = predictor.evaluate(dataset)
        time_taken = time() - t

        print(f'acc: {metric}, time_taken: {time_taken}')
        self.assertGreater(metric, 0.8)
        self.assertLess(time_taken, 60)

