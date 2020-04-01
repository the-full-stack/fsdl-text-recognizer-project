"""Run validation test for LinePredictor."""
import os
from pathlib import Path
from time import time
import unittest

from text_recognizer.datasets import EmnistLinesDataset
from text_recognizer.datasets import IamLinesDataset
from text_recognizer.line_predictor import LinePredictor

os.environ["CUDA_VISIBLE_DEVICES"] = ""

EMNIST_SUPPORT_DIRNAME = Path(__file__).parents[0].resolve() / "support" / "emnist_lines"
IAM_SUPPORT_DIRNAME = Path(__file__).parents[0].resolve() / "support" / "iam_lines"


class TestEvaluateLinePredictorEmnist(unittest.TestCase):
    def test_evaluate(self):
        predictor = LinePredictor(EmnistLinesDataset)
        dataset = EmnistLinesDataset()

        dataset.load_or_generate_data()

        t = time()
        metric = predictor.evaluate(dataset)
        time_taken = time() - t

        print(f"acc: {metric}, time_taken: {time_taken}")
        self.assertGreater(metric, 0.6)
        self.assertLess(time_taken, 120)


class TestEvaluateLinePredictorIam(unittest.TestCase):
    def test_evaluate(self):
        predictor = LinePredictor(IamLinesDataset)
        dataset = IamLinesDataset()

        dataset.load_or_generate_data()

        t = time()
        metric = predictor.evaluate(dataset)
        time_taken = time() - t

        print(f"acc: {metric}, time_taken: {time_taken}")
        self.assertGreater(metric, 0.6)
        self.assertLess(time_taken, 180)
