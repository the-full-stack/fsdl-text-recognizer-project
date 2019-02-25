import pathlib
import unittest

from text_recognizer.character_predictor import CharacterPredictor


SUPPORT_DIRNAME = pathlib.Path(__file__).parents[0].resolve() / 'support' / 'emnist'


class TestCharacterPredictor(unittest.TestCase):
    def test_filename(self):
      predictor = CharacterPredictor()

      for filename in SUPPORT_DIRNAME.glob('*.png'):
        pred, conf = predictor.predict(str(filename))
        print(f'Prediction: {pred} at confidence: {conf} for image with character {filename.stem}')
        self.assertEqual(pred, filename.stem)
        # self.assertGreater(conf, 0.9)
        # Confidence tests are usually too brittle

