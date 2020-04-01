"""Tests for web app."""
import os
from pathlib import Path
from unittest import TestCase
import base64

from api.app import app

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_DIRNAME = Path(__file__).parents[2].resolve()
# SUPPORT_DIRNAME = REPO_DIRNAME / 'text_recognizer' / 'tests' / 'support' / 'iam_lines'
SUPPORT_DIRNAME = REPO_DIRNAME / "text_recognizer" / "tests" / "support" / "emnist_lines"


class TestIntegrations(TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_index(self):
        response = self.app.get("/")
        assert response.get_data().decode() == "Hello, world!"

    def test_predict(self):
        with open(SUPPORT_DIRNAME / "or if used the results.png", "rb") as f:
            b64_image = base64.b64encode(f.read())
        response = self.app.post("/v1/predict", json={"image": f"data:image/jpeg;base64,{b64_image.decode()}"})
        json_data = response.get_json()
        self.assertEqual(json_data["pred"], "or if used the resuits")
