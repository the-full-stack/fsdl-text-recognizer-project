"""Flask web server serving text_recognizer predictions."""
# From https://github.com/UnitedIncome/serverless-python-requirements
try:
    import unzip_requirements  # pylint: disable=unused-import
except ImportError:
    pass

import os

from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras import backend

from text_recognizer.line_predictor import LinePredictor
# from text_recognizer.datasets import IamLinesDataset
import text_recognizer.util as util

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Do not use GPU

app = Flask(__name__)  # pylint: disable=invalid-name


@app.before_first_request
def load_model_to_app():
    """Instantiate tensorflow session and load model."""
    # NOTE: following https://github.com/keras-team/keras/issues/2397#issuecomment-519128406
    app.session = tf.Session(graph=tf.Graph())
    with app.session.graph.as_default():
        backend.set_session(app.session)
        app.predictor = LinePredictor()


@app.route('/')
def index():
    """Provide simple health check route."""
    return 'Hello, world!'


@app.route('/v1/predict', methods=['GET', 'POST'])
def predict():
    """Provide main prediction API route. Responds to both GET and POST requests."""
    image = _load_image()
    with app.session.graph.as_default():
        backend.set_session(app.session)
        pred, conf = app.predictor.predict(image)
        print("METRIC confidence {}".format(conf))
        print("METRIC mean_intensity {}".format(image.mean()))
        print("INFO pred {}".format(pred))
    return jsonify({'pred': str(pred), 'conf': float(conf)})


def _load_image():
    if request.method == 'POST':
        data = request.get_json()
        if data is None:
            return 'no json received'
        return util.read_b64_image(data['image'], grayscale=True)
    if request.method == 'GET':
        image_url = request.args.get('image_url')
        if image_url is None:
            return 'no image_url defined in query string'
        print("INFO url {}".format(image_url))
        return util.read_image(image_url, grayscale=True)
    raise ValueError('Unsupported HTTP method')


def main():
    """Run the app."""
    app.run(host='0.0.0.0', port=8000, debug=False)  # nosec


if __name__ == '__main__':
    main()
