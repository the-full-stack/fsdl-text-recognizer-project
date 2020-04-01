"""Flask web server serving text_recognizer predictions."""
import os

from flask import Flask, request, jsonify
import tensorflow.keras.backend as K

from text_recognizer.line_predictor import LinePredictor
import text_recognizer.util as util

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Do not use GPU

app = Flask(__name__)  # pylint: disable=invalid-name


@app.route("/")
def index():
    """Provide simple health check route."""
    return "Hello, world!"


@app.route("/v1/predict", methods=["GET", "POST"])
def predict():
    """Provide main prediction API route. Responds to both GET and POST requests."""
    K.clear_session()
    predictor = LinePredictor()
    image = _load_image()
    pred, conf = predictor.predict(image)
    print("METRIC confidence {}".format(conf))
    print("METRIC mean_intensity {}".format(image.mean()))
    print("INFO pred {}".format(pred))
    return jsonify({"pred": str(pred), "conf": float(conf)})


def _load_image():
    if request.method == "POST":
        data = request.get_json()
        if data is None:
            return "no json received"
        return util.read_b64_image(data["image"], grayscale=True)
    if request.method == "GET":
        image_url = request.args.get("image_url")
        if image_url is None:
            return "no image_url defined in query string"
        print("INFO url {}".format(image_url))
        return util.read_image(image_url, grayscale=True)
    raise ValueError("Unsupported HTTP method")


def main():
    """Run the app."""
    app.run(host="0.0.0.0", port=8000, debug=False)  # nosec


if __name__ == "__main__":
    main()
