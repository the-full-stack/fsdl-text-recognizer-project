"""
Takes an image and returns all the text in it, by first segmenting the image with LineDetector, then extracting crops
of the image corresponding to the line regions, and running each line region crop through LinePredictor.
"""
from typing import List, Tuple, Union
import cv2
import numpy as np
from text_recognizer.datasets import IamLinesDataset
from text_recognizer.models.line_detector_model import LineDetectorModel
from text_recognizer.models.line_model_ctc import LineModelCtc
import text_recognizer.util as util


class ParagraphTextRecognizer:
    """Given an image of a single handwritten character, recognizes it."""

    def __init__(self):
        self.line_detector_model = LineDetectorModel()
        self.line_detector_model.load_weights()
        self.line_predictor_model = LineModelCtc(dataset_cls=IamLinesDataset)
        self.line_predictor_model.load_weights()

    def predict(self, image_or_filename: Union[np.ndarray, str]):
        """
        Take an image and return all the text in it.
        """
        if isinstance(image_or_filename, str):
            image = util.read_image(image_or_filename, grayscale=True)
        else:
            image = image_or_filename

        line_region_crops = self._get_line_region_crops(image=image)
        print([a.shape for a in line_region_crops])
        prepared_line_region_crops = [
            self._prepare_image_for_line_predictor_model(image=crop) for crop in line_region_crops
        ]

        line_region_strings = [
            self.line_predictor_model.predict_on_image(crop)[0] for crop in prepared_line_region_crops
        ]
        return " ".join(line_region_strings), line_region_crops

    def _get_line_region_crops(self, image: np.ndarray, min_crop_len_factor: float = 0.02) -> List[np.ndarray]:
        """Find all the line regions in square image and crop them out and return them."""
        prepared_image, scale_down_factor = self._prepare_image_for_line_detector_model(image)
        line_segmentation = self.line_detector_model.predict_on_image(prepared_image)
        bounding_boxes_xywh = _find_line_bounding_boxes(line_segmentation)

        bounding_boxes_xywh = (bounding_boxes_xywh * scale_down_factor).astype(int)

        min_crop_length = int(min_crop_len_factor * min(image.shape[0], image.shape[1]))
        line_region_crops = [
            image[y : y + h, x : x + w]
            for x, y, w, h in bounding_boxes_xywh
            if w >= min_crop_length and h >= min_crop_length
        ]
        return line_region_crops

    def _prepare_image_for_line_detector_model(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Convert uint8 image to float image with black background with shape self.line_detector_model.image_shape."""
        resized_image, scale_down_factor = _resize_image_for_line_detector_model(
            image=image, max_shape=self.line_detector_model.image_shape
        )
        resized_image = (1.0 - resized_image / 255).astype("float32")
        return resized_image, scale_down_factor

    def _prepare_image_for_line_predictor_model(self, image: np.ndarray) -> np.ndarray:
        """
        Convert uint8 image to float image with black background with shape self.line_predictor_model.image_shape
        while maintaining the image aspect ratio.
        """
        expected_shape = self.line_predictor_model.image_shape
        scale_factor = (np.array(expected_shape) / np.array(image.shape)).min()
        scaled_image = cv2.resize(image, dsize=None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

        pad_width = (
            (0, expected_shape[0] - scaled_image.shape[0]),
            (0, expected_shape[1] - scaled_image.shape[1]),
        )
        padded_image = np.pad(scaled_image, pad_width=pad_width, mode="constant", constant_values=255)
        return 1 - padded_image / 255


def _find_line_bounding_boxes(line_segmentation: np.ndarray):
    """Given a line segmentation, find bounding boxes for connected-component regions corresponding to non-0 labels."""

    def _find_line_bounding_boxes_in_channel(line_segmentation_channel: np.ndarray) -> np.ndarray:
        line_activation_image = cv2.dilate(line_segmentation_channel, kernel=np.ones((3, 3)), iterations=1)
        line_activation_image = (line_activation_image * 255).astype("uint8")
        line_activation_image = cv2.threshold(line_activation_image, 0.5, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        bounding_cnts, _ = cv2.findContours(line_activation_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return np.array([cv2.boundingRect(cnt) for cnt in bounding_cnts])

    bboxes_xywh = np.concatenate(
        [_find_line_bounding_boxes_in_channel(line_segmentation[:, :, i]) for i in [1, 2]], axis=0
    )
    return bboxes_xywh[np.argsort(bboxes_xywh[:, 1])]


def _resize_image_for_line_detector_model(image: np.ndarray, max_shape: Tuple[int, int]) -> Tuple[np.ndarray, float]:
    """Resize the image to less than the max_shape while maintaining aspect ratio."""
    scale_down_factor = max(np.array(image.shape) / np.array(max_shape))
    if scale_down_factor == 1:
        return image.copy(), scale_down_factor
    resized_image = cv2.resize(
        image, dsize=None, fx=1 / scale_down_factor, fy=1 / scale_down_factor, interpolation=cv2.INTER_AREA,
    )
    return resized_image, scale_down_factor
