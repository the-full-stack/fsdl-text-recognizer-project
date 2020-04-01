"""IamParagraphsDataset class and functions for data processing."""
from boltons.cacheutils import cachedproperty
from tensorflow.keras.utils import to_categorical
import cv2
import numpy as np

from text_recognizer.datasets.dataset import Dataset, _parse_args
from text_recognizer.datasets.iam_dataset import IamDataset
from text_recognizer import util

INTERIM_DATA_DIRNAME = Dataset.data_dirname() / "interim" / "iam_paragraphs"
DEBUG_CROPS_DIRNAME = INTERIM_DATA_DIRNAME / "debug_crops"
PROCESSED_DATA_DIRNAME = Dataset.data_dirname() / "processed" / "iam_paragraphs"
CROPS_DIRNAME = PROCESSED_DATA_DIRNAME / "crops"
GT_DIRNAME = PROCESSED_DATA_DIRNAME / "gt"

PARAGRAPH_BUFFER = 50  # pixels in the IAM form images to leave around the lines
TEST_FRACTION = 0.2


class IamParagraphsDataset(Dataset):
    """
    Paragraphs from the IAM dataset.
    """

    def __init__(self, subsample_fraction: float = None):
        self.iam_dataset = IamDataset()
        self.iam_dataset.load_or_generate_data()

        self.num_classes = 3
        self.input_shape = (256, 256)
        self.output_shape = (256, 256, self.num_classes)

        self.subsample_fraction = subsample_fraction
        self.ids = None
        self.x = None
        self.y = None
        self.ids = None

    def load_or_generate_data(self):
        """Load or generate dataset data."""
        num_actual = len(list(CROPS_DIRNAME.glob("*.jpg")))
        num_target = len(self.iam_dataset.line_regions_by_id)
        if num_actual < num_target - 2:  # There are a couple of instances that could not be cropped
            self._process_iam_paragraphs()

        self.x, self.y, self.ids = _load_iam_paragraphs()
        self.train_ind, self.test_ind = _get_random_split(self.x.shape[0])
        self._subsample()

    @cachedproperty
    def x_train(self):
        return self.x[self.train_ind]

    @cachedproperty
    def y_train(self):
        return self.y[self.train_ind]

    @cachedproperty
    def x_test(self):
        return self.x[self.test_ind]

    @cachedproperty
    def y_test(self):
        return self.y[self.test_ind]

    @cachedproperty
    def ids_train(self):
        return self.ids[self.train_ind]

    @cachedproperty
    def ids_test(self):
        return self.ids[self.test_ind]

    def get_x_and_y_from_id(self, id_):
        ind = self.ids.index(id_)
        return self.x[ind], self.y[ind]

    def _process_iam_paragraphs(self):
        """
        For each page, crop out the part of it that correspond to the paragraph of text, and make sure all crops are
        self.input_shape. The ground truth data is the same size, with a one-hot vector at each pixel
        corresponding to labels 0=background, 1=odd-numbered line, 2=even-numbered line
        """
        crop_dims = self._decide_on_crop_dims()
        CROPS_DIRNAME.mkdir(parents=True, exist_ok=True)
        DEBUG_CROPS_DIRNAME.mkdir(parents=True, exist_ok=True)
        GT_DIRNAME.mkdir(parents=True, exist_ok=True)
        print(f"Cropping paragraphs, generating ground truth, and saving debugging images to {DEBUG_CROPS_DIRNAME}")
        for filename in self.iam_dataset.form_filenames:
            id_ = filename.stem
            line_region = self.iam_dataset.line_regions_by_id[id_]
            _crop_paragraph_image(filename, line_region, crop_dims, self.input_shape)

    def _decide_on_crop_dims(self):
        """
        Decide on the dimensions to crop out of the form image.
        Since image width is larger than a comfortable crop around the longest paragraph,
        we will make the crop a square form factor.

        And since the found dimensions 610x610 are pretty close to 512x512,
        we might as well resize crops and make it exactly that, which lets us
        do all kinds of power-of-2 pooling and upsampling should we choose to.
        """
        sample_form_filename = self.iam_dataset.form_filenames[0]
        sample_image = util.read_image(sample_form_filename, grayscale=True)
        max_crop_width = sample_image.shape[1]
        max_crop_height = _get_max_paragraph_crop_height(self.iam_dataset.line_regions_by_id)
        assert max_crop_height <= max_crop_width
        crop_dims = (max_crop_width, max_crop_width)
        print(f"Max crop width and height were found to be {max_crop_width}x{max_crop_height}.")
        print(f"Setting them to {max_crop_width}x{max_crop_width}")
        return crop_dims

    def _subsample(self):
        """Only this fraction of data will be loaded."""
        if self.subsample_fraction is None:
            return
        num_subsample = int(self.x.shape[0] * self.subsample_fraction)
        self.x = self.x[:num_subsample]
        self.y = self.y[:num_subsample]
        self.ids = self.ids[:num_subsample]

    def __repr__(self):
        """Print info about the dataset."""
        return (
            "IAM Paragraphs Dataset\n"  # pylint: disable=no-member
            f"Num classes: {self.num_classes}\n"
            f"Train: {self.x_train.shape} {self.y_train.shape}\n"
            f"Test: {self.x_test.shape} {self.y_test.shape}\n"
        )


def _get_max_paragraph_crop_height(line_regions_by_id):
    heights = []
    for regions in line_regions_by_id.values():
        min_y1 = min(r["y1"] for r in regions) - PARAGRAPH_BUFFER
        max_y2 = max(r["y2"] for r in regions) + PARAGRAPH_BUFFER
        height = max_y2 - min_y1
        heights.append(height)
    return max(heights)


def _crop_paragraph_image(filename, line_regions, crop_dims, final_dims):  # pylint: disable=too-many-locals
    image = util.read_image(filename, grayscale=True)

    min_y1 = min(r["y1"] for r in line_regions) - PARAGRAPH_BUFFER
    max_y2 = max(r["y2"] for r in line_regions) + PARAGRAPH_BUFFER
    height = max_y2 - min_y1
    crop_height = crop_dims[0]
    buffer = (crop_height - height) // 2

    # Generate image crop
    image_crop = 255 * np.ones(crop_dims, dtype=np.uint8)
    try:
        image_crop[buffer : buffer + height] = image[min_y1:max_y2]
    except Exception as e:  # pylint: disable=broad-except
        print(f"Rescued {filename}: {e}")
        return

    # Generate ground truth
    gt_image = np.zeros_like(image_crop, dtype=np.uint8)
    for ind, region in enumerate(line_regions):
        gt_image[(region["y1"] - min_y1 + buffer) : (region["y2"] - min_y1 + buffer), region["x1"] : region["x2"]] = (
            ind % 2 + 1
        )

    # Generate image for debugging
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    cmap = plt.get_cmap("Set1")
    image_crop_for_debug = np.dstack([image_crop, image_crop, image_crop])
    for ind, region in enumerate(line_regions):
        color = [255 * _ for _ in cmap(ind)[:-1]]
        cv2.rectangle(
            image_crop_for_debug,
            (region["x1"], region["y1"] - min_y1 + buffer),
            (region["x2"], region["y2"] - min_y1 + buffer),
            color,
            3,
        )
    image_crop_for_debug = cv2.resize(image_crop_for_debug, final_dims, interpolation=cv2.INTER_AREA)
    util.write_image(image_crop_for_debug, DEBUG_CROPS_DIRNAME / f"{filename.stem}.jpg")

    image_crop = cv2.resize(image_crop, final_dims, interpolation=cv2.INTER_AREA)  # Quality interpolation for input
    util.write_image(image_crop, CROPS_DIRNAME / f"{filename.stem}.jpg")

    gt_image = cv2.resize(gt_image, final_dims, interpolation=cv2.INTER_NEAREST)  # No interpolation for labels
    util.write_image(gt_image, GT_DIRNAME / f"{filename.stem}.png")


def _load_iam_paragraphs():
    print("Loading IAM paragraph crops and ground truth from image files...")
    images = []
    gt_images = []
    ids = []
    for filename in CROPS_DIRNAME.glob("*.jpg"):
        id_ = filename.stem
        image = util.read_image(filename, grayscale=True)
        image = 1.0 - image / 255

        gt_filename = GT_DIRNAME / f"{id_}.png"
        gt_image = util.read_image(gt_filename, grayscale=True)

        images.append(image)
        gt_images.append(gt_image)
        ids.append(id_)
    images = np.array(images).astype(np.float32)
    gt_images = to_categorical(np.array(gt_images), 3).astype(np.uint8)
    return images, gt_images, np.array(ids)


def _get_random_split(num_total):
    np.random.seed(42)
    num_train = int((1 - TEST_FRACTION) * num_total)
    ind = np.random.permutation(num_total)
    train_ind, test_ind = (
        ind[:num_train],
        ind[num_train:],
    )  # pylint: disable=unsubscriptable-object
    return train_ind, test_ind


def main():
    """Load dataset and print info."""
    args = _parse_args()
    dataset = IamParagraphsDataset(subsample_fraction=args.subsample_fraction)
    dataset.load_or_generate_data()
    print(dataset)


if __name__ == "__main__":
    main()
