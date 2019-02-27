"""IamParagraphsDataset class and functions for data processing."""
from pathlib import Path

from boltons.cacheutils import cachedproperty
from tensorflow.keras.utils import to_categorical
import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt

from text_recognizer.datasets.base import Dataset, _parse_args
from text_recognizer.datasets.iam import IamDataset
from text_recognizer import util


DATA_DIRNAME = Path(__file__).parents[2].resolve() / 'data'
INTERIM_DATA_DIRNAME = DATA_DIRNAME / 'interim' / 'iam_paragraphs'
DEBUG_CROPS_DIRNAME = INTERIM_DATA_DIRNAME / 'paragraph_crops_for_debug'
PROCESSED_DATA_DIRNAME = DATA_DIRNAME / 'processed' / 'iam_paragraphs'
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / 'data.h5'

# On top of IAM forms being downsampled, we will downsample the paragraph crops once more, to make convnets tractable.
DOWNSAMPLE_FACTOR = 2
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
        self.input_shape = (512, 512)
        self.output_shape = (512, 512, self.num_classes)

        self.subsample_fraction = subsample_fraction
        self.x_train = None
        self.x_test = None
        self.y_train_int = None
        self.y_test_int = None

    def load_or_generate_data(self):
        """Load or generate dataset data."""
        if not PROCESSED_DATA_FILENAME.exists():
            self._process_iam_paragraphs()
        with h5py.File(PROCESSED_DATA_FILENAME, 'r') as f:
            self.x_train = f['x_train'][:] / 255
            self.y_train_int = f['y_train_int'][:]
            self.x_test = f['x_test'][:] / 255
            self.y_test_int = f['y_test_int'][:]
        self._subsample()

    @cachedproperty
    def y_train(self):
        return _transform_to_categorical(self.y_train_int)

    @cachedproperty
    def y_test(self):
        return _transform_to_categorical(self.y_test_int)

    def _process_iam_paragraphs(self):
        """
        For each page, crop out the part of it that correspond to the paragraph of text, and make sure all crops are
        512x512. The ground truth data is the same size, with a one-hot vector at each pixel corresponding to labels
        0=background, 1=odd-numbered line, 2=even-numbered line
        """
        crop_dims = self._decide_on_crop_dims()
        DEBUG_CROPS_DIRNAME.mkdir(parents=True, exist_ok=True)
        print(f'Cropping paragraphs, generating ground truth, and saving debugging images to {DEBUG_CROPS_DIRNAME}')
        x = []
        y = []
        for filename in self.iam_dataset.form_filenames:
            name = filename.stem
            line_region = self.iam_dataset.line_regions_by_name[name]
            crop_image, gt_image = _crop_paragraph_image(filename, line_region, crop_dims, self.input_shape)
            if crop_image is None or gt_image is None:
                continue
            x.append(crop_image)
            y.append(gt_image)
        x = np.array(x)
        y = np.array(y)

        num_total = x.shape[0]
        num_train = int((1 - TEST_FRACTION) * num_total)
        shuffled_ind = np.random.permutation(num_total)
        train_ind, test_ind = shuffled_ind[:num_train], shuffled_ind[num_train:]

        print('Saving to HDF5 in a compressed format...')
        PROCESSED_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
        with h5py.File(PROCESSED_DATA_FILENAME, 'w') as f:
            f.create_dataset('x_train', data=x[train_ind], dtype='u1', compression='lzf')
            f.create_dataset('y_train_int', data=y[train_ind], dtype='u1', compression='lzf')
            f.create_dataset('x_test', data=x[test_ind], dtype='u1', compression='lzf')
            f.create_dataset('y_test_int', data=y[test_ind], dtype='u1', compression='lzf')

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
        max_crop_height = _get_max_paragraph_crop_height(self.iam_dataset.line_regions_by_name)
        assert max_crop_height <= max_crop_width
        crop_dims = (max_crop_width, max_crop_width)
        print(f'Max crop width and height were found to be {max_crop_width}x{max_crop_height}.')
        print(f'Setting them to {max_crop_width}x{max_crop_width}')
        return crop_dims

    def _subsample(self):
        """Only this fraction of data will be loaded."""
        if self.subsample_fraction is None:
            return
        num_train = int(self.x_train.shape[0] * self.subsample_fraction)
        num_test = int(self.x_test.shape[0] * self.subsample_fraction)
        self.x_train = self.x_train[:num_train]
        self.y_train_int = self.y_train_int[:num_train]
        self.x_test = self.x_test[:num_test]
        self.y_test_int = self.y_test_int[:num_test]

    def __repr__(self):
        """Print info about the dataset."""
        return (
            'IAM Paragraphs Dataset\n'  # pylint: disable=no-member
            f'Num classes: {self.num_classes}\n'
            f'Train: {self.x_train.shape} {self.y_train.shape}\n'
            f'Test: {self.x_test.shape} {self.y_test.shape}\n'
        )


def _get_max_paragraph_crop_height(line_regions_by_name):
    heights = []
    for regions in line_regions_by_name.values():
        min_y1 = min(r['y1'] for r in regions) - PARAGRAPH_BUFFER
        max_y2 = max(r['y2'] for r in regions) + PARAGRAPH_BUFFER
        height = max_y2 - min_y1
        heights.append(height)
    return max(heights)


def _crop_paragraph_image(filename, line_regions, crop_dims, final_dims):
    image = util.read_image(filename, grayscale=True)

    min_y1 = min(r['y1'] for r in line_regions) - PARAGRAPH_BUFFER
    max_y2 = max(r['y2'] for r in line_regions) + PARAGRAPH_BUFFER
    height = max_y2 - min_y1
    crop_height = crop_dims[0]
    buffer = (crop_height - height) // 2

    # Generate image crop
    image_crop = 255 * np.ones(crop_dims, dtype=np.uint8)
    try:
        image_crop[buffer:buffer + height] = image[min_y1:max_y2]
    except Exception as e:
        print(f'Rescued {filename}: {e}')
        return None, None

    # Generate ground truth
    gt_image = np.zeros_like(image_crop, dtype=np.uint8)
    for ind, region in enumerate(line_regions):
        gt_image[
            (region['y1'] - min_y1 + buffer):(region['y2'] - min_y1 + buffer),
            region['x1']:region['x2']
        ] = ind + 1

    # Generate image for debugging
    cmap = plt.get_cmap('Set1')
    image_crop_for_debug = np.dstack([image_crop, image_crop, image_crop])
    for ind, region in enumerate(line_regions):
        color = [255 * _ for _ in cmap(ind)[:-1]]
        cv2.rectangle(
            image_crop_for_debug,
            (region['x1'], region['y1'] - min_y1 + buffer),
            (region['x2'], region['y2'] - min_y1 + buffer),
            color,
            3
        )
    image_crop_for_debug = cv2.resize(image_crop_for_debug, final_dims, interpolation=cv2.INTER_CUBIC)
    util.write_image(image_crop_for_debug, DEBUG_CROPS_DIRNAME / f'{filename.stem}.jpg')

    image_crop = cv2.resize(image_crop, final_dims, interpolation=cv2.INTER_CUBIC)
    gt_image = cv2.resize(gt_image, final_dims, interpolation=cv2.INTER_CUBIC)

    return image_crop, gt_image


def _transform_to_categorical(gt_images):
    """
    gt_images has a unique value for every line.
    We transform to just three values, and one-hot encode.
    """
    categorical_gt_images = []
    for gt_image in gt_images:
        for ind, value in enumerate(np.unique(gt_image)):
            if value == 0:
                continue
            gt_image[gt_image == value] = ind % 2 + 1
        categorical_gt_images.append(to_categorical(gt_image, 3))
    return np.array(categorical_gt_images).astype(bool)


def main():
    """Load dataset and print info."""
    args = _parse_args()
    dataset = IamParagraphsDataset(subsample_fraction=args.subsample_fraction)
    dataset.load_or_generate_data()
    print(dataset)


if __name__ == '__main__':
    main()

