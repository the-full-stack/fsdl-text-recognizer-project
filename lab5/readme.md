# Lab 5: Line Detection

At this point, we have trained a model that can recognize text in a line, given an image of a single line.

## Goal of the lab

Our next task is to automatically detect line regions in an image of a whole paragraph of text.

Our approach will be to train a model that, when given an image containing lines of text, returns a pixelwise labeling of that image, with each pixel belonging to either background, odd line of handwriting, or even line of handwriting.
Given the output of the model, we can find line regions with an easy image processing operation.

## Setup

- As always, `git pull` in the `~/fsdl-text-recognizer-project` repo to get the latest code.
- Then `cd lab5`.

## Data

We are starting from the IAM dataset, which includes not only lines but the original writing sample forms, with each line and word region annotated.

Let's load the IAM dataset and then look at the data files.
Run `python text_recognizer/datasets/iam_dataset.py`
Let's look at the raw data files, which are in `~/fsdl-text-recognizer-project/data/raw/iam/iamdb/forms`.

We want to crop out the region of each page corresponding to the handwritten paragraph as our model input, and generate corresponding ground truth.

Code to do this is in `text_recognizer/datasets/iam_paragraphs_dataset.py`

We can look at the results in `notebooks/04-look-at-iam-paragraphs.ipynb` and by looking at some debug images we output in `data/interim/iam_paragraphs`.

## Training data augmentation

The model code for our new `LineDetector` is in `text_recognizer/models/line_detector_model.py`.

Because we only have about a thousand images to learn this task on, data augmentation will be crucial.
Image augmentations such as streching, slight rotations, offsets, contrast and brightness changes, and potentially even mirror-flipping are tedious to code, and most frameworks provide optimized utility code for the task.

We use Keras's `ImageDataGenerator`, and you can see the parameters for it in `text_recognizer/models/line_detector_model.py`.
We can take a look at what the data transformations look like in the same notebook.

## Network description

The network used in this model is `text_recognizer/networks/fcn.py`.

The basic idea is a deep convolutional network with resnet-style blocks (input to block is concatenated to block output).
We call it FCN, as in "Fully Convolutional Network," after the seminal paper that first used convnets for segmentation.

Unlike the original FCN, however, we do not maxpool or upsample, but instead rely on dilated convolutions to rapidly increase the effective receptive field.
[Here](https://fomoro.com/research/articles/receptive-field-calculator) is a very calculator of the effective receptive field size of a convnet.

The crucial thing to understand is that because we are labeling odd and even lines differently, each predicted pixel must have the context of the entire image to correctly label -- otherwise, there is no way to know whether the pixel is on an odd or even line.

## Review results

The model converges to something really good.

Check out `notebooks/04b-look-at-line-detector-predictions.ipynb` to see sample predictions on the test set.

We also plot some sample training data augmentation in that notebook.

## Combining the two models

Now we are ready to combine the new `LineDetector` model and the `LinePredictor` model that we trained yesterday.

This is done in `text_recognizer/paragraph_text_recognizer.py`, which loads both models, find line regions with one, and runs each crop through the other.

We can see that it works as expected (albeit not too accurately yet) by running `pytest -s text_recognizer/tests/test_paragraph_text_recognizer.py`.

## Things to try

- Try adding more data augmentations, or mess with the parameters of the existing ones
- Try the U-Net architecture, which MaxPools down and then UpSamples back up, with increased conv layer channel dimensions in the middle (https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).
