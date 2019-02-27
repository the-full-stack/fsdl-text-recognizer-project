# Lab 2

In this lab we will implement convolutional networks: first for character recognition, and then for recognizing text in an image of a handwritten line.

## LeNet

To warm up, let's implement an old classic.

Add enough code to `networks/lenet.py` to be able to run

```sh
pipenv run training/run_experiment.py '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "lenet", "train_args": {"epochs": 1}}'
```

Training the single epoch will take about 2 minutes (that's why we only do one epoch in this lab :)).
Leave it running while we go on to look at EmnistLines.

## Emnist Lines

We are generating synthetic data from composing EMNIST characters into a line, sampling text from the Brown corpus!
We can see the results by opening up `notebooks/02-look-at-emnist-lines.ipynb`.

## Train an all-conv model on EmnistLines

Add code to `networks/line_cnn_all_conv.py` to make the following command train successfully.

```sh
pipenv run training/run_experiment.py '{"train_args": {"epochs": 16}, "dataset": "EmnistLinesDataset", "model": "LineModel", "network": "line_cnn_all_conv", "network_args": {"window_width": 16, "window_stride": 8}}'
```

Again, it will take a few minutes to train the model.

You should be able to get to ~70% character accuracy with the default params.
