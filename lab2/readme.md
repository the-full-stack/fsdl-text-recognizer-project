# Lab 2: Convolutional Nets

## Goal of the lab

- Use a simple convolutional network to recognize EMNIST characters.
- Construct a synthetic dataset of EMNIST lines.
- Move from reading single characters to reading lines.

## Follow along

```
git pull
cd lab2
```

## Using a convolutional network for recognizing EMNIST characters

We left off in Lab 1 having trained an MLP model on the EMNIST characters dataset.

Let's also train a CNN on the same task.
We can start in the notebook `notebooks/01b-cnn-for-emnist.ipynb`.

We can also run the same experiment with

```sh
training/run_experiment.py '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "lenet", "train_args": {"epochs": 1}}'
```

Training the single epoch will take about 2 minutes (that's why we only do one epoch in this lab :)).
Leave it running while we go on to the next part.

### Subsampling data

It is very useful to be able to subsample the dataset for quick experiments.
This is possibe by passing `subsample_fraction=0.1` (or some other fraction) at dataset initialization, or in `dataset_args` in the `run_experiment.py` dictionary, for example:

```sh
training/run_experiment.py '{"dataset": "EmnistDataset", "dataset_args": {"subsample_fraction": 0.25}, "model": "CharacterModel", "network": "lenet"}'
```

## Making a synthetic dataset of EMNIST Lines

- Synthetic dataset we built for this project
- Sample sentences from Brown corpus
- For each character, sample random EMNIST character and place on a line (with some random overlap)
- Look at: `notebooks/02-look-at-emnist-lines.ipynb`

## Reading multiple characters at once

Now that we have a dataset of lines and not just single characters, we can apply our convolutional net to it.

Let's look at `notebooks/02b-cnn-for-simple-emnist-lines.ipynb`, where we generate a datset with at most 8 characters and no overlap.

The first network we try is simply the same LeNet network we used for single characters, applied to each character in sequence, using the `TimeDistributed` layer.

We can also express the same network using all convolutional layers, which we do next.

We can train this model with a command, too:

```sh
python training/run_experiment.py --save '{"train_args": {"epochs": 5}, "dataset": "EmnistLinesDataset", "dataset_args": {"max_length": 8, "max_overlap": 0}, "model": "LineModel", "network": "line_cnn_all_conv"}'
```
