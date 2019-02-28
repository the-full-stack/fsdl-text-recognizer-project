# Lab 2

## Goal of the lab

Move from reading single characters to reading entire lines.

## Outline

- Intro to EMNIST Lines
- Overview of the model and loss
- Explore LSTM training code
- Train an LSTM on EMNIST

## Follow along

```
cd lab2_soln/
```

## Intro to the EMNIST Lines dataset

- Synthetic dataset we built for this project
- Sample sentences from Brown corpus
- For each character, sample random EMNIST character and place on a line (with some random overlap)
- Look at: notebooks/02-look-at-emnist-lines.ipynb

## Overview of model and loss

In this lab we'll keep working with the EmnistLines dataset.

We will be implementing LSTM model with CTC loss.
CTC loss needs to be implemented kind of strangely in Keras: we need to pass in all required data to compute the loss as inputs to the network (including the true label).
This is an example of a multi-input / multi-output network.

The relevant files to review are `models/line_model_ctc.py`, which shows the batch formatting that needs to happen for the CTC loss to be computed inside of the network, `networks/line_lstm_ctc.py`, which has the network definition.

## Train LSTM model with CTC loss

You need to write code in `networks/line_lstm_ctc.py` to make training work.
Training can be done via

```sh
pipenv run python training/run_experiment.py --save '{"train_args": {"epochs": 16}, "dataset": "EmnistLinesDataset", "model": "LineModelCtc", "network": "line_lstm_ctc"}'
```

or the shortcut `tasks/train_lstm_line_predictor.sh`

## Make sure the model is able to predict

You will also need to write some code in `models/line_model_ctc.py` to predict on images.
After that, you should see tests pass when you run

```sh
pipenv run pytest -s text_recognizer/tests/test_line_predictor.py
```

Or you can do `tasks/run_prediction_tests.sh`, which will also run the CharacterModel tests.

## Things to try

If you have time left over, or want to play around with this later on, you can try writing your own non-CTC `line_lstm` network (define it in `text_recognizer/networks/line_lstm.py`).
For example, you could code up an encoder-decoder architecture with attention.
