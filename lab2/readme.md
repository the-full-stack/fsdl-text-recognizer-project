# Lab 2: Line text recognizer

## Goal of the lab

Move from reading single characters to reading entire lines.

## Outline

- Intro to EMNIST Lines
- Overview of the model, network, and loss
- Train an LSTM on EMNIST

## Follow along

```
git pull
cd lab2/
```

## Intro to the EMNIST Lines dataset

- Synthetic dataset we built for this project
- Sample sentences from Brown corpus
- For each character, sample random EMNIST character and place on a line (with some random overlap)
- Look at: `notebooks/02-look-at-emnist-lines.ipynb`

## Overview of model and loss

- Look at slides for CTC loss
- Look at `networks/line_lstm_ctc.py`
- Look at `models/line_model_ctc.py`

## Train LSTM model with CTC loss

Let's train an LSTM model with CTC loss.

```sh
pipenv run python training/run_experiment.py --save '{"train_args": {"epochs": 16}, "dataset": "EmnistLinesDataset", "model": "LineModelCtc", "network": "line_lstm_ctc"}'
```

or the shortcut `tasks/train_lstm_line_predictor.sh`

## Things to try

If you have time left over, or want to play around with this later on, you can try writing your own non-CTC `line_lstm` network (define it in `text_recognizer/networks/line_lstm.py`).
For example, you could code up an encoder-decoder architecture with attention.
