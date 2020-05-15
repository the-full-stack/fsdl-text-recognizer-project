# Lab 3: Using a sequence model for line text recognition

## Goal of the lab

Use sequence modeling to be able to handle overlapping characters (input sequence no longer maps neatly onto output sequence).

## Outline

- Overview of the model, network, and loss
- Train an LSTM on EMNIST

## Follow along

```
git pull
cd lab3
```

## Overview of model and loss

- Look at slides for CTC loss
- Look at `networks/line_lstm_ctc.py`
- Look at `models/line_model_ctc.py`

## Train LSTM model with CTC loss

Let's train an LSTM model with CTC loss.

```sh
python training/run_experiment.py --save '{"train_args": {"epochs": 16}, "dataset": "EmnistLinesDataset", "dataset_args": {"categorical_format": true}, "model": "LineModelCtc", "network": "line_lstm_ctc"}'
```

or the shortcut `tasks/train_lstm_line_predictor.sh`

## Things to try

If you have time left over, or want to play around with this later on, you can try writing your own non-CTC `line_lstm` network (define it in `text_recognizer/networks/line_lstm.py`).
For example, you could code up an encoder-decoder architecture with attention.

## Addendum: Transformer-based model

We have updated the data format for `EmnistLinesDataset`, so make sure to

```
git pull
rm -r fsdl-text-recognizer/data/processed/emnist_lines/
```

Go through `notebooks/02c-transformer.ipynb` and the files it imports to see a Transformer-based approach to this problem.
