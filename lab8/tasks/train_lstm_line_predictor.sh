#!/bin/bash
python training/run_experiment.py --save '{"dataset": "EmnistLinesDataset", "dataset_args": {"categorical_format": true}, "model": "LineModelCtc", "network": "line_lstm_ctc"}' "$@"
