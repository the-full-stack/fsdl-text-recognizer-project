#!/bin/bash
python training/run_experiment.py --save '{"dataset": "EmnistLinesDataset", "model": "LineModelCtc", "network": "line_lstm_ctc"}'
