#!/bin/bash
pipenv run python training/run_experiment.py --save '{"dataset": "IamLinesDataset", "model": "LineModelCtc", "network": "line_lstm_ctc", "network_args": {"window_width": 12, "window_stride": 5}}'
