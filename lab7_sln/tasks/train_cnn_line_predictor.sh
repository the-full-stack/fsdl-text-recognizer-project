#!/bin/bash
pipenv run python training/run_experiment.py '{"dataset": "EmnistLinesDataset", "model": "LineModel", "network": "line_cnn_all_conv"}'
