#!/bin/bash
pipenv run python training/run_experiment.py --save '{"dataset": "IamParagraphsDataset", "model": "LineDetectorModel", "network": "fcn"}'
