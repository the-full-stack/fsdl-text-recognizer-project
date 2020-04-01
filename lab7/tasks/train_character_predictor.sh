#!/bin/bash
python training/run_experiment.py --save '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "mlp", "train_args": {"batch_size": 256}}'
