#!/usr/bin/env python
"""Script to run an experiment."""
import argparse
import json
import importlib
from typing import Dict
import os

# Hide lines below until Lab 3
import wandb

from training.gpu_manager import GPUManager

# Hide lines above until Lab 3
from training.util import train_model

DEFAULT_TRAIN_ARGS = {"batch_size": 64, "epochs": 16}


def run_experiment(experiment_config: Dict, save_weights: bool, gpu_ind: int, use_wandb: bool = True):
    """
    Run a training experiment.

    Parameters
    ----------
    experiment_config (dict)
        Of the form
        {
            "dataset": "EmnistLinesDataset",
            "dataset_args": {
                "max_overlap": 0.4,
                "subsample_fraction": 0.2
            },
            "model": "LineModel",
            "network": "line_cnn_all_conv",
            "network_args": {
                "window_width": 14,
                "window_stride": 7
            },
            "train_args": {
                "batch_size": 128,
                "epochs": 10
            }
        }
    save_weights (bool)
        If True, will save the final model weights to a canonical location (see Model in models/base.py)
    gpu_ind (int)
        specifies which gpu to use (or -1 for first available)
    use_wandb (bool)
        sync training run to wandb
    """
    print(f"Running experiment with config {experiment_config} on GPU {gpu_ind}")

    datasets_module = importlib.import_module("text_recognizer.datasets")
    dataset_class_ = getattr(datasets_module, experiment_config["dataset"])
    dataset_args = experiment_config.get("dataset_args", {})
    dataset = dataset_class_(**dataset_args)
    dataset.load_or_generate_data()
    print(dataset)

    models_module = importlib.import_module("text_recognizer.models")
    model_class_ = getattr(models_module, experiment_config["model"])

    networks_module = importlib.import_module("text_recognizer.networks")
    network_fn_ = getattr(networks_module, experiment_config["network"])
    network_args = experiment_config.get("network_args", {})
    model = model_class_(
        dataset_cls=dataset_class_, network_fn=network_fn_, dataset_args=dataset_args, network_args=network_args,
    )
    print(model)

    experiment_config["train_args"] = {
        **DEFAULT_TRAIN_ARGS,
        **experiment_config.get("train_args", {}),
    }
    experiment_config["experiment_group"] = experiment_config.get("experiment_group", None)
    experiment_config["gpu_ind"] = gpu_ind

    # Hide lines below until Lab 3
    if use_wandb:
        wandb.init(config=experiment_config)
    # Hide lines above until Lab 3

    train_model(
        model,
        dataset,
        epochs=experiment_config["train_args"]["epochs"],
        batch_size=experiment_config["train_args"]["batch_size"],
        use_wandb=use_wandb,
    )
    score = model.evaluate(dataset.x_test, dataset.y_test)
    print(f"Test evaluation: {score}")

    # Hide lines below until Lab 3
    if use_wandb:
        wandb.log({"test_metric": score})
    # Hide lines above until Lab 3

    if save_weights:
        model.save_weights()


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="Provide index of GPU to use.")
    parser.add_argument(
        "--save",
        default=False,
        dest="save",
        action="store_true",
        help="If true, then final weights will be saved to canonical, version-controlled location",
    )
    parser.add_argument(
        "experiment_config",
        type=str,
        help='Experimenet JSON (\'{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "mlp"}\'',
    )
    parser.add_argument(
        "--nowandb", default=False, action="store_true", help="If true, do not use wandb for this run",
    )
    args = parser.parse_args()
    return args


def main():
    """Run experiment."""
    args = _parse_args()
    # Hide lines below until Lab 3
    if args.gpu < 0:
        gpu_manager = GPUManager()
        args.gpu = gpu_manager.get_free_gpu()  # Blocks until one is available
    # Hide lines above until Lab 3

    experiment_config = json.loads(args.experiment_config)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    run_experiment(experiment_config, args.save, args.gpu, use_wandb=not args.nowandb)


if __name__ == "__main__":
    main()
