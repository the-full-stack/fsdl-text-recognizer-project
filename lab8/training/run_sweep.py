"""W&B Sweep Functionality."""
import os
import signal
import subprocess
import sys
import json
from typing import Tuple
from ast import literal_eval

DEFAULT_CONFIG = {
    "dataset": "IamLinesDataset",
    "dataset_args": {"subsample_fraction": 0.33},
    "model": "LineModel",
    "network": "line_lstm_ctc",
    "train_args": {"batch_size": 128, "epochs": 10},
}


def args_to_json(default_config: dict, preserve_args: tuple = ("gpu", "save")) -> Tuple[dict, list]:
    """Convert command line arguments to nested config values

    i.e. run_sweep.py --dataset_args.foo=1.7

    {
        "dataset_args": {
            "foo": 1.7
        }
    }

    """
    args = []
    config = default_config.copy()
    key, val = None, None
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, val = arg.split("=")
        elif key:
            val = arg
        else:
            key = arg
        if key and val:
            parsed_key = key.lstrip("-").split(".")
            if parsed_key[0] in preserve_args:
                args.append("--{}={}".format(parsed_key[0], val))
            else:
                nested = config
                for level in parsed_key[:-1]:
                    nested[level] = config.get(level, {})
                    nested = nested[level]
                try:
                    # Convert numerics to floats / ints
                    val = literal_eval(val)
                except ValueError:
                    pass
                nested[parsed_key[-1]] = val
            key, val = None, None
    return config, args


def main():
    config, args = args_to_json(DEFAULT_CONFIG)
    env = {k: v for k, v in os.environ.items() if k not in ("WANDB_PROGRAM", "WANDB_ARGS")}
    # pylint: disable=subprocess-popen-preexec-fn
    run = subprocess.Popen(
        ["python", "training/run_experiment.py", *args, json.dumps(config)], env=env, preexec_fn=os.setsid,
    )  # nosec
    signal.signal(signal.SIGTERM, lambda *args: run.terminate())
    run.wait()


if __name__ == "__main__":
    main()
