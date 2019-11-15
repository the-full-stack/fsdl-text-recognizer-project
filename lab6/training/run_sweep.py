# pylint: skip-file
import os
import sys
import json
from typing import Tuple
from ast import literal_eval

default_config = {
    "dataset": "EmnistLinesDataset",
    "dataset_args": {
        "max_overlap": 0.4,
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


def args_to_json(default_config: dict, preserve_args: list = ["gpu", "save"]) -> Tuple[dict, list]:
    """Converts command line arguments to nested config values

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


if __name__ == "__main__":
    for key, val in os.environ.items():
        os.putenv(key, val)
    config, args = args_to_json(default_config)
    os.system("python training/run_experiment.py {} '{}'".format((" ").join(args), json.dumps(config)))  # nosec
