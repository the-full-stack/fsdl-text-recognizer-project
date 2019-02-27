# Lab 1: Plumbing

To start with lab1, `cd lab1`.

## Tour of the codebase

I am going to give you a tour of the codebase, but before we get started, please run `pipenv run python text_recognizer/datasets/emnist.py` to kick off download of the EMNIST dataset.
This can take a few minutes.

We will cover:
- Overall layout: datasets, models, networks, weights, predictor, and training harness
- EmnistDataset
- DatasetSequence
- CharacterModel
- mlp
- CharacterPredictor
- run_experiment.py
- training/util.py

### Project Structure

```
text_recognizer/
    api/                        # Code for serving predictions as a REST API.
        app.py                      # Flask web server that serves predictions.
        Dockerfile                  # Specificies Docker image that runs the web server.
        serverless.yml              # Specifies AWS Lambda deployment of the REST API.

    data/                       # Data for training. Not under version control.
        raw/                        # The raw data source. Perhaps from an external source, perhaps from your DBs. Contents of this should be re-creatable via scripts.
            emnist-matlab.zip
        processed/                  # Data in a format that can be used by our Dataset classses.
            byclass.npz

    experiments/                # Not under code version control.
        emnist_mlp/                 # Name of the experiment
            models/
            logs/

    notebooks/                  # For snapshots of initial exploration, before solidfying code as proper Python files.
        00-download-emnist.ipynb    # Naming pattern is <order>-<initials>-<description>.ipynb
        01-train-emnist-mlp.ipynb

    text_recognizer/            # Package that can be deployed as a self-contained prediction system.
        __init__.py

        datasets/                   # Code for loading datasets
            __init__.py
            emnist.py

        models/                     # Code for instantiating models, including data preprocessing and loss functions
            __init__.py
            emnist_mlp.py               # Code
            emnist_mlp.h5               # Learned weights
            emnist_mlp.config           # Experimental config that led to the learned weights

        predict/
            __init__.py
            emnist_mlp.py

        test/                       # Code that tests functionality of the other code.
            support/                    # Support files for the tests
                emnist/
                    a.png
                    3.png
            test_emnist_predict.py  # Lightweight test to ensure that the trained emnist_mlp correctly classifies a few data points.

    tasks/
        train_emnist_mlp.py
        run_emnist_mlp_experiments.py
        update_model_with_best_experiment.py
        evaluate_emnist_mlp_model.py
        tasks/deploy_web_server_to_aws.py

    train/                       # Code for running training experiments and selecting the best model.
        run_experiment.py           # Script for running a training experiment.
        gpu_manager.py              # Support script for distributing work onto multiple GPUs.
        select_best_model.py        # Script for selecting the best model out of multiple experimental instances.

    Pipfile
    Pipfile.lock
    README.md
    setup.py
```

### Pipenv

Pipenv is useful in order to precisely specify dependencies.
TODO: explain that want to stay up to date with packages, but only update them intentionally, not randomly. Explain sync vs install.

```
# Workhorse command when adding another dependency
pipenv install --dev --keep-outdated

# Periodically, update all versions
pipenv install --dev

# For deployment, no need to install dev packages
pipenv install
```

## A look at the data

EMNIST stands for Extended Mini-NIST :)
It has many samples of all English letters and digits, all nicely cropped and presented in the MNIST format.
We have a notebook showing what it looks like: `notebooks/01-look-at-emnist.ipynb`

## Training the network

You will have to add a little bit of code to `text_recognizer/networks/mlp.py` before being able to train.
When you finish writing your code, you can train a canonical model and save the weights.

You can run the shortcut command `tasks/train_character_predictor.sh`, which runs the following:

```sh
pipenv run training/run_experiment.py --save '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "mlp",  "train_args": {"batch_size": 256}}'
```

It will take a couple of minutes to train your model.

Just for fun, you could also try a larger MLP, with a smaller batch size:

```sh
pipenv run training/run_experiment.py '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "mlp", "network_args": {"num_layers": 8}, "train_args": {"batch_size": 128}}'
```

## Subsampling the dataset

It is very useful to be able to subsample the dataset for quick experiments.
This is possibe by passing `subsample_fraction=0.1` (or some other fraction) at dataset initialization, or in `dataset_args` in the `run_experiment.py` dictionary, for example:

```sh
pipenv run training/run_experiment.py '{"dataset": "EmnistDataset", "dataset_args": {"subsample_fraction": 0.1}, "model": "CharacterModel", "network": "mlp"}'
```

## Testing

Your network is trained, but you need to write a bit more code to get the `CharacterModel` to use it to predict.
Open up `text_recognizer/models/character_model.py` and write some code there to make it work.
You can test that it works by running

```sh
pipenv run pytest -s text_recognizer/tests/test_character_predictor.py
```

Or, use the shorthand `tasks/run_prediction_tests.sh`

Testing should finish quickly.
