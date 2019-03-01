# Lab 1: Codebase tour

Before we get started, please run

```
pipenv run python text_recognizer/datasets/emnist.py
```

## Goal of the lab

Familiarize you with the high-level organizational design of the codebase

## Follow along

```
cd lab1/
```

## Project structure

Web backend

```
text_recognizer/
    api/                        # Code for serving predictions as a REST API.
        tests/test_app.py           # Test that predictions are working
        Dockerfile                  # Specificies Docker image that runs the web server.
        __init__.py
        app.py                      # Flask web server that serves predictions.
        serverless.yml              # Specifies AWS Lambda deployment of the REST API.
```

Data (not under version control)

```
    data/                            # Training data lives here
        raw/
            emnist/metadata.toml     # Specifications for downloading data
    experiments/                     # Experiment results live here
        emnist_mlp/                  # Name of the experiment
            models/              
            logs/
```

Experimentation

```
    evaluation/                     # Scripts for evaluating model on eval set.
        evaluate_character_predictor.py
        
    notebooks/                  # For snapshots of initial exploration, before solidfying code as proper Python files.
        01-look-at-emnist.ipynb
```

Convenience scripts

```
    tasks/
        # Deployment
        build_api_docker.sh
        deploy_api_to_lambda.sh

        # Code quality
        lint.sh

        # Tests
        run_prediction_tests.sh
        run_validation_tests.sh
        test_api.sh

        # Training
        train_character_predictor.sh
```

Main model and training code

```
    text_recognizer/                # Package that can be deployed as a self-contained prediction system
        __init__.py

        character_predictor.py      # Takes a raw image and obtains a prediction

        datasets/                   # Code for loading datasets
            __init__.py
            base.py                 # Base class for models - logic for downloading data
            emnist.py
            emnist_essentials.json
            sequence.py 

        models/                     # Code for instantiating models, including data preprocessing and loss functions
            __init__.py
            base.py                 # Base class for models
            character_model.py

        networks/                   # Code for building neural networks (i.e., 'dumb' input->output mappings) used by models
            __init__.py
            mlp.py

        tests/
            support/                        # Raw data used by tests
            test_character_predictor.py     # Test model on a few key examples

        weights/                            # Weights for production model
            CharacterModel_EmnistDataset_mlp_weights.h5

        predict/
            __init__.py
            emnist_mlp.py

        test/                       # Code that tests functionality of the other code
            support/                    # Support files for the tests
                emnist/
                    a.png
                    3.png
            test_emnist_predict.py  # Lightweight test to ensure that the trained emnist_mlp correctly classifies a few data points

        util.py

    tasks/                      # Scripts for running training, downloading data, etc.
        # Training scripts
        train_character_predictor.sh
        # Deployment scripts
        deploy_api_to_lambda.sh
        build_api_docker.sh
        prepare_sample_experiments.sh
        # Tests
        run_prediction_tests.sh
        run_validation_tests.sh
        test_api.sh
        lint.sh

    training/                       # Code for running training experiments and selecting the best model.
        gpu_util_sampler.py
        run_experiment.py           # Parse experiment config and launch training.
        util.py                     # Logic for training a model with a given config
```