# Lab 3: Experiment Management

## Goal of the lab

Get familiar with our experiment running and experiment management tools

## Outline

- Introduction to Weights & Biases
- Running multiple experiments in parallel

## Follow along

```
git pull
cd lab3/
```

## Intro to Weights & Biases

Weights & Biases is an experiment tracking tool that ensures you never lose track of your progress.

### Motivation for W&B

- Keep track of all experiments in one place
- Easily compare runs
- Create reports to document your progress
- Look at results from the whole team

### Let's get started with W&B!

> NOTE: These instructions are optional if you're working in the pre-configured Jupyter hub.

```
pipenv run wandb init
```

You should see something like:

```
? Which team should we use? (Use arrow keys)
> your_username
Manual Entry
```

Select your username.

```
Which project should we use?
> Create New
```

Select `fsdl-text-recognizer-project`.

How to implement W&B in training code?

Look at `training/run_experiment.py` and `training/util.py`

### Your first W&B experiment

Run

```
tasks/train_character_predictor.sh
```

You should see:

```
wandb: Tracking run with wandb version 0.8.15
wandb: Run data is saved locally in wandb/run-20191116_020355-1n7aaz5g
wandb: Syncing run flowing-waterfall-1
```

Click the link to see your run train.

## Running multiple experiments

### Your second W&B experiment

- Open up another terminal (click File->New->Terminal)
- `cd fsdl-text-recognizer-project/lab3`
- launch the same experiment, but with a bigger batch size

```sh
pipenv run python training/run_experiment.py --save '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "mlp", "train_args": {"batch_size": 512}}' --gpu=1
```

Check out both runs at https://app.wandb.ai/<USERNAME>/fsdl-text-recognizer-project

### Automatically running multiple experiments

Desiderata for single-machine parallel experimentation code

- Define multiple experiments and run them simultaneously on all available GPUs
- Run more experiments than GPUs and automatically queue up extras

Let's look at a simple implementation of these:

- Look at `training/prepare_experiments.py`
- Look at `training/gpu_manager.py`

Let's check it out. Run

```
tasks/prepare_sample_experiments.sh
```

or `pipenv run training/prepare_experiments.py training/experiments/sample.json`

You should see the following:

```
pipenv run python training/run_experiment.py --gpu=-1 '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "mlp", "network_args": {"num_layers": 2}, "train_args": {"batch_size": 256}, "experiment_group": "Sample Experiments 2"}'
pipenv run python training/run_experiment.py --gpu=-1 '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "mlp", "network_args": {"num_layers": 4}, "train_args": {"batch_size": 256}, "experiment_group": "Sample Experiments 2"}'
pipenv run python training/run_experiment.py --gpu=-1 '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "lenet", "train_args": {"batch_size": 256}, "experiment_group": "Sample Experiments 2"}'
```

Each line corresponds to an experiment.

Because of this behavior, we can run all these lines in parallel:

```sh
tasks/prepare_sample_experiments.sh | parallel -j2
```

This will run experiments two at a time, and as soon as one finishes, another one will start.

Although you can't see output in the terminal, you can confirm that the experiments are running by going to Weights and Biases.

## More cool things about W&B

- `pipenv run wandb restore <run_id>` will check out the code and the best model
- sample project showing cool plots: https://app.wandb.ai/wandb/face-emotion?view=default
