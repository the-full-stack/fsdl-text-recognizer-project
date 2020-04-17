# Lab 4: Real Handwriting Dataset and Experiment Management

In this lab we will introduce the IAM handwriting dataset, and give you a chance to try out different things, run experiments, and review results on W&B.

## Goal of the lab

- Introduce IAM handwriting dataset
- Introduction to Weights & Biases
- Running multiple experiments in parallel
- Automate trials with hyper-parameter sweeps
- Try some ideas & review results on W&B

## Follow along

```
git pull
cd lab4
```

## IAM Lines Dataset

- Look at `notebooks/03-look-at-iam-lines.ipynb`.

## Training individual runs

Let's train with the default params by running `tasks/train_lstm_line_predictor_on_iam.sh`, which runs the following command:

```bash
python training/run_experiment.py --save '{"dataset": "IamLinesDataset", "model": "LineModelCtc", "network": "line_lstm_ctc"}'
```

This uses our LSTM with CTC model. 8 epochs gets accuracy of 40% and takes about 10 minutes.

Training longer will keep improving: the same settings get to 60% accuracy in 40 epochs.

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
wandb init
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
python training/run_experiment.py --save '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "mlp", "train_args": {"batch_size": 512}}' --gpu=1
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

or `training/prepare_experiments.py training/experiments/sample.json`

You should see the following:

```
python training/run_experiment.py --gpu=-1 '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "mlp", "network_args": {"num_layers": 2}, "train_args": {"batch_size": 256}, "experiment_group": "Sample Experiments 2"}'
python training/run_experiment.py --gpu=-1 '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "mlp", "network_args": {"num_layers": 4}, "train_args": {"batch_size": 256}, "experiment_group": "Sample Experiments 2"}'
python training/run_experiment.py --gpu=-1 '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "lenet", "train_args": {"batch_size": 256}, "experiment_group": "Sample Experiments 2"}'
```

Each line corresponds to an experiment.

Because of this behavior, we can run all these lines in parallel:

```sh
tasks/prepare_sample_experiments.sh | parallel -j2
```

This will run experiments two at a time, and as soon as one finishes, another one will start.

Although you can't see output in the terminal, you can confirm that the experiments are running by going to Weights and Biases.

## More cool things about W&B

- `wandb restore <run_id>` will check out the code and the best model
- sample project showing cool plots: https://app.wandb.ai/wandb/face-emotion?view=default

## Configuring sweeps

Sweeps enable automated trials of hyper-parameters. W&B provides built in support for running [sweeps](https://docs.wandb.com/library/sweeps). We've setup an initial configuration file for sweeps in `training/sweeps.yaml`. It performs a basic grid search across 3 parameters. There are lots of different [configuration options](https://docs.wandb.com/library/sweeps/configuration) for defining more complex sweeps. Anytime you modify this configuration you'll need to create a sweep in wandb by running:

```bash
wandb sweep training/sweep.yaml
```

```text
Creating sweep from: sweep.yaml
Create sweep with ID: 0nnj74vx
```

Take note of the 8 character ID that's returned by this command. It's best to store this in an environment variable by running `export SWEEP_ID=0nnj74vx`. W&B sweeps work by running a command and passing arguments into it. We wrote a wrapper at `training/run_sweep.py` to convert these arguments into a JSON config object.

> NOTE: Be sure to edit **config_defaults** in `training/run_sweep.py` if you train on different datasets or models.

To run a sweep you can start multiple agents to query for and run the next set of parameters. This is done with the command:

```bash
wandb agent $SWEEP_ID
```

This will print a url to W&B which you can use to monitor or control the sweep.

### Stopping a sweep

If you choose the **random** sweep strategy, the agent will run forever. Our **grid** search strategy will stop once all options have been tried. You can stop a sweep from the W&B UI, or directly from the terminal. Hitting CTRL-C once will prevent the agent from running a new experiment but allow the current experiment to finish. Hitting CTRL-C again will kill the current running experiment.

## Ideas for things to try

For the rest of the lab, let's play around with different things and see if we can improve performance quickly.

You can see all of our training runs here: https://app.wandb.ai/fsdl/fsdl-text-recognizer-nov2019
Feel free to peek in on your neighbors!

If you commit and push your code changes, then the run will also be linked to the exact code your ran, which you will be able to review months later if necessary.

- Change sliding window width/stride
- Not using a sliding window: instead of sliding a LeNet over, you could just run the input through a few conv/pool layers, squeeze out the last (channel) dimension (which should be 0), and input the result into the LSTM. You can play around with the parameters there.
- Change number of LSTM dimensions
- Wrap the LSTM in a Bidirectional() wrapper, which will have two LSTMs read the input forward and backward and concatenate the outputs
- Stack a few layers of LSTMs
- Try to get an all-conv approach to work for faster training
- Add BatchNormalization
- Play around with learning rate. In order to launch experiments with different learning rates, you will have to implement something in `training/run_experiment.py` and `text_recognizer/datasets/base.py`
- Train on EmnistLines and fine-tune on IamLines. In order to do that, you might want to implement a model wrapper class that can take multiple datasets.
- Come up with your own!
