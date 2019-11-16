# Lab 4: Real handwriting dataset and experimentation

In this lab we will introduce the IAM handwriting dataset, and give you a chance to try out different things, run experiments, and review results on W&B.

## Goal of the lab

- Introduce IAM handwriting dataset
- Try some ideas & review results on W&B
- See who can get the best score :)
- Automate trials with hyper-parameter sweeps

## Outline

- Intro to IAM datasets
- Train a baseline model
- Try your own ideas
- Run a sweep

## Follow along

```
git pull
cd lab4/
```

## IAM Lines Dataset

- Look at `notebooks/03-look-at-iam-lines.ipynb`.

## Training individual runs

Let's train with the default params by running `tasks/train_lstm_line_predictor_on_iam.sh`, which runs the following command:

```bash
pipenv run python training/run_experiment.py --save '{"dataset": "IamLinesDataset", "model": "LineModelCtc", "network": "line_lstm_ctc"}'
```

This uses our LSTM with CTC model. 8 epochs gets accuracy of 40% and takes about 10 minutes.

Training longer will keep improving: the same settings get to 60% accuracy in 40 epochs.

## Configuring sweeps

Sweeps enable automated trials of hyper-parameters. W&B provides built in support for running [sweeps](https://docs.wandb.com/library/sweeps). We've setup an initial configuration file for sweeps in `training/sweeps.yaml`. It performs a basic grid search across 3 parameters. There are lots of different [configuration options](https://docs.wandb.com/library/sweeps/configuration) for defining more complex sweeps. Anytime you modify this configuration you'll need to create a sweep in wandb by running:

```bash
pipenv run wandb sweep training/sweep.yaml
```

```text
Creating sweep from: sweep.yaml
Create sweep with ID: 0nnj74vx
```

Take note of the 8 character ID that's returned by this command. It's best to store this in an environment variable by running `export SWEEP_ID=0nnj74vx`. W&B sweeps work by running a command and passing arguments into it. We wrote a wrapper at `training/run_sweep.py` to convert these arguments into a JSON config object.

> NOTE: Be sure to edit **config_defaults** in `training/run_sweep.py` if you train on different datasets or models.

To run a sweep you can start multiple agents to query for and run the next set of parameters. This is done with the command:

```bash
pipenv run wandb agent $SWEEP_ID
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
