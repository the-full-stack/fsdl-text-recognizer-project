# Lab 4

In this lab, we will get familiar with Weights & Biases, and start using an experiment-running framework that will make it easy to distribute work onto multiple GPUs.

Before getting started, make sure to `git pull` to ensure you have the latest version of the labs and instructions!

## Weights & Biases

In lab4, you'll notice some new lines in `training/run_experiment.py` now: we are now importing and initializing `wandb`, the Weights & Biases package.

Because of this, you need to run  `wandb init`. For the team, you can choose your W&B username, and for the project, you can name it `fsdl-text-recognizer-project`.
Note that `wandb init` will give you some instructions about lines to add to your training script. You can ignore that, as we've already done so as described above.

Now let's test it out with a quick experiment: run `tasks/train_character_predictor.sh`

When the run starts, you'll see some output from `wandb` that looks like this:

```
wandb: Started W&B process version 0.6.17 with PID <xxxx>
wandb: Syncing https://api.wandb.ai/<USERNAME>/fsdl-text-recognizer-project/runs/<xxxxxx>
```

Click on the link in the second line (you may need to scroll up a bit), and check out the progress as your model trains. Don't stay there too long though!
Head back and kick off another experiment -- you'll be able to see both runs happen in parallel.

Let's launch another experiment in a different terminal window, on a different GPU.
Open up another terminal (by clicking File->New->Terminal), `cd fsdl-text-recognizer-project/lab4`, and launch the same experiment, but with a bigger batch size:

```sh
pipenv run python training/run_experiment.py --save '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "mlp", "train_args": {"batch_size": 512}}' --gpu=1
```

Note the `--gpu=1` flag at the end. Because the default gpu index is 0, if we launched this experiment without the flag, it would try allocating on the GPU that's already in use.
With the flag, it runs on a different GPU.

You can now go to https://app.wandb.ai, click into your project, and see both runs happening at the same time!
We'll show you how you can add a chart to visualize all of your training runs.

## Running multiple experiments

It would be nice to be able to define multiple experiments and then just queue them up for training.
We can do that with the `training/prepare_experiments.py` framework.

Let's check it out. Run `tasks/prepare_sample_experiments.sh` or `pipenv run training/prepare_experiments.py training/experiments/sample.json`

You should see the following:

```
pipenv run python training/run_experiment.py --gpu=-1 '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "mlp", "network_args": {"num_layers": 2}, "train_args": {"batch_size": 256}, "experiment_group": "Sample Experiments 2"}'
pipenv run python training/run_experiment.py --gpu=-1 '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "mlp", "network_args": {"num_layers": 4}, "train_args": {"batch_size": 256}, "experiment_group": "Sample Experiments 2"}'
pipenv run python training/run_experiment.py --gpu=-1 '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "lenet", "train_args": {"batch_size": 256}, "experiment_group": "Sample Experiments 2"}'
```

Each line corresponds to an experiment.
The `--gpu=-1` flag makes use of a new file in this lab: `training/gpu_manager.py`, which finds an unused GPU, or waits until one is available.

Because of this behavior, we can run all these lines in parallel:

```sh
tasks/prepare_sample_experiments.sh | parallel -j2
```

This will run experiments two at a time, and as soon as one finishes, another one will start.

Although you can't see output in the terminal, you can confirm that the experiments are running by going to Weights and Biases.

## More cool things about W&B

- `pipenv run wandb restore <run_id>` will check out the code and the best model
- sample project showing cool plots: https://app.wandb.ai/wandb/face-emotion?view=default
