# Full Stack Deep Learning Labs

Welcome!

Project developed during lab sessions of the [Full Stack Deep Learning Bootcamp](https://fullstackdeeplearning.com).

- We will build a handwriting recognition system from scratch, and deploy it as a web service.
- Uses Keras, but designed to be modular, hackable, and scalable
- Provides code for training models in parallel and store evaluation in Weights & Biases
- We will set up continuous integration system for our codebase, which will check functionality of code and evaluate the model about to be deployed.
- We will package up the prediction system as a REST API, deployable as a Docker container.
- We will deploy the prediction system as a serverless function to Amazon Lambda.
- Lastly, we will set up monitoring that alerts us when the incoming data distribution changes.

## Schedule for the Spring 2019 Bootcamp

- First session (90 min)
  - Introduction to the labs.
  - Lab 0 (15 min): gather handwriting data and get setup
  - Lab 1 (45 min): Project structure. Intro to EMNIST. Simple EMNIST MLP model, training, testing.
  - Lab 2 (20 min): Introduce approach of synthetic data, go through EMNIST lines, and then CNN solution for EMNIST Lines
  - Lab 3 (10 min): LSTM+CTC solution for EMNIST Lines
- Second session (60 min)
  - Lab 4 (20 min): Weights & Biases + parallel experiments
  - Lab 5 (40 min): IAM Lines and experimentation time (launch a bunch of experiments, leave running overnight in a shared W&B)
- Third session (90 min)
  - Review results from the class on W&B
  - Lab 6 (60 min) line detection task
  - Lab 7 (30 min) data labeling
    - Go through data versioning and even have a little labeling interface for fresh data that they generated on the first day
- Fourth session (75 min)
  - Lab 8 (75 min) testing & deployment

# Setup

## 1. Setup a JupyterLab instance

- Go to https://app.wandb.ai/profile
- Enter the code we will share with you at the session into Access Code field.
- You should be dropped into a JuypyterLab instance with 2 GPUs to use for labs.

*From now on, do everything else in that instance*

## 2. Check out the repo

Open a shell in your JupyterLab instance and run

```sh
git clone https://github.com/gradescope/fsdl-text-recognizer-project.git
cd fsdl-text-recognizer-project
```

If you already have the repo in your home directory, then simply go into it and pull the latest version.

```sh
cd fsdl-text-recognizer-project
git pull origin master
```

## 3. Set up the Python environment

Run

```sh
pipenv install --dev
```

From now on, precede commands with `pipenv run` to make sure they use the correct
environment.

# Ready

Now you should be setup for the labs. The instructions for each lab are in readme files in their folders.

You will notice that there are solutions for all the labs right here in the repo, too.
If you get stuck, you are welcome to take a look!
