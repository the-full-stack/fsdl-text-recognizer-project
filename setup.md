# Setup

## 1. Check out the repo

You should already have the repo in your home directory. Go into it and make sure you have the latest.

```sh
cd fsdl-text-recognizer-project
git pull origin master
```

If not, open a shell in your JupyterLab instance and run

```sh
git clone https://github.com/full-stack-deep-learning/fsdl-text-recognizer-project.git
cd fsdl-text-recognizer-project
```

## 2. Set up the Python environment

Run `conda env create` to create an environment called `fsdl-text-recognizer`, as defined in `environment.yml`.
This environment will provide us with the right Python version as well as the CUDA and CUDNN libraries.
We will install Python libraries using `pip-sync`, however, which will let us do three nice things:

1. Separate out dev from production dependencies (`requirements-dev.in` vs `requirements.in`).
2. Have a lockfile of exact versions for all dependencies (the auto-generated `requirements-dev.txt` and `requirements.txt`).
3. Allow us to easily deploy to targets that may not support the `conda` environment.

So, after running `conda env create`, activate the new environment and install the requirements:

```sh
conda activate fsdl-text-recognizer
pip-compile requirements.in && pip-compile requirements-dev.in
pip-sync requirements.txt requirements-dev.txt
```

The latter two commands are provided for you as `tasks/update_requirements.sh` and `tasks/sync_requirements.sh`.

Now, every time you work in this directory, make sure to start your session with `conda activate fsdl-text-recognizer`.

## 3. Kick off a command

Before we get started, please run a command that will take a little bit of time to execute.

```sh
git pull
cd lab1/
python text_recognizer/datasets/emnist_dataset.py
cd ..
```

# Ready

Now you should be setup for the labs. The instructions for each lab are in readme files in their folders.

You will notice that there are solutions for all the labs right here in the repo, too.
If you get stuck, you are welcome to take a look!
