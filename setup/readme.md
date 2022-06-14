# Setup

Deep learning requires access to accelerated computation hardware.
Most commonly, those are NVIDIA GPUs or Google TPUs.

If you have access to a computer that has an NVIDIA GPU and runs Linux, you're welcome to [set it up](#Local) for local user.

If you don't, you can compute using [Google Colab](#Colab).

## Colab

Google Colab is a great way to get access to fast GPUs for free.
All you need is a Google account (Berkeley accounts will work).

Go to [https://colab.research.google.com](https://colab.research.google.com), and create a New Notebook.

Connect your new notebook to a GPU runtime by doing Runtime > Change Runtime type > GPU.
![](colab_runtime.png)

Now, run `!nvidia-smi` in the first cell (press Shift+Enter to run a cell).
You should see a table showing your precious GPU :)

Now, paste the following into a cell and run it:

```
# FSDL Spring 2022 Setup
!git clone https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2022-labs
%cd fsdl-text-recognizer-2022-labs
!pip install -r requirements/prod.in
!sed 1d requirements/dev.in | grep -v "#" | xargs pip install
%env PYTHONPATH=.:$PYTHONPATH
```

This will check out our lab repository, `cd` into it, and install required packages.
It also allows Python to find packages in the current working directory.

Now we can enter the `lab01` directory and make sure things work:
![](colab_lab1.png)

### Colab Pro

You may be interested in signing up for [Colab Pro](https://colab.research.google.com/signup).

For $10/month, you get priority access to fast GPUs (e.g. [P100 vs K80](https://www.xcelerit.com/computing-benchmarks/insights/nvidia-p100-vs-k80-gpu/)) and TPUs, a 24h rather than 12h maximum runtime, and more RAM.

### VSCode on Google Colab

(This is advanced and not necessary -- feel free to skip for now!).

It is possible to use the VSCode interface in Colab.

Open a Colab notebook (or continue in the one you may already have going), make sure you're connected to the runtime you want, and run this in a cell:

```
# Launch VSCode server
!curl -fsSL https://code-server.dev/install.sh | sh
!nohup code-server --port 9000 --auth none &

# Tunnel its IP using ngrok
!pip install pyngrok
from pyngrok import ngrok
# ngrok.set_auth_token("get from https://dashboard.ngrok.com/auth/your-authtoken, if you want to pay $10/month for a little bit better service")
url = ngrok.connect(9000)
print(url)
```

This is what you should see:
![](colab_vscode.png)

Clicking the ngrok link takes you to a web VSCode interface:
![](colab_vscode_2.png)

You can sign up for a paid version of ngrok ($10/month) in order to get `https` tunneling and a slightly nicer experience.

## Local

Setting up a machine you can sit in front of or SSH into is easy.

### Summary

- `environment.yml` specifies Python and optionally CUDA/CUDNN
- `make conda-update` creates/updates a virtual environment
- `conda activate fsdl-text-recognizer-2022` activates the virtual environment
- `requirements/prod.in` and `requirements/dev.in` specify core Python packages in that environment
- `make pip-tools` resolves all other Python dependencies and installs them
- `export PYTHONPATH=.:$PYTHONPATH` makes the current directory visible on your Python path -- add it to your `~/.bashrc` and `source ~/.bashrc`

### 1. Check out the repo

```
git clone https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2022-labs.git
cd fsdl-text-recognizer-2022-labs
```

### 2. Set up the Python environment

We use [`conda`](https://docs.conda.io/en/latest/miniconda.html) for managing Python and CUDA versions, and [`pip-tools`](https://github.com/jazzband/pip-tools) for managing Python package dependencies.

We add a `Makefile` for making setup dead-simple.

#### First: Install the Python + CUDA environment using Anaconda

**Note**: this step is not necessary if you are using the GCP AI Platform Notebooks instance.

Conda is an open-source package management system and environment management system that runs on Windows, macOS, and Linux.

To install `conda`, follow instructions at https://docs.conda.io/en/latest/miniconda.html.

Note that you will likely need to close and re-open your terminal.
Afterward, you should have ability to run the `conda` command in your terminal.

Run `make conda-update` to create an environment called `fsdl-text-recognizer-2022`, as defined in `environment.yml`.
This environment will provide us with the right Python version as well as the CUDA and CUDNN libraries.

If you edit `environment.yml`, just run `make conda-update` again to get the latest changes.

Next, activate the conda environment.

```sh
conda activate fsdl-text-recognizer-2022
```

**IMPORTANT**: every time you work in this directory, make sure to start your session with `conda activate fsdl-text-recognizer-2022`.

#### Next: install Python packages

Next, install all necessary Python packages by running `make pip-tools`

Using `pip-tools` lets us do three nice things:

1. Separate out dev from production dependencies (`requirements-dev.in` vs `requirements.in`).
2. Have a lockfile of exact versions for all dependencies (the auto-generated `requirements-dev.txt` and `requirements.txt`).
3. Allow us to easily deploy to targets that don't support the `conda` environment, like Colab.

If you add, remove, or need to update versions of some requirements, edit the `.in` files, and simply run `make pip-tools` again.

#### Set PYTHONPATH

Last, run `export PYTHONPATH=.` before executing any commands later on, or you will get errors like `ModuleNotFoundError: No module named 'text_recognizer'`.

In order to not have to set `PYTHONPATH` in every terminal you open, just add that line as the last line of the `~/.bashrc` file using a text editor of your choice (e.g. `nano ~/.bashrc`)
