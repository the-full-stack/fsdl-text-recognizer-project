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

## Testing

Your network is trained, but you need to write a bit more code to get the `CharacterModel` to use it to predict.
Open up `text_recognizer/models/character_model.py` and write some code there to make it work.
You can test that it works by running

```sh
pipenv run pytest -s text_recognizer/tests/test_character_predictor.py
```

Or, use the shorthand `tasks/run_prediction_tests.sh`

Testing should finish quickly.

## Submitting to Gradescope

Before submitting to Gradescope, add your trained weights, commit, and push your changes:

```sh
git add text_recognizer
git commit -am "my lab1 work"
git push mine master
```

Now go to https://gradescope.com/courses/21098 and click on Lab 1.
Select the Github submission option, and there select your fork of the `fsdl-text-recognizer-project` repo and the master branch, and click Submit.
Don't forget to enter a name for the leaderboard :)

The autograder treats code that is in `lab1/text_recognizer` as your submission, so make sure your code is indeed there.

The autograder should finish in <1 min, and display the results.
Your name will show up in the Leaderboard.
