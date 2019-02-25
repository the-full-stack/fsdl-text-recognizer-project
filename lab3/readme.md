# Lab 3

In this lab we'll keep working with the EmnistLines dataset.

We will be implementing LSTM model with CTC loss.
CTC loss needs to be implemented kind of strangely in Keras: we need to pass in all required data to compute the loss as inputs to the network (including the true label).
This is an example of a multi-input / multi-output network.

The relevant files to review are `models/line_model_ctc.py`, which shows the batch formatting that needs to happen for the CTC loss to be computed inside of the network, `networks/line_lstm_ctc.py`, which has the network definition.

## Train LSTM model with CTC loss

You need to write code in `networks/line_lstm_ctc.py` to make training work.
Training can be done via

```sh
pipenv run python training/run_experiment.py --save '{"dataset": "EmnistLinesDataset", "model": "LineModelCtc", "network": "line_lstm_ctc"}'
```

or the shortcut `tasks/train_lstm_line_predictor.sh`

## Make sure the model is able to predict

You will also need to write some code in `models/line_model_ctc.py` to predict on images.
After that, you should see tests pass when you run

```sh
pipenv run pytest -s text_recognizer/tests/test_line_predictor.py
```

Or you can do `tasks/run_prediction_tests.sh`, which will also run the CharacterModel tests.

## Submitting to Gradescope

Before submitting to Gradescope, add your trained weights to the repo, commit and push your changes:

```sh
git add text_recognizer
git commit -am "my lab3 work"
git push mine master
```

Now go to https://gradescope.com/courses/21098 and click on Lab 3.
Select your fork of the `fsdl-text-recognizer-project` repo, and click Submit.
Don't forget to enter a name for the leaderboard :)

The autograder treats code that is in `lab3/text_recognizer` as your submission, so make sure your code is indeed there.

The autograder should finish in a couple of minutes, and display the results.
Your name will show up in the Leaderboard.

While you wait for the autograder to complete, feel free to try some experiments!

## Things to try

If you have time left over, or want to play around with this later on, you can try writing your own non-CTC `line_lstm` network (define it in `text_recognizer/networks/line_lstm.py`).
For example, you could code up an encoder-decoder architecture with attention.
