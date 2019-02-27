# Lab 5: Experimentation

In this lab we will introduce the IAM handwriting dataset, and give you a chance to try out different things, run experiments, and review results on W&B.

## W&B Setup

First, let's set up W&B -- a little differently this time. Run `wandb init` inside of the `lab5` directory. For the team, choose `fsdl` as your "username or team" (it's a team), and name the project `fsdl-text-recognizer-project`.

This will let us all share a project. We'll be able to see all of our runs, including network parameters and performance.

## IAM Lines Dataset

TODO: show raw data first, then explain that we're going to load them all into a single array with the same length for ease of use

This dataset for handwriting recognition has over 13,000 handwritten lines from 657 different writers.

Let's take a look at what it looks like in `notebooks/03-look-at-iam-lines.ipynb`.

The image width is also 952px, as in our synthetic `EmnistLines` dataset.
The maximum output length is 97 characters, however, vs. our 34 characters.

## Training

Let's train with the default params by running `tasks/train_lstm_line_predictor_on_iam.sh`, which runs the follwing command:

```bash
pipenv run python training/run_experiment.py --save '{"dataset": "IamLinesDataset", "model": "LineModelCtc", "network": "line_lstm_ctc"}'
```

This uses the same LSTM with CTC model that we saw succeed on EmnistLines yesterday.
For me, training for 8 epochs gets test set character accuracy of ~40%, and takes about 10 minutes.

Training longer will keep improving: the same settings get to 60% accuracy in 40 epochs: https://app.wandb.ai/sergeyk/fsdl-text-recognizer/runs/a6ucf77m

For the rest of the lab, let's play around with different things and see if we can improve performance quickly.

You can see all of our training runs here: https://app.wandb.ai/fsdl/fsdl-text-recognizer-project
Feel free to peek in on your neighbors!

If you commit and push your code changes, then the run will also be linked to the exact code your ran, which you will be able to review months later if necessary.

## Ideas for things to try

- Change sliding window width/stride
- Not using a sliding window: instead of sliding a LeNet over, you could just run the input through a few conv/pool layers, squeeze out the last (channel) dimension (which should be 0), and input the result into the LSTM. You can play around with the parameters there.
- Change number of LSTM dimensions
- Wrap the LSTM in a Bidirectional() wrapper, which will have two LSTMs read the input forward and backward and concatenate the outputs
- Stack a few layers of LSTMs
- Try to get an all-conv approach to work for faster training
- Add BatchNormalization
- Come up with your own!
