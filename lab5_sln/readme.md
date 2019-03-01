# Lab 5: Experimentation

In this lab we will introduce the IAM handwriting dataset, and give you a chance to try out different things, run experiments, and review results on W&B.

## Goal of the lab
- Introduce IAM handwriting dataset
- Try some ideas & review results on W&B
- See who can get the best score :)

## Outline
- Intro to IAM datasets
- Train a baseline model
- Try your own ideas

## Follow along

```
cd lab5_soln/
wandb init
   - team: fsdl
   - project: fsdl-text-recognizer-project
```

## IAM Lines Dataset

- Look at `notebooks/03-look-at-iam-lines.ipynb`.

## Training

Let's train with the default params by running `tasks/train_lstm_line_predictor_on_iam.sh`, which runs the follwing command:

```bash
pipenv run python training/run_experiment.py --save '{"dataset": "IamLinesDataset", "model": "LineModelCtc", "network": "line_lstm_ctc"}'
```

This uses our LSTM with CTC model. 8 epochs gets accuracy of 40% and takes about 10 minutes.

Training longer will keep improving: the same settings get to 60% accuracy in 40 epochs. 

## Ideas for things to try

For the rest of the lab, let's play around with different things and see if we can improve performance quickly.

You can see all of our training runs here: https://app.wandb.ai/fsdl/fsdl-text-recognizer-project
Feel free to peek in on your neighbors!

If you commit and push your code changes, then the run will also be linked to the exact code your ran, which you will be able to review months later if necessary.


- Change sliding window width/stride
- Not using a sliding window: instead of sliding a LeNet over, you could just run the input through a few conv/pool layers, squeeze out the last (channel) dimension (which should be 0), and input the result into the LSTM. You can play around with the parameters there.
- Change number of LSTM dimensions
- Wrap the LSTM in a Bidirectional() wrapper, which will have two LSTMs read the input forward and backward and concatenate the outputs
- Stack a few layers of LSTMs
- Try to get an all-conv approach to work for faster training
- Add BatchNormalization
- Come up with your own!
