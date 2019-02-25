# Full Stack Deep Learning Labs

Welcome!

## Schedule for the Spring 2019 Bootcamp

- First session (90 min)
  - Lab 0 (15 min): gather handwriting data and get setup
  - Lab 1 (45 min): task intro, intro to IAM, intro to EMNIST, project structure explained on simple EMNIST MLP model, linting in editor, testing
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
  - Lab 8 (15 min) testing
  - Lab 9 (60 min) deployment

## Tasks for morning of 2019 Feb 25

- [x] set up linting
- [x] get rid of sliding-window cnn
- [x] get rid of non-ctc lstm
- [x] get to ~100% linted
- [ ] add training tests
- [ ] take some screenshots of looking at IAM dataset
- [ ] add metadata.toml and download data in a separate script, not from dataset python file directly
- [ ] add "subsample" mode to dataset
- [ ] add to lab 5: output sample predictions every epoch so that they can be reviewed in weights and biases
- [ ] go through the first 5 labs and make sure it all works
- [ ] use git-lfs for models
- [ ] sync with josh and give him latitude to make improvements, particularly in saving models
