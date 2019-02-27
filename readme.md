# Full Stack Deep Learning Labs

Welcome!

Project developed during lab sessions of the [Full Stack Deep Learning Bootcamp](https://fullstackdeeplearning.com).

- In this lab we will build a handwriting recognition system from scratch, and deploy it as a web service.
- We will use Keras with Tensorflow backend as the underlying framework.
- The framework will not lock us in to many design choices, and can easily be replaced with, for example, PyTorch.
- We will structure the project in a way that will scale with future development and allow us to run experiments.
- We will evaluate both convolutional and sequence methods for the task, and will see an example of how to compute loss in a more advanced way.
- We will sync our experimental runs to Weights & Biases, and use it as a leaderboard.
- We will run experiments on multiple GPUs, and store results to an online experiment management platform.
- We will set up continuous integration system for our codebase, which will check functionality of code and evaluate the model about to be deployed.
- We will package up the prediction system as a REST API, deployable as a Docker container.
- We will deploy the prediction system as a serverless function to Amazon Lambda.
- Lastly, we will set up monitoring that alerts us when the incoming data distribution changes.

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
  - Lab 8 (75 min) testing & deployment
