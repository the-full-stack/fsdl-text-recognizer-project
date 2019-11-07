# Lab 7: Testing and Continuous Integration

## Goal of the lab

- Add evaluation tests
- Add linting to our codebase
- Set up continuous integration via CircleCI, and see our commits pass/fail

## Follow along

```
git pull
cd lab7/
```

## Linting script

Running `tasks/lint.sh` fully lints our codebase with a few different checkers:

- `pipenv check` scans our Python package dependency graph for known security vulnerabilities
- `pylint` does static analysis of Python files and reports both style and bug problems
- `pycodestyle` checks for simple code style guideline violations (somewhat overlapping with `pylint`)
- `mypy` performs static type checking of Python files
- `bandit` performs static analysis to find common security vulnerabilities in Python code
- `shellcheck` finds bugs and potential bugs in shell scrips

A note: in writing Bash scripts, I often refer to [this excellent guide](http://redsymbol.net/articles/unofficial-bash-strict-mode/).

Note that the linters are configured using `.pylintrc` and `setup.cfg` files, as well as flags specified in `lint.sh`.

Getting linting right will pay off in no time, and is a must for any multi-developer codebase.

## Setting up CircleCI

The relevant new files for setting up continuous integration are

- `evaluation/evaluate_character_predictor.py`
- `evaluation/evaluate_line_predictor.py`
- `tasks/test_validation.sh`

There is one additional file that is outside of the lab7 directory (in the top-level directory): `.circleci/config.yml`

Let's set up CircleCI first and then look at the new evaluation files.

Go to https://circleci.com and log in with your Github account.
Click on Add Project. Select your fork of the `fsdl-text-recognizer-project` repo.
It will ask you to place the `config.yml` file in the repo.
Good news -- it's already there, so you can just hit the "Start building" button.

While CircleCI starts the build, let's look at the `config.yml` file.

Let's also check out the new validation test files: they simply evaluate the trained predictors on respective test sets, and make sure they are above threshold accuracy.

Now that CircleCI is done building, let's push a commit so that we can see it build again, and check out the nice green chechmark in our commit history (https://github.com/sergeyktest/fsdl-text-recognizer-project/commits/master)
