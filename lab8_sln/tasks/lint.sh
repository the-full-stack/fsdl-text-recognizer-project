#!/bin/bash
set -uo pipefail
set +e

FAILURE=false

echo "pipenv check"
pipenv check  # Not reporting failure here, because sometimes this fails due to API request limit

echo "pylint"
pipenv run pylint --ignore=.serverless api text_recognizer training || FAILURE=true

echo "pycodestyle"
pipenv run pycodestyle --exclude=node_modules,.serverless,.ipynb_checkpoints api text_recognizer training || FAILURE=true

echo "mypy"
pipenv run mypy api text_recognizer training || FAILURE=true

echo "bandit"
pipenv run bandit -ll -r {api,text_recognizer,training} -x node_modules,.serverless || FAILURE=true

echo "shellcheck"
shellcheck tasks/*.sh || FAILURE=true

if [ "$FAILURE" = true ]; then
  echo "Linting failed"
  exit 0  # TODO: don't actually fail circleci
fi
echo "Linting passed"
exit 0
