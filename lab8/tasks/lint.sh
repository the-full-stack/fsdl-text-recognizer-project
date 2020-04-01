#!/bin/bash
set -uo pipefail
set +e

FAILURE=false

echo "safety"
safety check -r requirements.txt -r requirements-dev.txt || FAILURE=true

echo "pylint"
pylint --ignore=.serverless api text_recognizer training || FAILURE=true

echo "pycodestyle"
pycodestyle --exclude=node_modules,.serverless,.ipynb_checkpoints api text_recognizer training || FAILURE=true

echo "pydocstyle"
pydocstyle api text_recognizer training || FAILURE=true

echo "mypy"
mypy api text_recognizer training || FAILURE=true

echo "bandit"
bandit -ll -r {api,text_recognizer,training} -x node_modules,.serverless || FAILURE=true

echo "shellcheck"
shellcheck tasks/*.sh || FAILURE=true

if [ "$FAILURE" = true ]; then
  echo "Linting failed"
  exit 1
fi
echo "Linting passed"
exit 0
