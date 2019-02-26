#!/bin/bash
set -uo pipefail
set +e

FAILURE=false

echo "pipenv check"
pipenv check  # Not reporting failure here, because sometimes this fails due to API request limit

echo "pylint"
pipenv run pylint api text_recognizer training || FAILURE=true

echo "pycodestyle"
pipenv run pycodestyle --exclude=node_modules api text_recognizer training || FAILURE=true

echo "mypy"
pipenv run mypy api text_recognizer training || FAILURE=true

echo "bandit"
pipenv run bandit -ll -r {api,text_recognizer,training} -x node_modules || FAILURE=true

echo "shellcheck"
shellcheck ./**/*.sh || FAILURE=true

if [ "$FAILURE" = true ]; then
  exit 1
fi
exit 0
