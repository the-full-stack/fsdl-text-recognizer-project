#!/bin/bash

pipenv lock --requirements --keep-outdated > api/requirements.txt
cd api || exit 1
npm install
pipenv run sls deploy -v
