#!/bin/sh
pipenv lock --requirements --keep-outdated > api/requirements.txt
cd api
npm install
pipenv run sls deploy -v
