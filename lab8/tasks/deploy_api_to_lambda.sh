#!/bin/bash

pipenv lock --requirements --keep-outdated > api/requirements.txt
sed -i 's/tensorflow-gpu/tensorflow/' api/requirements.txt
cd api || exit 1
npm install
export PATH="$PWD/node_modules/serverless/bin:$PATH"
serverless deploy -v
