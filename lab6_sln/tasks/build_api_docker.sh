pipenv lock --requirements --keep-outdated > api/requirements.txt
sed -i 's/-gpu//g' api/requirements.txt
docker build -t text_recognizer_api -f api/Dockerfile .
