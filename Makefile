# Arcane incantation to print all the other targets, from https://stackoverflow.com/a/26339924
help:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'

# Install exact Python and CUDA versions
conda-update:
	conda env update --prune -f environment.yml
	echo "!!!RUN THE conda activate COMMAND ABOVE RIGHT NOW!!!"

# Compile and install exact pip packages
pip-tools:
	pip install pip-tools==6.3.1 setuptools==59.5.0
	pip-compile requirements/prod.in && pip-compile requirements/dev.in
	pip-sync requirements/prod.txt requirements/dev.txt

# Compile and install the requirements for local linting (optional)
pip-tools-lint:
	pip install pip-tools==6.3.1 setuptools==59.5.0
	pip-compile requirements/prod.in && pip-compile requirements/dev.in && pip-compile requirements/dev-lint.in
	pip-sync requirements/prod.txt requirements/dev.txt requirements/dev-lint.txt

# bump versions of transitive dependencies
pip-tools-upgrade:
	pip install pip-tools==6.3.1 setuptools==59.5.0
	pip-compile --upgrade requirements/prod.in && pip-compile --upgrade requirements/dev.in && pip-compile --upgrade requirements/dev-lint.in

# Example training command
train-mnist-cnn-ddp:
	python training/run_experiment.py --max_epochs=10 --gpus=-1 --accelerator=ddp --num_workers=20 --data_class=MNIST --model_class=CNN

# Lint
lint:
	tasks/lint.sh
