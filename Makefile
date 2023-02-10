# Makefile
# Assumes MacOS x64 (Intel) using Homebrew
.PHONY: build publish_to_testpypi conda dev lint security upgrade_pip
SHELL := /bin/bash
PYTHON_DEFAULT = $(or $(shell which python3), $(shell which python))
# PYTHON_ENV = $$HOME/opt/miniconda3/envs/oreum_core/bin/python
PYTHON_ENV = $(HOME)/opt/miniconda3/envs/oreum_core/bin/python

ifneq ("$(wildcard $(PYTHON_ENV))","")
    PYTHON = $(PYTHON_ENV)
else
    PYTHON = $(PYTHON_DEFAULT)
endif

# PYTHON_ENV = $(or $($$HOME/opt/miniconda3/envs/oreum_core/bin/python), $(PYTHON_DEFAULT))

yo:
	echo $(PYTHON)

build:  ## build package oreum_core
	make upgrade_pip
	$(PYTHON_DEFAULT) -m pip install flit
	export SOURCE_DATE_EPOCH=$(shell date +%s)
	$(PYTHON_DEFAULT) -m flit build


# publish_to_test:  ## build and publish to testpypi from local dev machine
# 	$(PYTHON) -m pip install flit keyring
# 	$(PYTHON) -m flit publish --repository testpypi

publish_to_testpypi:  ## build and publish to testpypi from local dev machine
	make upgrade_pip
	$(PYTHON_DEFAULT) -m pip install flit keyring
	export FLIT_INDEX_URL=https://test.pypi.org/legacy/; \
		export FLIT_USERNAME=__token__; \
		$(PYTHON_DEFAULT) -m flit publish


conda:  ## get miniconda for MacOS x64 (Intel)
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ~/miniconda.sh
	sleep 1
	bash ~/miniconda.sh -b -p $$HOME/miniconda
	sleep 1
	export PATH=$$HOME/miniconda/bin:$$PATH; \
	conda update --prefix $$HOME/miniconda --yes conda

dev:  # create local condaenv for dev
	export PATH=$$HOME/miniconda/bin:$$PATH; \
		if which conda; then echo conda ready; else make conda; fi
	conda update --yes --name base --channel defaults conda
	conda install --yes --name base --channel conda-forge mamba
	mamba env create --file condaenv_oreum_core.yml
	export PATH=$$HOME/opt/miniconda3/bin:$$PATH; \
		export PATH=$$HOME/opt/miniconda3/envs/oreum_core/bin:$$PATH; \
		export CONDA_ENV_PATH=$$HOME/opt/miniconda3/envs/oreum_core/bin; \
		export CONDA_DEFAULT_ENV=oreum_core; \
		source dev_env_install.sh

# pip install oreum_core[linter_check]
lint:  ## run code linters (checks only)
	$(PYTHON) -m pip install black flake8 interrogate isort
	black --check --diff --config pyproject.toml oreum_core/
	isort --check-only oreum_core/
	flake8 oreum_core/
	interrogate oreum_core/

# pip install oreum_core[security_check]
security:  ## run basic python code security check
	$(PYTHON) -m pip install bandit
	bandit --config pyproject.toml -r oreum_core/


upgrade_pip:
	$(PYTHON_DEFAULT) -m pip install --upgrade pip

# for ref
# $(eval PYTHON = /Users/jon/opt/miniconda3/envs/oreum_core/bin/python)
# $(PYTHON) -m pip install -e $(shell pwd)
