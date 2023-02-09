# Makefile
# Assumes MacOS x64 (Intel) using Homebrew
SHELL := /bin/bash
.PHONY: build conda dev env-remove linter security
PYTHON = $(or $($$HOME/opt/miniconda3/envs/oreum_core/bin/python), $(shell which python))

build:  ## build package oreum_core
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install build
	$(PYTHON) -m build

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

linter:  ## run code linters (checks only)
	pip install oreum_core[linter_check]
	black -SC --check --diff --config pyproject.toml oreum_core/
	isort --check-only oreum_core/
	flake8 oreum_core/
	interrogate oreum_core/

security:  ## run basic python code security check
	pip install oreum_core[security_check]
	bandit --config pyproject.toml -r oreum_core/

# for ref
# $(eval PYTHON = /Users/jon/opt/miniconda3/envs/oreum_core/bin/python)
# $(PYTHON) -m pip install -e $(shell pwd)
