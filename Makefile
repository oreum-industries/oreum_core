# Makefile
# Assume dev on MacOS x64 (Intel) using brew & miniconda, publish via GH Actions
.PHONY: build publish publish_to_testpypi conda dev lint
SHELL := /bin/bash
PYTHON_DEFAULT = $(or $(shell which python3), $(shell which python))
PYTHON_ENV = $(HOME)/opt/miniconda3/envs/oreum_core/bin/python
ifneq ("$(wildcard $(PYTHON_ENV))","")
    PYTHON = $(PYTHON_ENV)
else
    PYTHON = $(PYTHON_DEFAULT)
endif


build:  ## build package oreum_core
	$(PYTHON_DEFAULT) -m pip install --upgrade pip
	$(PYTHON_DEFAULT) -m pip install .[publish]
	export SOURCE_DATE_EPOCH=$(shell date +%s)
	$(PYTHON_DEFAULT) -m flit build


publish:  ## build and publish to pypi
	$(PYTHON_DEFAULT) -m pip install --upgrade pip
	$(PYTHON_DEFAULT) -m pip install .[publish]
	export FLIT_INDEX_URL=https://upload.pypi.org/legacy/; \
		export FLIT_USERNAME=__token__; \
		$(PYTHON_DEFAULT) -m flit publish


publish_to_testpypi:  ## build and publish to testpypi
	$(PYTHON_DEFAULT) -m pip install --upgrade pip
	$(PYTHON_DEFAULT) -m pip install .[publish]
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


lint:  ## run code lint & security checks
	$(PYTHON) -m pip install .[lint]
	black --check --diff --config pyproject.toml oreum_core/
	isort --check-only oreum_core/
	flake8 oreum_core/
	interrogate oreum_core/
	bandit --config pyproject.toml -r oreum_core/


# TODO install again post-publish
# https://flit.pypa.io/en/stable/cmdline.html#flit-install
