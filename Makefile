# Makefile
# Assume dev on MacOS x64 (Intel) using brew & miniconda, publish via GH Actions
.PHONY: conda dev lint pre_build build publish test_publish test_install
SHELL := /bin/bash
PYTHON_DEFAULT = $(or $(shell which python3), $(shell which python))
PYTHON_ENV = $(HOME)/opt/miniconda3/envs/oreum_core/bin/python
ifneq ("$(wildcard $(PYTHON_ENV))","")
    PYTHON = $(PYTHON_ENV)
else
    PYTHON = $(PYTHON_DEFAULT)
endif


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
		$(PYTHON_ENV) -m pip install -e .[dev]; \
		$(PYTHON_ENV) -c "import numpy as np; np.__config__.show()" > blas_info.txt; \
		pip-licenses -saud -f markdown --output-file LICENSES_THIRD_PARTY.md; \
		pre-commit install; \
		pre-commit autoupdate


lint:  ## run code lint & security checks
	$(PYTHON) -m pip install black flake8 interrogate isort bandit
	black --check --diff --config pyproject.toml oreum_core/
	isort --check-only oreum_core/
	flake8 oreum_core/
	interrogate oreum_core/
	bandit --config pyproject.toml -r oreum_core/


pre_build:  # setup env for flit build or flit publish
	rm -rf dist
	$(PYTHON) -m pip install flit keyring
	export SOURCE_DATE_EPOCH=$(shell date +%s)


build:  ## build package oreum_core (actually more of an "assemble" than build)
	make pre_build
	$(PYTHON) -m flit build


publish:  ## all-in-one build and publish to pypi
	make pre_build
	export FLIT_INDEX_URL=https://upload.pypi.org/legacy/; \
		$(PYTHON) -m flit publish


test_publish:  ## all-in-one build and publish to testpypi
	make pre_build
	export FLIT_INDEX_URL=https://test.pypi.org/legacy/; \
		$(PYTHON) -m flit publish


test_install:  # test dl & install from testpypi, set env var or pass in VERSION
	$(PYTHON) -m pip uninstall -y oreum_core
	$(PYTHON) -m pip install -i https://test.pypi.org/simple/ oreum_core==$(VERSION)
	$(PYTHON) -c "import oreum_core; assert oreum_core.__version__ == '$(VERSION)'"
