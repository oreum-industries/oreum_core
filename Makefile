# Makefile
# Assume dev on MacOS x64 (Intel) using brew & miniconda, publish via GH Actions
.PHONY: build conda dev lint help pre-build pub pub-test test-dev-env test-dl-ins uninstall
SHELL := /bin/bash
PYTHON_DEFAULT = $(or $(shell which python3), $(shell which python))
PYTHON_ENV = $(HOME)/opt/miniconda3/envs/oreum_core/bin/python
ifneq ("$(wildcard $(PYTHON_ENV))","")
    PYTHON = $(PYTHON_ENV)
else
    PYTHON = $(PYTHON_DEFAULT)
endif
VERSION := $(shell echo $(VVERSION) | sed 's/v//')


build:  ## build package oreum_core (actually more of an "assemble" than build)
	make pre_build
	$(PYTHON) -m flit build


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
		$(PYTHON_ENV) -c "import numpy as np; np.__config__.show()" > tests/results/blas_info.txt; \
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

help:
	@echo "Use \`make <target>' where <target> is:"
	@echo "  build         build package oreum_core"
	@echo "  dev           create local dev env"
	@echo "  lint          run code lint & security checks"
	@echo "  pub           all-in-one build and publish to pypi"
	@echo "  pub-test      all-in-one build and publish to testpypi"
	@echo "  test-dev-env  optional test the local dev env numeric packages"
	@echo "  test-dl-ins   test dl & install from testpypi"
	@echo "  uninstall     uninstall local dev env (use from parent dir as `make -C oreum_core uninstall`)"


pre-build:  # setup env for flit build or flit publish
	rm -rf dist
	$(PYTHON) -m pip install flit keyring
	export SOURCE_DATE_EPOCH=$(shell date +%s)


pub:  ## all-in-one build and publish to pypi
	make pre-build
	export FLIT_INDEX_URL=https://upload.pypi.org/legacy/; \
		$(PYTHON) -m flit publish

pub-test:  ## all-in-one build and publish to testpypi
	make pre-build
	export FLIT_INDEX_URL=https://test.pypi.org/legacy/; \
		$(PYTHON) -m flit publish

test-dev-env: ## test the dev machine install of critial numeric packages
	export PATH=$$HOME/opt/miniconda3/bin:$$PATH; \
		export PATH=$$HOME/opt/miniconda3/envs/oreum_lab/bin:$$PATH; \
		export CONDA_ENV_PATH=$$HOME/opt/miniconda3/envs/oreum_lab/bin; \
		export CONDA_DEFAULT_ENV=oreum_lab; \
		$(PYTHON_ENV) -c "import numpy as np; np.test()" > tests/results/numpy.txt; \
		$(PYTHON_ENV) -c "import scipy as sp; sp.test()" > tests/results/scipy.txt

# $(PYTHON_ENV) -c "import pymc as pm; pm.test()" > tests/results/pymc.txt; \


test-dl-ins:  # test dl & install from testpypi, set env var or pass in VERSION
	$(PYTHON) -m pip uninstall -y oreum_core
	$(PYTHON) -m pip index versions --pre -i https://test.pypi.org/simple/ oreum_core
	$(PYTHON) -m pip install --pre -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple oreum_core==$(VERSION)
	$(PYTHON) -c "import oreum_core; assert oreum_core.__version__ == '$(VERSION)'"


uninstall:  # uninstall local condaenv for dev (run from base env)
	mamba env remove --name oreum_core
