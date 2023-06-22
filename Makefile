# Makefile
# Assume dev on MacOS x64 (Intel) using brew & miniconda, publish via GH Actions
.PHONY: build create-env dev lint help mamba pre-build pub pub-test test-dev-env test-dl-ins uninstall
.SILENT: build create-env dev lint help mamba pre-build pub pub-test test-dev-env test-dl-ins uninstall
SHELL := /bin/bash
MAMBADL = https://github.com/conda-forge/miniforge/releases/latest/download/
MAMBAV = Mambaforge-MacOSX-x86_64.sh
MAMBARC = $$HOME/.mambarc
MAMBARCMSG = Please create file $(MAMBARC), particularly to set `platform: osx-64`
MAMBADIR = $$HOME/.mamba
PYTHON_DEFAULT = $(or $(shell which python3), $(shell which python))
PYTHON_ENV = $(MAMBADIR)/envs/oreum_core/bin/python
ifneq ("$(wildcard $(PYTHON_ENV))","")
    PYTHON = $(PYTHON_ENV)
else
    PYTHON = $(PYTHON_DEFAULT)
endif
VERSION := $(shell echo $(VVERSION) | sed 's/v//')

build:  ## build package oreum_core (actually more of an "assemble" than a compile)
	make pre_build
	$(PYTHON) -m flit build

create-env: ## create mamba (conda) environment  CONDA_SUBDIR=osx-64
	export PATH=$(MAMBADIR)/bin:$$PATH; \
		if which mamba; then echo "mamba ready"; else make mamba; fi
	mamba env create --file condaenv_oreum_core.yml;

dev:  # create env for local dev on any machine MacOS x64 (Intel)
	make create-env
	export PATH=$(MAMBADIR)/envs/oreum_core/bin:$$PATH; \
		export CONDA_ENV_PATH=$(MAMBADIR)/envs/oreum_core/bin; \
		export CONDA_DEFAULT_ENV=oreum_core; \
		export CONDA_SUBDIR=osx-64; \
		$(PYTHON_ENV) -m pip index versions oreum_core; \
		$(PYTHON_ENV) -m pip install -e .[dev]; \
		$(PYTHON_ENV) -c "import numpy as np; np.__config__.show()" > dev/install_log/blas_info.txt; \
		pipdeptree -a > dev/install_log/pipdeptree.txt; \
		pipdeptree -a -r > dev/install_log/pipdeptree_rev.txt; \
		pip-licenses -saud -f markdown --output-file LICENSES_THIRD_PARTY.md; \
		pre-commit install; \
		pre-commit autoupdate;

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

mamba:  ## get mamba via mambaforge for MacOS x86_64 (Intel via Rosetta2)
	test -f $(MAMBARC) || { echo $(MAMBARCMSG); exit 1; }
	wget $(MAMBADL)$(MAMBAV) -O $$HOME/mambaforge.sh
	bash $$HOME/mambaforge.sh -b -p $(MAMBADIR)
	export PATH=$(MAMBADIR)/bin:$$PATH; \
		conda init zsh;
	rm $$HOME/mambaforge.sh

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
	export PATH=$(MAMBADIR)/bin:$$PATH; \
		export PATH=$(MAMBADIR)/envs/oreum_core/bin:$$PATH; \
		export CONDA_ENV_PATH=$(MAMBADIR)/envs/oreum_core/bin; \
		export CONDA_DEFAULT_ENV=oreum_core; \
		$(PYTHON_ENV) -c "import numpy as np; np.test()" > dev/install_log/numpy.txt; \
		$(PYTHON_ENV) -c "import scipy as sp; sp.test()" > dev/install_log/scipy.txt;

# $(PYTHON_ENV) -c "import pymc as pm; pm.test()" > dev/install_log/pymc.txt; \

test-dl-ins:  # test dl & install from testpypi, set env var or pass in VERSION
	$(PYTHON) -m pip uninstall -y oreum_core
	$(PYTHON) -m pip index versions --pre -i https://test.pypi.org/simple/ oreum_core
	$(PYTHON) -m pip install --pre -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple oreum_core==$(VERSION)
	$(PYTHON) -c "import oreum_core; assert oreum_core.__version__ == '$(VERSION)'"


uninstall:  # uninstall local mamba env (run from base env)
	mamba env remove --name oreum_core -y
	mamba clean -ay
