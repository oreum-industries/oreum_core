# Makefile
# NOTE:
# + Intended for dev install on MacOS x64 (Intel via Rosetta 2)
# + On MacOS, the miniforge.sh installer is too clever and complains
#   that "$(uname -m)" != "x86_64", but we ignore that: Rosetta2 will run x86_64
.PHONY: build create-env dev lint help mamba pre-build pub test-pub test-dev-env test-dl-ins uninstall uninstall-mamba
.SILENT: build create-env dev lint help mamba pre-build pub test-pub test-dev-env test-dl-ins uninstall uninstall-mamba
MAMBADL := https://github.com/conda-forge/miniforge/releases/download/23.3.1-1
MAMBAV := Miniforge3-MacOSX-x86_64.sh
MAMBARCMSG := Please create file $(MAMBARC), particularly to set `platform: osx-64`
MAMBARC := $(HOME)/.mambarc
MAMBADIR := $(HOME)/miniforge
PYTHON_DEFAULT = $(or $(shell which python3), $(shell which python))
PYTHON_ENV = $(MAMBADIR)/envs/oreum_core/bin/python
ifneq ("$(wildcard $(PYTHON_ENV))","")
    PYTHON = $(PYTHON_ENV)
else
    PYTHON = $(PYTHON_DEFAULT)
endif
VERSION := $(shell echo $(VVERSION) | sed 's/v//')

build:  ## build package oreum_core (actually more of an "assemble" than a compile)
	make pre-build
	$(PYTHON) -m flit build

create-env:  ## create mamba (conda) environment
	export PATH=$(MAMBADIR)/bin:$$PATH; \
		if which mamba; then echo "mamba ready"; else make mamba; fi
	export PATH=$(MAMBADIR)/bin:$$PATH; \
		export CONDA_SUBDIR=osx-64; \
		mamba update -n base mamba; \
		mamba env create --file condaenv_oreum_core.yml;

dev:  ## create env for local dev on any machine MacOS x64 (Intel)
	make create-env
	export PATH=$(MAMBADIR)/envs/oreum_core/bin:$$PATH; \
		export CONDA_ENV_PATH=$(MAMBADIR)/envs/oreum_core/bin; \
		export CONDA_DEFAULT_ENV=oreum_core; \
		export CONDA_SUBDIR=osx-64; \
		$(PYTHON_ENV) -m pip index versions oreum_core; \
		$(PYTHON_ENV) -m pip install -e ".[dev]"; \
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
	@echo "  test-pub      all-in-one build and publish to testpypi"
	@echo "  test-dev-env  optional test the local dev env numeric packages"
	@echo "  test-dl-ins   test dl & install from testpypi"
	@echo "  uninstall     remove local dev env (use from parent dir as `make -C oreum_core uninstall`)"

mamba:  ## get mamba via miniforge for MacOS x86_64 (Intel via Rosetta2) use zsh
	test -f $(MAMBARC) || { echo $(MAMBARCMSG); exit 1; }
	wget $(MAMBADL)/$(MAMBAV) -O $(HOME)/miniforge.sh
	chmod 755 $(HOME)/miniforge.sh
	zsh $(HOME)/miniforge.sh -b -p $(MAMBADIR)
	export PATH=$(MAMBADIR)/bin:$$PATH; \
		conda init zsh;
	rm $(HOME)/miniforge.sh

# NOTE as-at 2023-09-27 this craps out with
#  /Users/jon/miniforge.sh:342: no matches found: /Users/jon/miniforge/pkgs/envs/*/
# make: *** [mamba] Error 1
# but does actually do what it should

pre-build:  ## setup env for flit build or flit publish
	rm -rf dist
	$(PYTHON) -m pip install flit keyring
	export SOURCE_DATE_EPOCH=$(shell date +%s)

pub:  ## all-in-one build and publish to pypi
	make pre-build
	export FLIT_INDEX_URL=https://upload.pypi.org/legacy/; \
		$(PYTHON) -m flit publish

test-pub:  ## all-in-one build and publish to testpypi
	make pre-build
	export FLIT_INDEX_URL=https://test.pypi.org/legacy/; \
		$(PYTHON) -m flit publish

test-dev-env:  ## test the dev machine install of critial numeric packages
	export PATH=$(MAMBADIR)/bin:$$PATH; \
		export PATH=$(MAMBADIR)/envs/oreum_core/bin:$$PATH; \
		export CONDA_ENV_PATH=$(MAMBADIR)/envs/oreum_core/bin; \
		export CONDA_DEFAULT_ENV=oreum_core; \
		$(PYTHON_ENV) -c "import numpy as np; np.test()" > dev/install_log/numpy.txt; \
		$(PYTHON_ENV) -c "import scipy as sp; sp.test()" > dev/install_log/scipy.txt;

# $(PYTHON_ENV) -c "import pymc as pm; pm.test()" > dev/install_log/pymc.txt; \

test-dl-ins:  ## test dl & install from testpypi, set env var or pass in VERSION
	$(PYTHON) -m pip uninstall -y oreum_core
	$(PYTHON) -m pip index versions --pre -i https://test.pypi.org/simple/ oreum_core
	$(PYTHON) -m pip install --pre -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple oreum_core==$(VERSION)
	$(PYTHON) -c "import oreum_core; assert oreum_core.__version__ == '$(VERSION)'"

uninstall: ## remove mamba env (use from parent dir as `make -C oreum_core/ uninstall`)
	mamba env remove --name oreum_core -y
	mamba clean -afy

uninstall-mamba:  ## last ditch per https://github.com/conda-forge/miniforge#uninstallation
	conda init zsh --reverse
	rm -rf $(MAMBADIR)
	rm -rf $(HOME)/.conda
	rm -f $(HOME)/.condarc
	source $(HOME)/.zshrc
