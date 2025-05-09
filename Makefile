# Makefile
# NOTE:
# + Intended for install on MacOS Apple Silicon arm64 using Accelerate
#   (NOT Intel x86 using MKL via Rosetta 2)
# + Uses sh by default: to confirm shell create a recipe with $(info $(SHELL))
.PHONY: build dev help install-env install-mamba lint pre-build pub test-pub\
	    test-dev-env test-dl-ins uninstall-env uninstall-mamba
.SILENT: build dev help install-env install-mamba lint pre-build pub test-pub\
	    test-dev-env test-dl-ins uninstall-env uninstall-mamba
MAMBADL := https://github.com/conda-forge/miniforge/releases/download/24.11.3-0
MAMBAV := Miniforge3-MacOSX-arm64.sh
MAMBARC := $(HOME)/.mambarc
MAMBARCMSG := Please create file $(MAMBARC), importantly set \`platform: osx-arm64\`
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

dev:  ## create env for local dev
	make install-env
	export PATH=$(MAMBADIR)/envs/oreum_core/bin:$$PATH; \
		export MAMBA_EXE='$(MAMBADIR)/bin/mamba'; \
		export MAMBA_ROOT_PREFIX='$(MAMBADIR)'; \
		$(PYTHON_ENV) -m pip install -e ".[all]"; \
		$(PYTHON_ENV) -c "import numpy as np; np.__config__.show()" > dev/install_log/blas_info.txt; \
		pipdeptree -a > dev/install_log/pipdeptree.txt; \
		pipdeptree -a -r > dev/install_log/pipdeptree_rev.txt; \
		pip-licenses -saud -f markdown -i csv2md --output-file LICENSES_THIRD_PARTY.md; \
		pre-commit install; \
		pre-commit autoupdate;

install-env:  ## create mamba (conda) environment
	export PATH=$(MAMBADIR)/bin:$$PATH; \
		if which mamba; then echo "mamba ready"; else make install-mamba; fi
	export PATH=$(MAMBADIR)/bin:$$PATH; \
		export MAMBA_EXE='$(MAMBADIR)/bin/mamba'; \
		export MAMBA_ROOT_PREFIX='$(MAMBADIR)'; \
		mamba update -n base mamba; \
		mamba env create --file condaenv_oreum_core.yml -y;

install-mamba:  ## get mamba via miniforge, explicitly use bash
	test -f $(MAMBARC) || { echo $(MAMBARCMSG); exit 1; }
	wget $(MAMBADL)/$(MAMBAV) -O $(HOME)/miniforge.sh
	chmod 755 $(HOME)/miniforge.sh
	bash $(HOME)/miniforge.sh -b -p $(MAMBADIR)
	export PATH=$(MAMBADIR)/bin:$$PATH; \
		mamba shell init;
	rm $(HOME)/miniforge.sh

lint: ## run code linters and static security (checks only)
	$(PYTHON) -m pip install bandit interrogate ruff
	ruff check --diff
	ruff format --no-cache --diff
	interrogate --config pyproject.toml oreum_core/
	bandit --config pyproject.toml -r oreum_core/

help:
	@echo "Use \make <target> where <target> is:"
	@echo "  build         build package oreum_core"
	@echo "  dev           create local dev env"
	@echo "  lint          run code lint & security checks"
	@echo "  pub           all-in-one build and publish to pypi"
	@echo "  test-pub      all-in-one build and publish to testpypi"
	@echo "  test-dev-env  optional test local dev env numeric packages, v.slow"
	@echo "  test-dl-ins   test dl & install from testpypi"
	@echo "  uninstall-env remove env (use from parent dir \make -C oreum_core ...)"

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

test-dev-env:  ## test the dev machine install of critical numeric packages
	export PATH=$(MAMBADIR)/bin:$$PATH; \
		export MAMBA_EXE='$(MAMBADIR)/bin/mamba'; \
		export MAMBA_ROOT_PREFIX='$(MAMBADIR)'; \
		$(PYTHON_ENV) -c "import numpy as np; np.test()" > dev/install_log/tests_numpy.txt; \
		$(PYTHON_ENV) -c "import scipy as sp; sp.test()" > dev/install_log/tests_scipy.txt;

test-dl-ins:  ## test dl & install from testpypi, set env var or pass in VERSION
	$(PYTHON) -m pip uninstall -y oreum_core
	$(PYTHON) -m pip index versions --pre -i https://test.pypi.org/simple/ oreum_core
	$(PYTHON) -m pip install --pre -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple oreum_core==$(VERSION)
	$(PYTHON) -c "import oreum_core; assert oreum_core.__version__ == '$(VERSION)'"

uninstall-env: ## remove mamba env
	export PATH=$(MAMBADIR)/bin:$$PATH; \
		export MAMBA_ROOT_PREFIX='$(MAMBADIR)'; \
		mamba env remove --name oreum_core -y; \
		mamba clean -ay

uninstall-mamba:  ## nuclear option https://github.com/conda-forge/miniforge?tab=readme-ov-file#uninstall
	mamba shell deinit
	rm -rf $(MAMBADIR)
	rm -rf $(HOME)/.mamba
	rm -f $(HOME)/.mambarc
	rm -f $(HOME)/.mambarc_x86
