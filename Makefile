# Makefile
# NOTE:
# + Intended for install on MacOS Apple Silicon arm64 using Accelerate
# + Uses local zsh, server sh: confirm shell create recipe w/ $(info $(SHELL))
.PHONY: brew build cleanup dev dev-test dev-uninstall help lint lint-ci \
	publish publish-test test-pkg-dl
.SILENT: brew build cleanup dev dev-test dev-uninstall help lint lint-ci \
	publish publish-test test-pkg-dl
PYTHON_NONVENV = $(or $(shell which python3), $(shell which python))
VERSION := $(shell echo $(VVERSION) | sed 's/v//')

brew:
	@echo "Install system-level packages for local dev on MacOS using brew..."
	brew update && brew upgrade && brew cleanup -s;
	brew install direnv git uv zsh;

build:
	@echo "Build package oreum_core"
	rm -rf dist
	uv sync --extra pub;
	. .venv/bin/activate; \
		export SOURCE_DATE_EPOCH=$(shell date +%s); \
		python -m flit build;

dev:
	@echo "Install dev env on local machine using uv..."
	git init;
	uv sync --all-extras;
	uv export --no-hashes --format requirements-txt -o requirements.txt;
	source .venv/bin/activate; \
		pip-licenses -saud -f markdown -i csv2md --output-file LICENSES_3P.md; \
		pre-commit install; \
		pre-commit autoupdate;

dev-test:
	@echo "Test dev machine installation of numpy and scipy"
	. .venv/bin/activate; \
		python -c "import numpy as np; np.test()" > dev/install_log/tests_numpy.txt;

# 		python -c "import scipy as sp; sp.test()" > dev/install_log/tests_scipy.txt;

dev-uninstall:
	@echo "Remove / uninstall dev env from local machine..."
	rm -rf .venv;
	rm -f uv.lock

help:
	@echo "Use \make <target> where <target> is:"
	@echo "  brew           install system-level packages for dev on MacOS"
	@echo "  build          build package oreum_core"
	@echo "  dev            create local dev env"
	@echo "  dev-test       test local dev env: numeric packages"
	@echo "  dev-uninstall  uninstall dev env"
	@echo "  lint           run lint & static checks on local dev machine"
	@echo "  lint-ci        run lint & static checks on ci"
	@echo "  publish        all-in-one build and publish to pypi"
	@echo "  publish-test   all-in-one build and publish to testpypi"
	@echo "  test-pkg-dl    test dl & install from testpypi"

lint:
	@echo "Run lint / format and static checks..."
	. .venv/bin/activate; \
		ruff check --config pyproject.toml --output-format=github; \
		ruff format --config pyproject.toml --diff --no-cache; \
		interrogate --config pyproject.toml oreum_core/; \
		bandit --config pyproject.toml -r oreum_core/ -f json -o reports/bandit-report.json;

lint-ci:
	@echo "Run lint / format and static checks on CI/CD (installs venv)..."
	$(PYTHON_NONVENV) -m pip install uv;
	uv sync --extra dev;
	make lint;

publish:
	@echo "All-in-one build and publish to pypi"
	uv sync --extra pub;
	source .venv/bin/activate; \
		export SOURCE_DATE_EPOCH=$(shell date +%s); \
		export FLIT_INDEX_URL=https://upload.pypi.org/legacy/; \
		python -m flit publish

publish-test:
	@echo "All-in-one build and publish to testpypi"
	uv sync --extra pub;
	source .venv/bin/activate; \
		export SOURCE_DATE_EPOCH=$(shell date +%s); \
		export FLIT_INDEX_URL=https://test.pypi.org/legacy/; \
		python -m flit publish

test-pkg-dl:
	@echo "Test pkg dl&install from testpypi. Set $VERSION. Not using venv"
	uv install pip;
	$(PYTHON_NONVENV) -m pip uninstall -y oreum_core;
	$(PYTHON_NONVENV) -m pip index versions --pre -i https://test.pypi.org/simple/ oreum_core
	$(PYTHON_NONVENV) -m pip install --pre -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple oreum_core==$(VERSION)
	$(PYTHON_NONVENV) -c "import oreum_core; assert oreum_core.__version__ == '$(VERSION)'"
