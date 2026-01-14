# Makefile
# NOTE:
# + Intended for install on MacOS Apple Silicon arm64 using Accelerate
# + Uses local sh: optionallay confirm shell create recipe w/ $(info $(SHELL))
# + Defaults to CI=0 (override for github actions in publish.yml)
.PHONY: brew build dev dev-test dev-uninstall help lint publish publish-test \
	test-pkg-dl
.SILENT: brew build dev dev-test dev-uninstall help lint publish publish-test \
	test-pkg-dl
PYTHON_NONVENV = $(or $(shell which python3), $(shell which python))
VERSION := $(shell echo $(VVERSION) | sed 's/v//')
CI?=0

brew:
	@echo "Install system-level packages for local dev on MacOS using brew..."
	brew update && brew upgrade && brew cleanup -s;
	brew install direnv gcc git graphviz uv zsh;

build:
	@echo "Build package oreum_core..."
	rm -rf dist
	if [ $(CI) -eq 1 ]; then \
		$(PYTHON_NONVENV) -m pip install uv; \
	fi;
	@uv venv .venv-temp; \
	trap "rm -rf .venv-temp" EXIT; \
	uv pip install --python .venv-temp flit keyring; \
	. .venv-temp/bin/activate; \
		export SOURCE_DATE_EPOCH=$(shell date +%s); \
		python -m flit build;

dev:
	@echo "Install project dev env on local machine using uv..."
	git init;
	uv sync --all-extras;
	uv export --no-hashes --format requirements-txt -o requirements.txt;
	. .venv/bin/activate; \
	 	python -c "import numpy as np; np.__config__.show()" > dev/install_log/blas_info.txt; \
		pip-licenses -saud -f markdown -i csv2md --output-file LICENSES_3P.md; \
		pre-commit install; \
		pre-commit autoupdate;

dev-test:
	@echo "Test dev machine installation of numpy and scipy..."
	. .venv/bin/activate; \
		python -c "import numpy as np; np.test()" > dev/install_log/tests_numpy.txt;
		python -c "import scipy as sp; sp.test()" > dev/install_log/tests_scipy.txt;

dev-uninstall:
	@echo "Uninstall project dev venv from local machine..."
	rm -rf .venv;
	rm -f uv.lock

help:
	@echo "Use \make <target> where <target> is:"
	@echo "  brew           install system-level packages for dev on MacOS"
	@echo "  build          build package oreum_core"
	@echo "  dev            create local dev env"
	@echo "  dev-test       test local dev env: numeric packages"
	@echo "  dev-uninstall  uninstall dev env"
	@echo "  lint           run lint & static checks"
	@echo "  publish        all-in-one build and publish to pypi"
	@echo "  publish-test   all-in-one build and publish to testpypi"
	@echo "  test-pkg-dl    test dl & install from testpypi"

lint:
	@echo "Run lint & format and static checks..."
	if [ $(CI) -eq 1 ]; then \
		$(PYTHON_NONVENV) -m pip install uv; \
	fi;
	@uv venv .venv-temp; \
	trap "rm -rf .venv-temp" EXIT; \
	uv pip install --python .venv-temp bandit interrogate ruff; \
	. .venv-temp/bin/activate; \
		ruff check --config pyproject.toml --output-format=github; \
		ruff format --config pyproject.toml --diff --no-cache; \
		interrogate --config pyproject.toml oreum_core/; \
		bandit --config pyproject.toml -r oreum_core/ -f json -o reports/bandit-report.json;

publish:
	@echo "All-in-one build and publish to pypi..."
	if [ $(CI) -eq 1 ]; then \
		$(PYTHON_NONVENV) -m pip install uv; \
	fi;
	@uv venv .venv-temp; \
	trap "rm -rf .venv-temp" EXIT; \
	uv pip install --python .venv-temp flit keyring; \
	. .venv-temp/bin/activate; \
		export SOURCE_DATE_EPOCH="$(shell date +%s)"; \
		export FLIT_INDEX_URL="https://upload.pypi.org/legacy/"; \
		if [ $(CI) -eq 1 ]; then \
			python -m flit publish; \
		else \
			set -a; \
			. .env; \
			set +a; \
			export FLIT_USERNAME="$$FLIT_USERNAME"; \
			export FLIT_PASSWORD="$$FLIT_PASSWORD_PYPI"; \
			python -m flit publish; \
		fi;


publish-test:
	@echo "All-in-one build and publish to testpypi..."
	if [ $(CI) -eq 1 ]; then \
		$(PYTHON_NONVENV) -m pip install uv; \
	fi;
	@uv venv .venv-temp; \
	trap "rm -rf .venv-temp" EXIT; \
	uv pip install --python .venv-temp flit keyring; \
	. .venv-temp/bin/activate; \
		export SOURCE_DATE_EPOCH="$(shell date +%s)"; \
		export FLIT_INDEX_URL="https://test.pypi.org/legacy/"; \
		if [ $(CI) -eq 1 ]; then \
			python -m flit publish; \
		else \
			set -a; \
			. .env; \
			set +a; \
			export FLIT_USERNAME="$$FLIT_USERNAME"; \
			export FLIT_PASSWORD="$$FLIT_PASSWORD_TESTPYPI"; \
			python -m flit publish; \
		fi;


test-pkg-dl:
	@echo "Test dl & install from testpypi using venv-temp. Pass VERSION=x.x.x"
	@uv venv .venv-temp; \
	trap "rm -rf .venv-temp" EXIT; \
	uv pip install --python .venv-temp pip; \
	. .venv-temp/bin/activate; \
		python -m pip index versions --pre -i https://test.pypi.org/simple/ oreum_core; \
		python -m pip install --pre -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple oreum_core==$(VERSION); \
		python -c "import oreum_core; assert oreum_core.__version__ == '$(VERSION)'";
