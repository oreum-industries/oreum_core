# Makefile
# NOTE:
# + Intended for install on MacOS Apple Silicon arm64 using Accelerate
# + Uses local sh: optionally confirm shell create recipe w/ $(info $(SHELL))
# + Defaults to CI=0 (override for github actions in publish.yml)
.PHONY: brew build dev dev-test dev-uninstall help lint publish publish-test \
	report test test-pkg-dl
.SILENT: brew build dev dev-test dev-uninstall help lint publish publish-test \
	report test test-pkg-dl
PYTHON_NONVENV = $(or $(shell which python3), $(shell which python))
VERSION := $(shell echo $(VVERSION) | sed 's/v//')
CI?=0


brew:
	@echo "Install system-level packages for local dev on MacOS using brew..."
	brew update && brew upgrade && brew cleanup -s;
	brew install direnv gcc git graphviz libomp uv zsh;


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
	rm -f requirements.txt
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
	@echo "  report         show test and coverage reports as markdown"
	@echo "  test           run pytest suite"
	@echo "  test-pkg-dl    test dl & install from testpypi"


lint:
	@echo "Run lint & format and static checks..."
	if [ $(CI) -eq 1 ]; then \
		$(PYTHON_NONVENV) -m pip install uv; \
	fi;
	uv run --extra dev ruff check --config pyproject.toml --output-format=github;
	uv run --extra dev ruff format --config pyproject.toml --diff --no-cache;
	uv run --extra dev interrogate --config pyproject.toml oreum_core/;
	uv run --extra dev bandit --config pyproject.toml -r oreum_core/ -f json -o reports/bandit-report.json;


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


report:
	@echo "Show test and coverage reports as markdown (not committed)..."
	@echo "## Test Report"
	@uv run --extra dev python3 -c "\
import xml.etree.ElementTree as ET; \
r = ET.parse('reports/test-report.xml').getroot(); \
s = r if r.tag == 'testsuite' else r.find('testsuite'); \
print('| Tests | Failures | Errors | Skipped | Time (s) |'); \
print('|------:|---------:|-------:|--------:|---------:|'); \
print(f'| {s.get(\"tests\",0)} | {s.get(\"failures\",0)} | {s.get(\"errors\",0)} | {s.get(\"skipped\",0)} | {float(s.get(\"time\",0)):.2f} |')"
	@echo ""
	@echo "## Coverage Report"
	@uv run --extra dev coverage report --format=markdown --data-file=reports/.coverage


test:
	@echo "Run pytest suite..."
	if [ $(CI) -eq 1 ]; then \
		$(PYTHON_NONVENV) -m pip install uv; \
	fi;
	uv run --extra dev --extra pymc --extra tree pytest tests/ -v \
		--junit-xml=reports/test-report.xml \
		--cov=oreum_core \
		--cov-report=term-missing \
		--cov-report=xml:reports/coverage-report.xml;


test-pkg-dl:
	@echo "Test dl & install from testpypi using venv-temp. Pass VERSION=x.x.x"
	@uv venv .venv-temp; \
	trap "rm -rf .venv-temp" EXIT; \
	uv pip install --python .venv-temp pip; \
	. .venv-temp/bin/activate; \
		python -m pip index versions --pre -i https://test.pypi.org/simple/ oreum_core; \
		python -m pip install --pre -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple oreum_core==$(VERSION); \
		python -c "from importlib.metadata import version; assert version('oreum_core') == '$(VERSION)'";
