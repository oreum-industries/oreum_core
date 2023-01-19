# Makefile
SHELL := /bin/bash

linter_check: ## run code linters (checks only)
	pip install black flake8 interrogate isort sqlfluff
	black --check --diff src/
	isort --check-only src/
	flake8 src/
	interrogate src/

security_check: ## run basic code security check
	pip install bandit
	bandit --config pyproject.toml -r src/
