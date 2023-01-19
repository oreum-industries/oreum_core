# Makefile
SHELL := /bin/bash

linter_check: ## run code linters (checks only)
	pip install black flake8 interrogate isort
	black --check --diff oreum_core/
	isort --check-only oreum_core/
	flake8 oreum_core/
	interrogate oreum_core/

security_check: ## run basic code security check
	pip install bandit
	bandit --config pyproject.toml -r oreum_core/
