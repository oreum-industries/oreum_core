# Oreum Core Tools `oreum_core`

[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-Apache2.0-blue.svg)](https://choosealicense.com/licenses/apache-2.0/)
[![GitHub Release](https://img.shields.io/github/v/release/oreum-industries/oreum_core?display_name=tag&sort=semver)](https://github.com/oreum-industries/oreum_core/releases)
[![PyPI](https://img.shields.io/pypi/v/oreum_core)](https://pypi.org/project/oreum_core)
[![CI](https://github.com/oreum-industries/oreum_core/workflows/ci/badge.svg)](https://github.com/oreum-industries/oreum_core/actions/workflows/ci.yml)
[![publish](https://github.com/oreum-industries/oreum_core/actions/workflows/publish.yml/badge.svg)](https://github.com/oreum-industries/oreum_core/actions/workflows/publish.yml)
[![code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![code style: interrogate](https://raw.githubusercontent.com/oreum-industries/oreum_core/master/assets/img/interrogate_badge.svg)](https://pypi.org/project/interrogate/)
[![code security: bandit](https://img.shields.io/badge/code%20security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)

---

## 1. Description and Scope

This is an ever-growing package of core tools for use on client projects by
Oreum Industries.

+ Provides an essential workflow for data curation, EDA, basic ML using the core
  scientific Python stack incl. `numpy`, `scipy`, `matplotlib`, `seaborn`,
  `pandas`, `scikit-learn`, `umap-learn`
+ Optionally provides an advanced Bayesian modeling workflow in R&D and
  Production using a leading probabilistic programming stack incl. `pymc`,
  `pytensor`, `arviz`
  (do `pip install oreum_core[pymc]`)
+ Optionally enables a generalist black-box ML workflow in R&D using a leading
  Gradient Boosted Trees stack incl. `catboost`, `xgboost`, `optuna`, `shap`
  (do `pip install oreum_core[tree]`)
+ Also includes several utilities for text cleaning, sql scripting, file handling


This package **is**:

+ A work in progress (v0.y.z) and liable to breaking changes and inconvenience
  to the user
+ Solely designed for ease of use and rapid development by employees of
  Oreum Industries, and selected clients with guidance

This package **is not**:

+ Intended for public usage and will not be supported for public usage
+ Intended for contributions by anyone not an employee of Oreum Industries,
  and unsolicited contributions will not be accepted.


### Notes

+ Project began on 2021-01-01
+ The `README.md` is MacOS and POSIX oriented
+ See `LICENCE.md` for licensing and copyright details
+ See `pyproject.toml` for various package details
+ This uses a logger named `'oreum_core'`, feel free to incorporate or ignore
  see `__init__.py` for details
+ Hosting:
  + Source code repo on [GitHub](https://github.com/oreum-industries/oreum_core)
  + Source code release on [GitHub](https://github.com/oreum-industries/oreum_core/releases)
  + Package release on [PyPi](https://pypi.org/project/oreum_core)
+ Implementation:
  + This project is enabled by a modern, open-source, advanced software stack
    for data curation, statistical analysis and predictive modelling
  + Specifically we use an open-source Python-based suite of software packages,
    the core of which is often known as the Scientific Python stack, supported
    by [NumFOCUS](https://numfocus.org)
  + Once installed (see section 2), see `LICENSES_THIRD_PARTY.md` for full
    details of all package licences
+ Environments: this project was originally developed on a Macbook Air M2
  (Apple Silicon ARM64) running MacOS 15.5 (Sequoia) using `osx-arm64` Accelerate

## 2. Instructions to Create Dev Environment

For local development on MacOS

### 2.0 Pre-requisite installs via `homebrew`

1. Install Homebrew, see instructions at [https://brew.sh](https://brew.sh)
2. Install `direnv`, `git`, `git-lfs`, `graphviz`, `tad`, `zsh`

```zsh
$> brew update && upgrade
$> brew install direnv git git-lfs graphviz tad zsh
```

### 2.1 Git clone the repo

Assumes `direnv`, `git`, `git-lfs` and `zsh` installed as above

```zsh
$> git clone https://github.com/oreum-industries/oreum_core
$> cd oreum_core
```
Then allow `direnv` on MacOS to autorun file `.envrc` upon directory open


### 2.2 Create virtual environment and install dev packages

Notes:

+ We use `conda` virtual envs controlled by `mamba` (quicker than `conda`)
+ We install packages using `miniforge` (sourced from the `conda-forge` repo)
  wherever possible and only use `pip` for packages that are handled better by
  `pip` and/or more up-to-date on [pypi](https://pypi.org)
+ Packages might not be the very latest because we want stability for `pymc`
  which is usually in a state of development flux
+ See [cheat sheet of conda commands](https://conda.io/docs/_downloads/conda-cheatsheet.pdf)
+ The `Makefile` creates a dev env and will also download and preinstall
  `miniforge` if not yet installed on your system


#### 2.2.1 Create the dev environment

From the dir above `oreum_core/` project dir:

```zsh
$> make -C oreum_core/ dev
```

This will also create some files to help confirm / diagnose successful installation:

+ `dev/install_log/blas_info.txt` for the `BLAS MKL` installation for `numpy`
+ `dev/install_log/pipdeptree[_rev].txt` lists installed package deps (and reversed)
+ `LICENSES_THIRD_PARTY.md` details the license for each package used


#### 2.2.2 (Optional best practice) Test successful installation of dev environment

From the dir above `oreum_core/` project dir:

```zsh
$> make -C oreum_core/ test-dev-env
```

This will also add files `dev/install_log/[numpy|scipy].txt` which detail
successful installation (or not) for `numpy`, `scipy`


#### 2.2.3 (Useful during env install experimentation): To remove the dev environment

From the dir above `oreum_core/` project dir:

```zsh
$> make -C oreum_core/ uninstall-env
```

### 2.3 Code Linting & Repo Control

#### 2.3.1 Pre-commit

We use [pre-commit](https://pre-commit.com) to run a suite of automated tests
for code linting & quality control and repo control prior to commit on local
development machines.

+ Precommit is already installed by the `make dev` command (which itself calls
`pip install -e .[dev]`)
+ The pre-commit script will then run on your system upon `git commit`
+ See this project's `.pre-commit-config.yaml` for details


#### 2.3.2 Github Actions

We use [Github Actions](https://docs.github.com/en/actions/using-workflows) aka
Github Workflows to run:

1. A suite of automated tests for commits received at the origin (i.e. GitHub)
2. Publishing to PyPi upon creating a GH Release

+ See `Makefile` for the CLI commands that are issued
+ See `.github/workflows/*` for workflow details


#### 2.3.3 Git LFS

We use [Git LFS](https://git-lfs.github.com) to store any large files alongside
the repo. This can be useful to replicate exact environments during development
and/or for automated tests

+ This requires a local machine install
  (see [Getting Started](https://git-lfs.github.com))
+ See `.gitattributes` for details


### 2.4 Configs for Local Development

Some notes to help configure local development environment

#### 2.4.1 Git config `~/.gitconfig`

```yaml
[user]
    name = <YOUR NAME>
    email = <YOUR EMAIL ADDRESS>
```


### 2.5 Install VSCode IDE

We strongly recommend using [VSCode](https://code.visualstudio.com) for all
development on local machines, and this is a hard pre-requisite to use
the `.devcontainer` environment (see section 3)

This repo includes relevant lightweight project control and config in:

```zsh
oreum_core.code-workspace
.vscode/extensions.json
.vscode/settings.json
```
---

## 3. Code Standards

Even when writing R&D code, we strive to meet and exceed (even define) best
practices for code quality, documentation and reproducibility for modern
data science projects.

### 3.1 Code Linting & Repo Control

We use a suite of automated tools to check and enforce code quality. We indicate
the relevant shields at the top of this README. See section 1.4 above for how
this is enforced at precommit on developer machines and upon PR at the origin as
part of our CI process, prior to master branch merge.

These include:

+ [`ruff`](https://docs.astral.sh/ruff/) - extremely fast standardised linting
  and formatting, which replaces `black`, `flake8`, `isort`
+ [`interrogate`](https://pypi.org/project/interrogate/) - ensure complete Python
  docstrings
+ [`bandit`](https://github.com/PyCQA/bandit) - test for common Python security
  issues

We also run a suite of general tests pre-packaged in
[`precommit`](https://pre-commit.com).



---

Copyright 2025 Oreum FZCO t/a Oreum Industries. All rights reserved.
Oreum FZCO, IFZA, Dubai Silicon Oasis, Dubai, UAE, reg. 25515
[oreum.io](https://oreum.io)

---
Oreum Industries &copy; 2025
