# Oreum Core Tools `oreum_core`

This is an ever-growing package of core tools for use on client projects by
Oreum Industries.

[![CI](https://github.com/oreum-industries/oreum_core/workflows/ci/badge.svg)](https://github.com/oreum-industries/oreum_core/actions/workflows/ci.yml)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![code style: flake8](https://img.shields.io/badge/code%20style-flake8-331188.svg)](https://flake8.pycqa.org/en/latest/)
[![code style: isort](https://img.shields.io/badge/code%20style-isort-%231674b1?style=flat)](https://pycqa.github.io/isort/)
[![code style: interrogate](https://raw.githubusercontent.com/oreum-industries/oreum_core/master/assets/img/interrogate_badge.svg)](https://pypi.org/project/interrogate/)
[![code security: bandit](https://img.shields.io/badge/code%20security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![GitHub Release](https://img.shields.io/github/v/release/oreum-industries/oreum_core?display_name=tag&sort=semver)](https://github.com/oreum-industries/oreum_core/releases)
[![publish](https://github.com/oreum-industries/oreum_core/actions/workflows/publish.yml/badge.svg)](https://github.com/oreum-industries/oreum_core/actions/workflows/publish.yml)
[![PyPI](https://img.shields.io/pypi/v/oreum_core)](https://pypi.org/project/oreum_core)
<!-- [![Conda Forge](https://img.shields.io/conda/vn/oreum-industries/oreum_core.svg)](https://anaconda.org/oreum-industries/oreum_core) -->



---

## 1. Description and Scope

This project uses a scientific Python stack, and enables & supports:

+ Exploratory data analysis via custom tabulations and plots using `seaborn`
+ Bayesian inferential modelling in R&D and Production via model helpers and custom distributions in `pymc` and `arviz`
+ Assorted data transformations, text cleaning, sql scripting and file handling


### Technical Overview

+ Project began on 2021-01-01
+ The `README.md` is MacOS and POSIX oriented
+ See `LICENCE.md` for licensing and copyright details
+ See `CONTRIBUTORS.md` for list of contributors
+ This uses a logger named `'oreum_core'`, feel free to incorporate or ignore
+ Hosting:
  + Source code repo on [GitHub](https://github.com/oreum-industries/oreum_core)
  + Source code release on [GitHub](https://github.com/oreum-industries/oreum_core/releases)
  + Package release on [PyPi](https://pypi.org/project/oreum_core)


### Scope

+ This package **is**:
  + A work in progress (v0.y.z) and liable to breaking changes and inconveniences
  to the user
  + Solely designed for ease of use and rapid development by employees of Oreum
  Industries, and selected clients with guidance

+ This package **is not**:
  + Intended for public usage and will not be supported for public usage
  + Intended for contributions by anyone not an employee of Oreum Industries, and unsolicitied contributions will not be accepted



## 2. Instructions to Create Dev Environment

For local development on MacOS

### 2.0 Pre-requisite installs via `homebrew`

1. Install Homebrew, see instuctions at [https://brew.sh](https://brew.sh)
2. Install `direnv`, `git`, `git-lfs`, `graphviz`

```zsh
$> brew update && upgrade
$> brew install direnv git git-lfs graphviz
```

### 2.1 Git clone the repo

Assumes `git`, `git-lfs` and `direnv` installed as above

```zsh
$> git clone https://github.com/oreum-industries/oreum_core
$> cd oreum_core
```
Then allow `direnv` on MacOS to automatically run file `.envrc` upon directory open


### 2.2 Create virtual environment and install dev packages

Notes:

+ We use `conda` virtual envs provided by `miniconda`
+ We install packages with compound method handled by `mamba` (quicker than `conda`)
for the main environment and packages, and `pip` for selected packages that are
handled better by pip and/or more up to date on pypi
+ Packages might not be the very latest because we want stability for `pymc`
which is usually in a state of development flux
+ See [cheat sheet of conda commands](https://conda.io/docs/_downloads/conda-cheatsheet.pdf)
+ The `Makefile` creates a dev env and will also download and preinstall `Miniconda`
if not yet installed on your system.

#### 2.2.1 Create the dev environment

```zsh
$> make dev
```
This will add a file `tests/results/blas_info.txt` which will detail
successful `BLAS MKL` installation (or not)


#### 2.2.2 (Optional best practice) Test successful installation of dev environment

```zsh
$> make test-dev-env
```

This will add files `tests/results/[numpy|scipy|pymc].txt` which will detail
successful installation (or not) for `numpy`, `scipy` ~~, and `pmyc`~~


#### 2.2.3 (Useful during env install experimentation): To remove the dev environment

Using the base environment, from a dir above the oreum_core project dir:

```zsh
$> make -C oreum_core uninstall
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


---
---

Copyright 2023 Oreum OÜ t/a Oreum Industries. All rights reserved.
See LICENSE.md.

Oreum OÜ t/a Oreum Industries, Sepapaja 6, Tallinn, 15551, Estonia,
reg.16122291, [oreum.io](https://oreum.io)

---
Oreum OÜ &copy; 2023
