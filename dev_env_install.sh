#!/bin/bash
# dev_env_install.sh
pip install -e .
# pip install oreum_core[dev]  # note this can only work after the first publish to pypi!
pip install pip-licenses pre-commit black flake8 interrogate isort
python -c "import numpy as np; np.__config__.show()" > blas_info.txt
pip-licenses -saud -f markdown --output-file LICENSES_THIRD_PARTY.md
pre-commit install
pre-commit autoupdate
