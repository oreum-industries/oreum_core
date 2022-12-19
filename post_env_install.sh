#!/bin/bash
# post_env_install.sh
set -euo pipefail  # strict mode
python -c "import numpy as np; np.__config__.show()" > blas_info.txt
pip-licenses -saud -f markdown --output-file LICENSES_THIRD_PARTY.md
pre-commit install
pre-commit autoupdate
