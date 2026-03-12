# Copyright 2026 Oreum Industries
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# utils.file_io.py
"""Common File IO utils"""

import logging
import subprocess
from pathlib import Path

from .snakey_lowercaser import SnakeyLowercaser

__all__ = ["BaseFileIO", "check_fqns_exist", "copy_csv2md"]

_log = logging.getLogger(__name__)


class BaseFileIO:
    """Base handler to read/write files in a predictable way
    NOTE
    + This class is to be inherited e.g. `super().__init__(*args, **kwargs)`
    + Checks for existence of files for reading and dirs for writing
    + Allows a rootdir / rootpath as we often use in R&D Notebooks
    """

    def __init__(self, rootdir: Path | None = None, **kwargs):
        """Allow set a root path for convenience in Notebooks
        e.g. rootdir = DIR_MODELS_A = ['data', 'models', 'a']
        If used, then read/write will prepend this root to their input fqns
        """
        if rootdir is None:
            self.rootdir = Path().cwd()
        elif not rootdir.is_dir():
            raise FileNotFoundError(
                f"Required dir does not exist {str(rootdir.resolve())}"
            )
        else:
            self.rootdir = rootdir
        self.snl = SnakeyLowercaser(allowed_punct="-")

    def get_path_read(self, fn: str | Path) -> Path:
        """Create and test fqn file existence for read"""
        fqn = self.rootdir.joinpath(fn)
        if not fqn.exists():
            raise FileNotFoundError(
                f"Required file does not exist {str(fqn.resolve())}"
            )
        return fqn

    def get_path_write(self, fn: str | Path) -> Path:
        """Create dir if needed and return fqn for write
        Ensure the passed fn is snl.cleaned"""
        fqn = self.rootdir.joinpath(fn)
        fqn.parent.mkdir(parents=True, exist_ok=True)
        return fqn


def copy_csv2md(fn: str) -> Path:
    """Convenience to copy csv 'path/x.csv' to markdown 'path/x.md'
    Requires optional dependency: pip install csv2md
    """
    try:
        import csv2md  # noqa: F401
    except ImportError as e:
        raise ImportError("copy_csv2md requires csv2md: pip install csv2md") from e
    fileio = BaseFileIO()
    fqn = fileio.get_path_read(fn)
    r = subprocess.run(["csv2md", f"{fqn}"], capture_output=True, check=True)
    fn_out = Path(fn).with_suffix(".md")
    fqn_out = fileio.get_path_write(fn_out)
    with open(fqn_out, "wb") as f:
        f.write(r.stdout)
    _log.info(f"Written to {str(fqn_out.resolve())}")
    return fqn_out


def check_fqns_exist(fqns: dict[str, Path]) -> bool:
    """Basic checks files required are present"""
    for path in fqns.values():
        if not path.resolve().exists():
            raise FileNotFoundError(
                f"Required file does not exist {str(path.resolve())}"
            )
    return True
