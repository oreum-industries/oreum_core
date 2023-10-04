# Copyright 2023 Oreum Industries
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

# file_io.py
"""Common File IO"""
from pathlib import Path

__all__ = ['BaseFileIO']


class BaseFileIO:
    """Base handler to read/write files in a predictable way
    NOTE
    + This class is to be inherited e.g. `super().__init__(*args, **kwargs)`
    + Checks for existence of files for reading and dirs for writing
    + Allows a rootpath as we often use in R&D Notebooks
    """

    def __init__(self, rootdir: list[str] = None):
        """Allow set a root path for convenience in Notebooks
        e.g. rootdir = DIR_MODELS_A = ['data', 'models', 'a']
        If used, then read/write will prepend this root to their input fqns
        """
        if rootdir is not None:
            self.rootdir = Path(*rootdir)
            if not self.rootdir.is_dir():
                raise FileNotFoundError(
                    f'Required dir does not exist {str(self.rootdir.resolve())}'
                )
        else:
            self.rootdir = None

    def get_path_read(self, fqn: str, use_rootdir: bool = True) -> Path:
        """Create and test fqn file existence for read"""
        path = Path(fqn)
        if (self.rootdir is not None) & use_rootdir:
            path = self.rootdir.joinpath(path)
        if not path.exists():
            raise FileNotFoundError(
                f'Required file does not exist {str(path.resolve())}'
            )
        return path

    def get_path_write(self, fqn: str, use_rootdir: bool = True) -> Path:
        """Create and test dir existence for write"""
        path = Path(fqn)
        if (self.rootdir is not None) & use_rootdir:
            path = self.rootdir.joinpath(path)
        dr = Path(*path.parts[:-1])
        if not dr.is_dir():
            raise FileNotFoundError(f'Required dir does not exist {str(dr.resolve())}')
        return path
