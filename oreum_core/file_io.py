# file_io.py
# copyright 2022 Oreum Industries
from pathlib import Path


class BaseFileIO:
    """Base handler to read/write files in a predictable way
    NOTE
    + This class is to be inherited e.g. `super().__init__(*args, **kwargs)`
    + Checks for existence of files for reading and dirs for writing
    + Allows a rootpath as we often use in R&D Notebooks
    """

    def __init__(self, rootdir: list = None):
        """Allow set a root path for convenience in Notebooks
        e.g. rootdir = DIR_MODELS_A = ['data', 'models', 'a']
        If used, then read/write will prepend this root to their input fqns
        """
        if rootdir is not None:
            self.rootdir = Path(*rootdir)
            if not self.rootdir.is_dir():
                raise FileNotFoundError(
                    f'Required dir does not exist {str(self.rootdir)}'
                )
        else:
            self.rootdir = None

    def get_path_read(self, fqn: str, use_rootdir: bool = True) -> Path:
        """Create and test fqn file existence for read"""
        path = Path(fqn)
        if (self.rootdir is not None) & use_rootdir:
            path = self.rootdir.joinpath(path)
        if not path.exists():
            raise FileNotFoundError(f'Required file does not exist {str(path)}')
        return path

    def get_path_write(self, fqn: str, use_rootdir: bool = True) -> Path:
        """Create and test dir existence for write"""
        path = Path(fqn)
        if (self.rootdir is not None) & use_rootdir:
            path = self.rootdir.joinpath(path)
        dr = Path(*path.parts[:-1])
        if not dr.is_dir():
            raise FileNotFoundError(f'Required dir does not exist {str(dr)}')
        return path
