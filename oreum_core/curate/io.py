# curate.io.py
# copyright 2022 Oreum Industries
import json
import os
import subprocess
import warnings
from pathlib import Path

__all__ = ['SimpleStringIO', 'copy_csv2md']


class PandasParquetIO:
    """Deprecated helper class to convert pandas to parquet and save to local
    path and vice-versa. No longer needed see:
    https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-parquet
    """

    def __init__(self, **kwargs):
        warnings.warn("Use pandas to_parquet instead", DeprecationWarning)


class SimpleStringIO:
    """Helper class to read/write stringlike objects to txt or json files
    at relative path
    Set kind to
        + 'txt' to read/write strings <-> text
        + 'json' to read/write dicts <-> json
    """

    def __init__(self, kind: str = 'txt'):
        assert kind in set(['txt', 'json']), "kind must be in {'txt', 'json'}"
        self.kind = kind

    def read(self, fqn: str) -> str:
        """Read a file from fqn according to kind of this object"""
        path = Path(fqn)
        if not path.exists():
            raise FileNotFoundError(f'Required file does not exist {str(path)}')
        with open(str(path), 'r') as f:
            s = f.read().rstrip('\n')
            f.close()
        if self.kind == 'json':
            s = json.loads(s)
        return s

    def write(self, s: str, fqn: str) -> str:
        path = Path(fqn)
        dr = Path(*path.parts[:-1])
        if not dr.is_dir():
            raise FileNotFoundError(f'Required dir does not exist {str(dr)}')
        if self.kind == 'json':
            s = json.dumps(s)
        with open(str(path), 'w') as f:
            f.write(f'{s}\n')
            f.close()
        return f'Written to {str(path)}'


def copy_csv2md(fqn: str) -> str:
    """Convenience to copy csv 'path/x.csv' to markdown 'path/x.md'"""
    assert os.path.exists(fqn)
    r = subprocess.run(['csv2md', f'{fqn}'], capture_output=True)
    with open(f'{fqn[:-3] + "md"}', 'wb') as f:
        f.write(r.stdout)
    return f'Created file {fqn} and {fqn[:-3]}md'
