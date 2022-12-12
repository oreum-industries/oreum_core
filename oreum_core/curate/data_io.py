# curate.data_io.py
# copyright 2022 Oreum Industries
import csv
import json
import subprocess
from pathlib import Path

import pandas as pd

from oreum_core.file_io import BaseFileIO

__all__ = ['PandasParquetIO', 'PandasToCSV', 'SimpleStringIO', 'copy_csv2md']


class PandasParquetIO(BaseFileIO):
    """Helper class to convert pandas to parquet and save to fqn and vice-versa.
    Not strictly needed, but adds a layer of path checking
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def read(self, fqn: str) -> pd.DataFrame:
        """Read arviz.InferenceData object from fqn e.g. `model/mdl.netcdf`"""
        path = self.get_path_read(fqn)
        return pd.read_parquet(str(path))

    def write(self, df: pd.DataFrame, fqn: str) -> str:
        """Accept pandas DataFrame and fqn e.g. `data/df.parquet`, write to fqn"""
        path = self.get_path_write(fqn)
        df.to_parquet(str(path))
        return f'Written to {str(path)}'


class PandasToCSV(BaseFileIO):
    """Very simple helper class to write a Pandas dataframe to CSV fil in a consistent way"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def write(self, df: pd.DataFrame, fqn: str) -> str:
        """Accept pandas DataFrame and fqn e.g. `data/df.parquet`, write to fqn"""
        path = self.get_path_write(fqn)
        df.to_csv(str(path), index_label='rowid', quoting=csv.QUOTE_NONNUMERIC)
        return f'Written to {str(path)}'


class SimpleStringIO(BaseFileIO):
    """Helper class to read/write stringlike objects to txt or json files
    Set kind to
        + 'txt' to read/write strings <-> text file
        + 'json' to read/write dicts <-> json file
    """

    def __init__(self, kind: str = 'txt', *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert kind in set(['txt', 'json']), "kind must be in {'txt', 'json'}"
        self.kind = kind

    def read(self, fqn: str) -> str:
        """Read a file from fqn according to kind of this object"""
        path = self.get_path_read(fqn)
        with open(str(path), 'r') as f:
            s = f.read().rstrip('\n')
            f.close()
        if self.kind == 'json':
            s = json.loads(s)
        return s

    def write(self, s: str, fqn: str) -> str:
        path = self.get_path_write(fqn)
        if self.kind == 'json':
            s = json.dumps(s)
        with open(str(path), 'w') as f:
            f.write(f'{s}\n')
            f.close()
        return f'Written to {str(path)}'


def copy_csv2md(fqn: str) -> str:
    """Convenience to copy csv 'path/x.csv' to markdown 'path/x.md'"""
    path = Path(fqn)
    if not path.exists():
        raise FileNotFoundError(f'Required file does not exist {str(path)}')
    r = subprocess.run(['csv2md', f'{path}'], capture_output=True)
    with open(f'{path[:-3] + "md"}', 'wb') as f:
        f.write(r.stdout)
    return f'Created file {path} and {path[:-3]}md'
