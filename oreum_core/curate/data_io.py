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

# curate.data_io.py
"""Data File Handling"""
import csv
import json
import logging
import subprocess
from pathlib import Path

import pandas as pd

from ..utils.file_io import BaseFileIO

__all__ = ['PandasParquetIO', 'PandasCSVIO', 'SimpleStringIO', 'copy_csv2md']

_log = logging.getLogger(__name__)


class PandasParquetIO(BaseFileIO):
    """Helper class to convert pandas to parquet and save to fqn and vice-versa.
    Not strictly needed, but adds a layer of path checking
    """

    def __init__(self, *args, **kwargs):
        """Inherit super"""
        super().__init__(*args, **kwargs)

    def read(self, fn: str, *args, **kwargs) -> pd.DataFrame:
        """Read parquet fn from rootdir, pass args kwargs to pd.read_parquet"""
        fn = Path(fn).with_suffix('.parquet')
        fqn = self.get_path_read(fn)
        _log.info(f'Read from {str(fqn.resolve())}')
        return pd.read_parquet(str(fqn), *args, **kwargs)

    def write(self, df: pd.DataFrame, fn: str, *args, **kwargs) -> Path:
        """Accept pandas DataFrame and fn e.g. `df.parquet`, write to fqn"""
        fqn = self.get_path_write(Path(self.snl.clean(fn)).with_suffix('.parquet'))
        df.to_parquet(str(fqn), *args, **kwargs)
        _log.info(f'Written to {str(fqn.resolve())}')
        return fqn


class PandasCSVIO(BaseFileIO):
    """Very simple helper class to write a Pandas dataframe to CSV file in a consistent way"""

    def __init__(self, *args, **kwargs):
        """Inherit super"""
        super().__init__(*args, **kwargs)

    def read(self, fn: str, *args, **kwargs) -> pd.DataFrame:
        """Read csv fn from rootdir, pass args kwargs to pd.read_csv"""
        fn = Path(fn).with_suffix('.csv')
        fqn = self.get_path_read(fn)
        _log.info(f'Read from {str(fqn.resolve())}')
        return pd.read_csv(str(fqn), *args, **kwargs)

    def write(self, df: pd.DataFrame, fn: str, *args, **kwargs) -> str:
        """Accept pandas DataFrame and fn e.g. `df`, write to fn.csv
        Consider using kwarg: float_format='%.3f'
        """
        fqn = self.get_path_write(Path(self.snl.clean(fn)).with_suffix('.csv'))
        kws = kwargs.copy()
        kws.update(quoting=csv.QUOTE_NONNUMERIC)
        if (len(df.index.names) == 1) & (df.index.names[0] is None):
            kws.update(index_label='rowid')
        df.to_csv(str(fqn), *args, **kws)
        _log.info(f'Written to {str(fqn.resolve())}')
        return fqn


class SimpleStringIO(BaseFileIO):
    """Helper class to read/write stringlike objects to txt or json files
    Set kind to
        + 'txt' to read/write strings <-> text file
        + 'json' to read/write dicts <-> json file
    """

    def __init__(self, kind: str = 'txt', *args, **kwargs):
        """Init for txt and json only"""
        super().__init__(*args, **kwargs)
        assert kind in set(['txt', 'json']), "kind must be in {'txt', 'json'}"
        self.kind = kind

    def read(self, fn: str) -> str:
        """Read a file from fn according to kind of this object"""
        fqn = self.get_path_read(fn)
        with open(str(fqn), 'r') as f:
            s = f.read().rstrip('\n')
            f.close()
        _log.info(f'Read text from {str(fqn.resolve())}')
        if self.kind == 'json':
            s = json.loads(s)
        return s

    def write(self, s: str, fn: str) -> str:
        fqn = self.get_path_write(Path(self.snl.clean(fn)).with_suffix(f'.{self.kind}'))
        if self.kind == 'json':
            s = json.dumps(s)
        with open(str(fqn), 'w') as f:
            f.write(f'{s}\n')
            f.close()
        _log.info(f'Written to {str(fqn.resolve())}')
        return fqn


def copy_csv2md(fn: str) -> str:
    """Convenience to copy csv 'path/x.csv' to markdown 'path/x.md'"""
    fileio = BaseFileIO()
    fqn = fileio.get_path_read(fn)
    r = subprocess.run(['csv2md', f'{fqn}'], capture_output=True)
    fn_out = f'{fn[:-3] + "md"}'
    fqn_out = fileio.get_path_write(fn_out)
    with open(fqn_out, 'wb') as f:
        f.write(r.stdout)
        f.close()
    _log.info(f'Written to {str(fqn_out.resolve())}')
    return fqn_out
