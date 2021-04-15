# curate.data_load.py
# copyright 2021 Oreum OÜ
import json
import os
import pyarrow
import pandas as pd
import pyarrow.parquet as pq
import subprocess

class PandasParquetIO:
    """ Helper class to convert pandas to parquet and save to local path
        and vice-versa
    """

    def __init__(self, relpath=[]):
        self.relpath = relpath

    def read_ppq(self, fn, relpath=[]):
        if len(relpath) == 0:
            relpath = self.relpath
        fqn = os.path.join(*relpath, f'{fn}.parquet')
        try:
            arw_df = pq.read_table(fqn, use_pandas_metadata=True)
        except OSError as e:
            raise e
        return pyarrow.Table.to_pandas(arw_df)

    def write_ppq(self, df, fn, relpath=[]):       
        if len(relpath) == 0:
            relpath = self.relpath
        fqn = os.path.join(*relpath, f'{fn}.parquet')
        arw_df = pyarrow.Table.from_pandas(df)
        try:
            pq.write_table(arw_df, fqn)
        except FileNotFoundError as e:
            raise e
        return f'Written to {fqn}'


class SimpleStringIO:
    """ Helper class to read/write stringlike objects to txt or json files 
        at relative path
        Set kind to 
            + 'txt' to read/write strings <-> text
            + 'json' to read/write dicts <-> json
    """

    def __init__(self, relpath=[], kind='txt'):
        self.relpath = relpath
        assert kind in set(['txt', 'json']), "kind must be in {'txt', 'json'}"
        self.kind = kind

    def read(self, fn, relpath=[]):
        if len(relpath) == 0:
            relpath = self.relpath
        fqn = os.path.join(*relpath, f'{fn}.{self.kind}')
        with open(fqn, 'r') as f:
            s = f.read()
        if self.kind == 'json':
            s = json.loads(s)
        return s

    def write(self, s, fn, relpath=[]):       
        if len(relpath) == 0:
            relpath = self.relpath
        fqn = os.path.join(*relpath, f'{fn}.{self.kind}')
        if self.kind =='json':
            s = json.dumps(s)
        with open(fqn, 'w') as f:
            f.write(s)
        return f'Written to {fqn}'


def copy_csv2md(fqn):
    """ Convenience to copy csv 'path/x.csv' to markdown 'path/x.md' """
    r = subprocess.run(['csv2md', f'{fqn}'], capture_output=True)
    with open(f'{fqn[:-3] + "md"}', 'wb') as f:
        f.write(r.stdout)
    return f'Created file {fqn} and {fqn[:-3]}md'


