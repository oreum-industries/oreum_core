# curate.data_load.py
# copyright 2021 Oreum OÜ
import os
import pyarrow
import pandas as pd
import pyarrow.parquet as pq
import subprocess
from time import sleep

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


def copy_csv2md(fqn):
    """ Convenience to copy csv 'path/x.csv' to markdown 'path/x.md' """
    r = subprocess.run(['csv2md', f'{fqn}', '>', f'{fqn[:-3] + "md"}'])
    # if r == 0:
    #     return f'Created files {fqn}.csv and {fqn}.md'
    # else:
    #     return r