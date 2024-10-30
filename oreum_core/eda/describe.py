# Copyright 2024 Oreum Industries
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

# eda.describe.py
"""Data Descriptions"""
import logging

import numpy as np
import pandas as pd
from IPython.display import display
from scipy import stats

__all__ = ['describe', 'display_fw', 'display_ht', 'get_fts_by_dtype']

_log = logging.getLogger(__name__)

RSD = 42
RNG = np.random.default_rng(seed=RSD)


def describe(
    df: pd.DataFrame,
    nobs: int = 3,
    nfeats: int = 30,
    limit: int = 50,  # MB
    get_mode: bool = False,
    get_counts: bool = True,
    get_cr94: bool = False,
    reset_index: bool = True,
    return_df: bool = False,
    **kwargs,
) -> pd.DataFrame | None:
    """Concat transposed topN rows, numerical desc & dtypes
    Beware a dataframe full of bools or categoricals will error
    thanks to pandas.describe() being too clever
    Assume df has index.
    """

    len_idx = df.index.nlevels
    note = ''
    if nfeats + len_idx < df.shape[1]:
        note = 'NOTE: nfeats + index shown {} < width {}'.format(
            nfeats + len_idx, df.shape[1]
        )
    nbytes = df.values.nbytes
    _log.info(f'Shape: {df.shape}')
    _log.info(f'Memsize: {nbytes // 1e6:,.1f} MB')
    _log.info(f'Index levels: {df.index.names}')
    _log.info(f'{note}')

    limit *= 1e6
    if df.values.nbytes > limit:
        return f'Array memsize {nbytes // 1e6:,.1f} MB > {limit / 1e6:,.1f} MB limit'

    df = df.copy()
    if reset_index:
        idx_new_names = [f'index: {c}' for c in list(df.index.names)]
        col_names = list(df.columns.values)
        nfeats += len(idx_new_names)
        df = df.reset_index()
        df.columns = idx_new_names + col_names

    # start with pandas describe, add on dtypes
    quantiles = [0.25, 0.5, 0.75]  # the default
    percentile_names = ['25%', '50%', '75%']
    if get_cr94:
        quantiles = [0.03] + quantiles + [0.97]
        percentile_names = ['3%'] + percentile_names + ['97%']
    dfdesc = df.describe(include='all', percentiles=quantiles).T

    dfout = pd.concat((dfdesc, df.dtypes), axis=1, join='outer', sort=False)
    dfout = dfout.loc[df.columns.values]
    dfout.rename(columns={0: 'dtype', 'unique': 'count_unique'}, inplace=True)
    dfout.index.name = 'ft'

    # add counts for all
    if get_counts:
        dfout['count_notnull'] = df.shape[0] - df.isnull().sum()
        dfout['count_null'] = df.isnull().sum(axis=0)
        dfout['count_inf'] = (
            np.isinf(df.select_dtypes(np.number)).sum().reindex(df.columns)
        )
        dfout['count_zero'] = (
            (df.select_dtypes(np.number) == 0).sum(axis=0).reindex(df.columns)
        )

    # add sum for numeric cols
    dfout['sum'] = np.nan
    idxs = (dfout['dtype'] == 'float64') | (dfout['dtype'] == 'int64')
    if np.sum(idxs.values) > 0:
        for ft in dfout.loc[idxs].index.values:
            dfout.loc[ft, 'sum'] = df[ft].sum()

    # add min, max for string cols (note the not very clever overwrite of count)
    idxs = (dfout['dtype'] == 'object') | (dfout['dtype'] == 'string[python]')
    if np.sum(idxs.values) > 0:
        for ft in dfout.loc[idxs].index.values:
            dfout.loc[ft, 'min'] = df[ft].value_counts().index.min()
            dfout.loc[ft, 'max'] = df[ft].value_counts().index.max()

    fts_out_all = (
        [
            'dtype',
            'count_null',
            'count_inf',
            'count_zero',
            'count_unique',
            'top',
            'freq',
            'sum',
            'mean',
            'std',
            'min',
        ]
        + percentile_names
        + ['max']
    )
    fts_out = [f for f in fts_out_all if f in dfout.columns.values]

    # add mode and mode count WARNING takes forever for large arrays (>10k row)
    if get_mode:
        dfnn = df.select_dtypes(exclude=np.number).copy()
        r = stats.mode(dfnn, axis=0, nan_policy='omit')
        dfmode = pd.DataFrame(
            {'mode': r[0][0], 'mode_count': r[1][0]}, index=dfnn.columns
        )
        dfout = dfout.join(dfmode, how='left', left_index=True, right_index=True)
        fts_out.append(['mode', 'mode_count'])

    # select summary states and prepend random rows for example cases
    rndidx = RNG.choice(np.arange(0, len(df)), size=nobs, replace=False)
    dfout = pd.concat(
        (df.iloc[rndidx].T, dfout[fts_out].copy()), axis=1, join='outer', sort=False
    )
    dfout.index.name = 'ft'

    if return_df:
        return dfout
    else:
        display_fw(
            dfout.iloc[: nfeats + len_idx, :],
            max_rows=nfeats,
            shape=df.shape,
            nbytes=nbytes,
            **kwargs,
        )


def display_fw(df: pd.DataFrame, **kwargs) -> None:
    """Conv fn: contextually display max cols"""

    shape = kwargs.pop('shape', df.shape)
    nbytes = kwargs.pop('nbytes', df.values.nbytes)

    options = {
        'display.precision': kwargs.pop('precision', 2),
        'display.max_colwidth': kwargs.pop('max_colwidth', 30),
        'display.max_rows': kwargs.pop('max_rows', 50),
        'display.max_columns': None,
    }

    if kwargs.pop('latex', False):
        options['styler.render.repr'] = 'latex'
        options['styler.latex.environment'] = 'longtable'

    with pd.option_context(*[i for tup in options.items() for i in tup]):
        display(df)
        display(f'Shape: {shape}, Memsize {nbytes / 1e6:,.1f} MB')


def display_ht(df: pd.DataFrame, nrows=3, **kwargs) -> None:
    """Convenience fn: Display head and tail n rows via display_fw"""

    nrows = min(nrows, len(df))
    dfd = df.iloc[np.r_[0:nrows, -nrows:0]].copy()
    display_fw(dfd, shape=df.shape, nbytes=df.values.nbytes, **kwargs)


def get_fts_by_dtype(df: pd.DataFrame, as_dataframe: bool = False) -> dict:
    """Return a dictionary of lists of feats within df according to dtype"""
    dtypes = df.dtypes.to_dict().items()
    fts = dict(
        categorical=[k for k, v in dtypes if v.name[:3] == 'cat'],  # category
        cat=[
            k
            for k, v in df.dtypes.to_dict().items()
            if (v.name[:3] == 'obj') | (v.name[:3] == 'str')
        ],
        bool=[k for k, v in dtypes if v.name == 'bool'],
        boolean=[k for k, v in dtypes if v.name == 'boolean'],
        datetime=[k for k, v in dtypes if v.name[:3] == 'dat'],
        int=[k for k, v in dtypes if v.name[:3] == 'int'],
        float=[k for k, v in dtypes if v.name[:3] == 'flo'],
    )
    w = []
    for _, v in fts.items():
        w += v

    mismatch = list(set(df.columns) - set(w))
    if len(mismatch) > 0:
        raise ValueError(
            f'Failed to match a dtype to {mismatch}'
            + f'\nThese fts did match correctly: {fts}'
        )

    if as_dataframe:
        dtypes = ['categorical', 'cat', 'bool', 'boolean', 'datetime', 'int', 'float']
        d = {w: k for k, v in fts.items() for w in v}
        dfd = pd.DataFrame.from_dict(d, orient='index', columns=['dtype'])
        dfd.index.set_names('ft', inplace=True)
        dfd['dtype'] = pd.Categorical(dfd['dtype'], categories=dtypes, ordered=True)
        return dfd

    return fts
