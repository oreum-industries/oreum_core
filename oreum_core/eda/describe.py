# eda.descibe.py
# copyright 2022 Oreum Industries
import os

import numpy as np
import pandas as pd
from IPython.display import display
from scipy import stats

RSD = 42
rng = np.random.default_rng(seed=RSD)


def display_fw(df, max_rows: int = 50, latex: bool = False):
    """Conv fn: contextually display max rows"""

    display_latex_repr = False
    display_latex_longtable = False
    if latex:
        display_latex_repr = True
        display_latex_longtable = True

    options = {
        'display.precision': 2,
        'display.max_rows': max_rows,
        'display.max_columns': None,
        'display.max_colwidth': 200,
        'display.latex.repr': display_latex_repr,
        'display.latex.longtable': display_latex_longtable,
    }

    with pd.option_context(*[i for tup in options.items() for i in tup]):
        display(df)


def display_ht(df, nrows: int = 3, latex: bool = False):
    """Convenience fn: Display head and tail n rows via display_fw"""

    dfd = df.iloc[np.r_[0:nrows, -nrows:0]].copy()
    display_fw(dfd, latex=latex)


def custom_describe(
    df,
    nrows=3,
    nfeats=30,
    limit=50e6,
    get_mode=False,
    reset_index=True,
    latex=False,
    return_df=False,
):
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

    print(f'Array shape: {df.shape}')
    print(f'Array memsize: {df.values.nbytes // 1000:,} kB')
    print(f'Index levels: {df.index.names}')
    print(f'{note}')

    if df.values.nbytes > limit:
        return 'Array memsize > 50MB limit, avoid performing descriptions'

    df = df.copy()
    if reset_index:
        nfeats += len(df.index.names)
        df = df.reset_index()

    # start with pandas describe, add on dtypes
    dfdesc = df.describe(include='all', datetime_is_numeric=True).T

    dfout = pd.concat((dfdesc, df.dtypes), axis=1, join='outer', sort=False)
    dfout = dfout.loc[df.columns.values]
    dfout.rename(columns={0: 'dtype', 'unique': 'count_unique'}, inplace=True)
    dfout.index.name = 'ft'

    # add null counts for all
    # dfout['count_notnull'] = df.shape[0] - df.isnull().sum()
    dfout['count_null'] = df.isnull().sum(axis=0)
    dfout['count_inf'] = np.isinf(df.select_dtypes(np.number)).sum().reindex(df.columns)
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
    idxs = dfout['dtype'] == 'object'
    if np.sum(idxs.values) > 0:
        for ft in dfout.loc[idxs].index.values:
            dfout.loc[ft, 'min'] = df[ft].value_counts().index.min()
            dfout.loc[ft, 'max'] = df[ft].value_counts().index.max()

    fts_out_all = [
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
        '25%',
        '50%',
        '75%',
        'max',
    ]
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
    rndidx = np.random.randint(0, len(df), nrows)
    dfout = pd.concat(
        (df.iloc[rndidx].T, dfout[fts_out].copy()), axis=1, join='outer', sort=False
    )
    dfout.index.name = 'ft'

    if return_df:
        return dfout
    else:
        display_fw(
            dfout.iloc[: nfeats + len_idx, :].fillna(''), max_rows=nfeats, latex=latex
        )


def get_fts_by_dtype(df, as_dataframe=False):
    """Return a dictionary of lists of feats within df according to dtype"""
    fts = dict(
        categorical=[
            k for k, v in df.dtypes.to_dict().items() if v.name[:3] == 'cat'
        ],  # category
        cat=[k for k, v in df.dtypes.to_dict().items() if v.name[:3] == 'obj'],
        bool=[k for k, v in df.dtypes.to_dict().items() if v.name[:3] == 'boo'],
        datetime=[k for k, v in df.dtypes.to_dict().items() if v.name[:3] == 'dat'],
        int=[k for k, v in df.dtypes.to_dict().items() if v.name[:3] == 'int'],
        float=[k for k, v in df.dtypes.to_dict().items() if v.name[:3] == 'flo'],
    )
    w = []
    for _, v in fts.items():
        w += v

    n = len(set(df.columns) - set(w))
    if n > 0:
        raise ValueError(
            f'Failed to match a dtype to {n} fts. Check again.'
            + f'\nThese fts did match correctly: {fts}'
        )

    if as_dataframe:
        dtypes = ['categorical', 'cat', 'bool', 'datetime', 'int', 'float']
        d = {w: k for k, v in fts.items() for w in v}
        dfd = pd.DataFrame.from_dict(d, orient='index', columns=['dtype'])
        dfd.index.set_names('ft', inplace=True)
        dfd['dtype'] = pd.Categorical(dfd['dtype'], categories=dtypes, ordered=True)
        return dfd

    return fts


def output_data_dict(df: pd.DataFrame, dd_notes: dict, dir_docs: list, fn: str = ''):
    """Convenience fn: output data dict"""

    # get desc overview
    nrows = 3
    dfd = custom_describe(df, nrows=nrows, return_df=True)
    cols = dfd.columns.values
    cols[:nrows] = [f'example_row_{i}' for i in range(nrows)]
    dfd.columns = cols

    # set dtypes categorical
    df_dtypes = get_fts_by_dtype(df.reset_index(), as_dataframe=True)
    dfd['dtype'] = df_dtypes['dtype']
    del df_dtypes

    # attached notes
    df_dd_notes = pd.DataFrame(dd_notes, index=['notes']).T
    df_dd_notes.index.name = 'ft'
    dfd = pd.merge(dfd, df_dd_notes, how='left', left_index=True, right_index=True)

    # write overview
    if fn != '':
        fn = f'_{fn}'
    writer = pd.ExcelWriter(
        os.path.join(*dir_docs, f'datadict{fn}.xlsx'), engine='xlsxwriter'
    )
    dfd.to_excel(writer, sheet_name='overview', index=True)

    # write cats to separate sheets for levels (but not indexes since they're unique)
    for ft in dfd.loc[dfd['dtype'].isin(['categorical', 'cat'])].index.values:
        if ft not in df.index.names:
            print(ft)
            dfg = (df[ft].value_counts(dropna=False) / len(df)).to_frame('prop')
            dfg.index.name = 'value'
            dfg.reset_index().to_excel(
                writer, sheet_name=ft, index=False, float_format='%.3f', na_rep='NULL'
            )

    writer.save()
