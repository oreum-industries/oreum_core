# eda.descibe.py
# copyright 2021 Oreum OÜ
import numpy as np
import pandas as pd
from IPython.display import display
from scipy import stats

RANDOM_SEED = 42
rng = np.random.default_rng(seed=RANDOM_SEED)


def custom_describe(df, nrows=3, nfeats=30, limit=50e6, 
                    get_mode=False, round_numerics=False, reset_index=True):
    """ Concat transposed topN rows, numerical desc & dtypes 
        Beware a dataframe full of bools or categoricals will error 
        thanks to pandas.describe() being too clever
        Assume df has index. I
    """
    
    len_index = df.index.nlevels
    note = ''
    if nfeats + len_index < df.shape[1]:
        note = 'NOTE: nfeats+index shown {} < width {}'.format(nfeats + len_index, df.shape[1])

    print(f'Array shape: {df.shape}')
    print(f'Array memsize: {df.values.nbytes // 1000:,} kB')
    print(f'Index levels: {df.index.names}')
    print(f'{note}')

    if (df.values.nbytes > limit):
        return 'Array memsize > 50MB limit, avoid performing descriptions'

    if reset_index:
        df = df.copy().reset_index()

    # start with pandas and round numerics
    dfdesc = df.describe(include='all', datetime_is_numeric=True).T
    if round_numerics:
        for ft in dfdesc.columns[1:]:
            dfdesc[ft] = dfdesc[ft].apply(lambda x: np.round(x,3))

    # prepend random rows for example cases
    rndidx = np.random.randint(0,len(df),nrows)
    dfout = pd.concat((df.iloc[rndidx].T, dfdesc, df.dtypes), 
                      axis=1, join='outer', sort=True)
    dfout = dfout.loc[df.columns.values]
    dfout.rename(columns={0:'dtype'}, inplace=True)
    dfout.index.name = 'ft'

    # add null counts for all
    # dfout['count_notnull'] = df.shape[0] - df.isnull().sum()
    dfout['count_null'] = df.isnull().sum(axis=0)
    dfout['count_inf'] = np.isinf(df.select_dtypes(np.number)).sum().reindex(df.columns)
    
    # add min, max for string cols (note the not very clever overwrite of count)
    idxs = dfout['dtype'] == 'object'
    for ft in dfout.loc[idxs].index.values:
        dfout.loc[ft, 'min'] = df[ft].value_counts().index.min()
        dfout.loc[ft, 'max'] = df[ft].value_counts().index.max()

    fts_out_all = ['dtype', 'count_null', 'count_inf', 
                   'unique', 'top', 'freq',
                   'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    fts_out = [f for f in fts_out_all if f in dfout.columns.values]

    # add mode and mode count WARNING takes forever for large arrays (>10k row)
    if get_mode:
        dfnn = df.select_dtypes(exclude=np.number).copy()
        r = stats.mode(dfnn, axis=0, nan_policy='omit')
        dfmode = pd.DataFrame({'mode': r[0][0], 'mode_count': r[1][0]}, index=dfnn.columns)
        dfout = dfout.join(dfmode, how='left', left_index=True, right_index=True)
        fts_out.append(['mode', 'mode_count'])
    
    dfout = dfout[fts_out].copy()
    return dfout.iloc[:nfeats+len_index,:].fillna('')


def display_fw(df, max_rows=20):
    """ Conv fn: contextually display max rows """
    with pd.option_context('display.max_rows', max_rows, 
                           'display.max_columns', None, 
                           'display.max_colwidth', 200):
        display(df)


def get_fts_by_dtype(df):
    """Return a dictionary of lists of feats within df according to dtype 
    """
    fts = dict(
        cat = [k for k, v in df.dtypes.to_dict().items() if v.name[:3] == 'obj'],
        bool = [k for k, v in df.dtypes.to_dict().items() if v.name[:3] == 'boo'],
        date = [k for k, v in df.dtypes.to_dict().items() if v.name[:3] == 'dat'],
        int = [k for k, v in df.dtypes.to_dict().items() if v.name[:3] == 'int'],
        float = [k for k, v in df.dtypes.to_dict().items() if v.name[:3] == 'flo']
        )
    w = []
    for _, v in fts.items():
        w += v

    n = len(set(df.columns) - set(w))
    if n > 0:
        raise ValueError(f'Failed to match a dtype to {n} fts. Check again.' +
                         f'\nThese fts did match correctly: {fts}')
    return fts
