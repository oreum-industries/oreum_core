# eda.descibe.py
# copyright 2021 Oreum OÜ
import numpy as np
import pandas as pd
from IPython.display import display
from scipy import stats

RANDOM_SEED = 42
rng = np.random.default_rng(seed=RANDOM_SEED)


def custom_describe(df, nrows=3, nfeats=30, limit=50e6, get_mode=False, round_numerics=False):
    """ Concat transposed topN rows, numerical desc & dtypes 
        Beware a dataframe full of bools or categoricals will error 
        thanks to pandas.describe() being too clever
    """
    
    note = ''
    if nfeats < df.shape[1]:
        note = '\nNOTE: nfeats shown {} < width {}'.format(nfeats, df.shape[1])

    print('Array shape: {}{}'.format(df.shape, note))
    print('Array memsize: {:,} bytes'.format(df.values.nbytes))

    if (df.values.nbytes > limit):
        return 'Array memsize > 50MB limit, avoid performing descriptions'

    # start with pandas and round numerics
    dfdesc = df.describe(include='all', datetime_is_numeric=True).T
    if round_numerics:
        for ft in dfdesc.columns[1:]:
            dfdesc[ft] = dfdesc[ft].apply(lambda x: np.round(x,3))

    # prepend random rows for example cases
    rndidx = np.random.randint(0,len(df),nrows)
    dfout = pd.concat((df.iloc[rndidx].T, dfdesc, df.dtypes), axis=1,join='outer', sort=True)
    dfout = dfout.loc[df.columns.values]
    dfout.rename(columns={0:'dtype'}, inplace=True)

    # add count, min, max for string cols (note the not very clever overwrite of count)
    # dfout['count_notnull'] = df.shape[0] - df.isnull().sum()
    dfout['count_null'] = df.isnull().sum(axis=0)
    dfout['count_inf'] = np.isinf(df.select_dtypes(np.number)).sum().reindex(df.columns)
    dfout['min'] = df.min().apply(lambda x: x[:8] if type(x) == str else x)
    dfout['max'] = df.max().apply(lambda x: x[:8] if type(x) == str else x)
    dfout.index.name = 'ft'

    fts_out = ['dtype', 'count_null', 'count_inf', 
                'mean', 'std', 'min', '25%', '50%', '75%', 'max']

    # add mode and mode count
    # WARNING takes forever for large (>10k row) arrays
    if get_mode:
        dfnn = df.select_dtypes(exclude=np.number).copy()
        r = stats.mode(dfnn, axis=0, nan_policy='omit')
        dfmode = pd.DataFrame({'mode': r[0][0], 'mode_count': r[1][0]}, index=dfnn.columns)
        dfout = dfout.join(dfmode, how='left', left_index=True, right_index=True)
        fts_out.append(['mode', 'mode_count'])
    
    dfout = dfout[fts_out].copy()
    return dfout.iloc[:nfeats,:]


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
