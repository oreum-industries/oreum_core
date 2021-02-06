# eda_tools.py
# copyright 2021 Oreum OÜ
import re
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from IPython.display import display
from scipy import stats


def custom_describe(df, nrows=3, nfeats=30, limit=50e6, get_mode=False):
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
    dfdesc = df.describe().T
    for ft in dfdesc.columns[1:]:
        dfdesc[ft] = dfdesc[ft].apply(lambda x: np.round(x,2))

    # prepend random rows for example cases
    rndidx = np.random.randint(0,len(df),nrows)
    dfout = pd.concat((df.iloc[rndidx].T, dfdesc, df.dtypes), axis=1,join='outer', sort=True)
    dfout = dfout.loc[df.columns.values]
    dfout.rename(columns={0:'dtype'}, inplace=True)

    # add count, min, max for string cols (note the not very clever overwrite of count)
    dfout['count_notnull'] = df.shape[0] - df.isnull().sum()
    dfout['count_null'] = df.isnull().sum()
    dfout['min'] = df.min().apply(lambda x: x[:8] if type(x) == str else x)
    dfout['max'] = df.max().apply(lambda x: x[:8] if type(x) == str else x)
    dfout.index.name = 'ft'

    fts_out = ['dtype', 'count_notnull', 'count_null', 'mean', 'std', 
                'min', '25%', '50%', '75%', 'max']

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
    with pd.option_context('display.max_rows', max_rows, 'display.max_columns', None, 'display.max_colwidth', 200):
        display(df)


def plot_cat_count(df, fts, topn=10, vsize=2):
    """ Conv fn: plot group counts for cats """
    
    if len(fts) == 0:
        return None

    vert = int(np.ceil(len(fts)/2))
    f, ax2d = plt.subplots(vert, 2, squeeze=False, figsize=(14, vert*vsize))

    for i, ft in enumerate(fts):
        counts_all = df.groupby(ft).size().sort_values(ascending=True)
        counts = counts_all[-topn:]
        ax = counts.plot(
            kind='barh', ax=ax2d[i//2, i%2],
            title='{}: {} factor levels'.format(ft, len(counts_all)), 
            label='{} NaNs'.format(pd.isnull(df[ft]).sum()))
        _ = [ax.annotate('{} ({:.0%})'.format(c, c/counts_all.sum()), 
                xy=(c, i), xycoords='data', 
                xytext=(4, -2), textcoords='offset points', 
                ha='left', fontsize=10, color='#666666')
             for i, c in enumerate(counts)]
        ax.legend(loc='lower right')
        _ = ax.set_yticklabels([lbl.get_text()[:30] for lbl in ax.get_yticklabels()])
        
    f.tight_layout()
    
    
def plot_date_count(df, fts, fmt='%Y-%m', vsize=2):
    """ Plot group sizes for dates by strftime format """

    if len(fts) == 0:
        return None

    vert = int(np.ceil(len(fts)))
    f, ax1d = plt.subplots(vert, 1, figsize=(14, vert * vsize), squeeze=True)
    
    if vert > 1:
        for i, ft in enumerate(fts):
            ax = df[ft].groupby(df[ft].dt.strftime(fmt)).size().plot(kind='bar',
                ax=ax1d[i], title=ft, label='{} NaNs'.format(pd.isnull(df[ft]).sum()))
            ax.legend(loc='upper right')
    else:
        ft = fts[0]
        ax = df[ft].groupby(df[ft].dt.strftime(fmt)).size().plot(kind='bar',
                title=ft, label='{} NaNs'.format(pd.isnull(df[ft]).sum()))
        ax.legend(loc='upper right')
    
    f.tight_layout()
        
        
def plot_int_dist(df, fts, log=False, vsize=2.5):
    """ Plot group counts (optionally logged) for ints """

    if len(fts) == 0:
        return None

    vert = int(np.ceil(len(fts)))
    f, ax1d = plt.subplots(len(fts), 1, figsize=(14, vert*vsize), squeeze=False)
    for i, ft in enumerate(fts):
        ax = sns.histplot(df.loc[df[ft].notnull(), ft], 
                            kde=False, stat='density', ax=ax1d[i][0],
                            label='{} NaNs'.format(pd.isnull(df[ft]).sum()), 
                            color=sns.color_palette()[i%4])
        if log:
            _ = ax.set(yscale='log', title=ft, ylabel='log10(count)')
        _ = ax.set(title=ft, ylabel='count', xlabel='value')
        _ = ax.legend(loc='upper right')
    f.tight_layout()


def fit_fn(obs, dist_names=['invgamma'], title_insert=None, plot=True):
    """
    Fit `dists` to 1d array of `observations`, report MSE and plot the fits
    # see https://stackoverflow.com/a/37616966 
    TODO: refactor this messy hack (plot T/F)
    """
    dists_options = {'invgamma':stats.invgamma, 
                     'gamma':stats.gamma, 
                     'lognorm': stats.lognorm,
                     'gumbel': stats.gumbel_r,
                     'invgauss': stats.invgauss,
                     'invweibull': stats.invweibull,
                     'expon': stats.expon,
                     'norm': stats.norm,
                     'halfnorm': stats.halfnorm,
                     'cauchy': stats.cauchy,
                     'halfcauchy': stats.halfcauchy}
                     #TODO add other scipy stats fns
                    #  https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invgauss.html#scipy.stats.invgauss
    dists = {d: dists_options.get(d) for d in dist_names}
   
    nbins = 100
    density, bin_edges = np.histogram(obs, bins=nbins, density=True)
    bin_edges = (bin_edges + np.roll(bin_edges, -1))[:-1] / 2.0
    params = {}
    if plot:
        f, ax1d = plt.subplots(1, 1, figsize=(16, 5))
        ax = sns.histplot(x=obs, bins=nbins, stat='density', kde=False, label='data', ax=ax1d)

    for i, (d, dist) in enumerate(dists.items()):
        ps = dist.fit(obs, floc=0)
        shape, loc, scale = ps[:-2], ps[-2], ps[-1]
        params[d] = dict(shape=shape, loc=loc, scale=scale)

        pdf = dist.pdf(bin_edges, loc=loc, scale=scale, *shape)
        mse = np.sum(np.power(density - pdf, 2.0)) / len(density)
        rmse = np.sqrt(mse)
        
        if plot:
            ax = sns.lineplot(x=bin_edges, y=pdf, label=f'{d}: {rmse:.2g}', lw=1, ax=ax1d)
    if plot:
        title = (f'Function approximations to `{title_insert}`')
        _ = f.suptitle(title, y=1)
        _ = f.axes[0].legend(title='dist: RMSE', title_fontsize=10)
        return f, params
    else:
        return params


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


def plot_mincovdet(df, mcd, thresh=0.99):
    """ Interactive plot of MDC delta results """
    
    dfp = df.copy()
    dfp['mcd_delta'] = mcd.dist_
    dfp = dfp.sort_values('mcd_delta')
    dfp['counter'] = np.arange(dfp.shape[0])

    cutoff = np.percentile(dfp['mcd_delta'], thresh*100)
    dfp['mcd_outlier'] = dfp['mcd_delta'] > cutoff

    f = plt.figure(figsize=(14, 8))
    f.suptitle('Distribution of outliers' + \
               '\n(thresh @ {:.1%}, cutoff @ {:.1f}, identified {} outliers)'.format(
            thresh, cutoff, dfp['mcd_outlier'].sum()), fontsize=16)

    grd = plt.GridSpec(nrows=1, ncols=2, wspace=0.05, width_ratios=[3,1])

    # sorted MCD dist plot
    ax0 = plt.subplot(grd[0])
    sc = ax0.scatter(dfp['counter'], dfp['mcd_delta']
                ,c=dfp['counter'] / 1, cmap='YlOrRd'
                ,alpha=0.8, marker='o', linewidths=0.05, edgecolor='#999999')

    _ = ax0.axhline(y=cutoff, xmin=0, xmax=1, linestyle='--', color='#DD0011')
    _ = ax0.annotate('Thresh @ {:.1%}, cutoff @ {:.1f}'.format(thresh, cutoff)
                 ,xy=(0,cutoff), xytext=(10,10), textcoords='offset points'
                 ,color='#DD0011', style='italic', weight='bold', size='large')
    _ = ax0.set_xlim((-10, dfp.shape[0]+10))
    _ = ax0.set_yscale('log')
    _ = ax0.set_ylabel('MCD delta (log)')
    _ = ax0.set_xlabel('Datapoints sorted by increasing MCD delta')

    # summary boxplot
    ax1 = plt.subplot(grd[1], sharey=ax0)
    bx = ax1.boxplot(dfp['mcd_delta'], sym='k', showmeans=True
                     ,meanprops={'marker':'D', 'markersize':10
                                 ,'markeredgecolor':'k','markerfacecolor':'w'
                                 ,'markeredgewidth': 1})

    _ = ax1.axhline(y=cutoff,xmin=0, xmax=1, linestyle='--', color='#DD0011')
    _ = ax1.set_xlabel('Log MCD distance')
    #     _ = ax1.set_yticklabels([lbl.set_visible(False) for lbl in ax1.get_yticklabels()])
    _ = plt.setp(ax1.get_yticklabels(), visible=False)
    _ = plt.setp(bx['medians'],color='blue')

    return None


def get_gini(r, n):
    """ For array r, return estimate of gini co-efficient over n
        g = A / (A+B)
    """
    return 1 - sum(r.sort_values().cumsum() * (2 / n))


def plot_rmse_range(rmse, rmse_pct, lims=(0, 80)):
    """ Convenience to plot RMSE range with mins """
    dfp = rmse_pct.reset_index()
    dfp = dfp.loc[(dfp['pct'] >= lims[0]) & (dfp['pct'] <= lims[1])].copy()
    min_rmse = rmse_pct.min()
    min_rmse_pct = rmse_pct.index[rmse_pct.argmin()]

    f, axs = plt.subplots(1, 1, figsize=(12, 6))
    ax = sns.lineplot(x='pct', y='rmse', data=dfp, lw=2, ax=axs)
    #     _ = ax.set_yscale('log')
    _ = ax.axhline(rmse, c='r', ls='--', label=f'mean @ {rmse:,.0f}')
    _ = ax.axhline(rmse_pct[50], c='b', ls='--', label=f'median @ {rmse_pct[50]:,.0f}')
    _ = ax.axhline(min_rmse, c='g', ls='--', label=f'min @ pct {min_rmse_pct} @ {min_rmse:,.0f}')
    _ = f.suptitle('RMSE ranges', y=.95)
    _ = ax.legend()
    
    
def plot_rmse_range_pair(rmse_t, rmse_pct_t, rmse_h, rmse_pct_h, lims=(0, 80)):
    """ Convenience to plot two rmse pct results """
    f, axs = plt.subplots(1, 2, figsize=(14, 6))
    t = ['train', 'holdout']
    _ = f.suptitle('RMSE ranges', y=.97)

    for i, (rmse, rmse_pct) in enumerate(zip([rmse_t, rmse_h], [rmse_pct_t, rmse_pct_h])):
        dfp = rmse_pct.reset_index()
        dfp = dfp.loc[(dfp['pct'] >= lims[0]) & (dfp['pct'] <= lims[1])].copy()
        min_rmse = rmse_pct.min()
        min_rmse_pct = rmse_pct.index[rmse_pct.argmin()]
        
        ax = sns.lineplot(x='pct', y='rmse', data=dfp, lw=2, ax=axs[i])
        _ = ax.axhline(rmse, c='r', ls='--', label=f'mean @ {rmse:,.0f}')
        _ = ax.axhline(rmse_pct[50], c='b', ls='--', label=f'median @ {rmse_pct[50]:,.0f}')
        _ = ax.axhline(min_rmse, c='g', ls='--', label=f'min @ pct {min_rmse_pct} @ {min_rmse:,.0f}')
        _ = ax.legend()
        _ = ax.set_title(t[i])
    _ = f.tight_layout()


def plot_r2_range(r2, r2_pct, lims=(0, 80)):
    """ Convenience to plot R2 range with max 
    """
    dfp = r2_pct.reset_index()
    dfp = dfp.loc[(dfp['pct'] >= lims[0]) & (dfp['pct'] <= lims[1])].copy()
    max_r2 = r2_pct.max()
    max_r2_pct = r2_pct.index[r2_pct.argmax()]
    
    f, axs = plt.subplots(1, 1, figsize=(12, 6))
    ax = sns.lineplot(x='pct', y='r2', data=dfp, lw=2, ax=axs)
    _ = ax.axhline(r2, c='r', ls='--', label=f'mean @ {r2:,.2f}')
    _ = ax.axhline(r2_pct[50], c='b', ls='--', label=f'median @ {r2_pct[50]:,.2f}')
    _ = ax.axhline(max_r2, c='g', ls='--', label=f'max @ pct {max_r2_pct} @ {max_r2:,.2f}')
    _ = f.suptitle('$R^{2}$ ranges', y=.95)
    _ = ax.legend()


def plot_r2_range_pair(r2_t, r2_pct_t, r2_h, r2_pct_h, lims=(0, 80)):
    """ Convenience to plot two r2 pct results (t)raining vs (h)oldout
    """

    f, axs = plt.subplots(1, 2, figsize=(14, 6))
    t = ['train', 'holdout']
    _ = f.suptitle('$R^{2}$ ranges', y=.97)

    for i, (r2, r2_pct) in enumerate(zip([r2_t, r2_h], [r2_pct_t, r2_pct_h])):
        dfp = r2_pct.reset_index()
        dfp = dfp.loc[(dfp['pct'] >= lims[0]) & (dfp['pct'] <= lims[1])].copy()
        max_r2 = r2_pct.max()
        max_r2_pct = r2_pct.index[r2_pct.argmax()]
        
        ax = sns.lineplot(x='pct', y='r2', data=dfp, lw=2, ax=axs[i])
        _ = ax.axhline(r2, c='r', ls='--', label=f'mean @ {r2:,.2f}')
        _ = ax.axhline(r2_pct[50], c='b', ls='--', label=f'median @ {r2_pct[50]:,.0f}')
        _ = ax.axhline(max_r2, c='g', ls='--', label=f'min @ pct {max_r2_pct} @ {max_r2:,.0f}')
        _ = ax.legend()
        _ = ax.set_title(t[i])
    _ = f.tight_layout()