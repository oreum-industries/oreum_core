# eda.plots.py
# copyright 2021 Oreum OÜ
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RANDOM_SEED = 42
rng = np.random.default_rng(seed=RANDOM_SEED)


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