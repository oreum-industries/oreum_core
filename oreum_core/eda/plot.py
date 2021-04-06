# eda.plot.py
# copyright 2021 Oreum OÜ
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.lines import Line2D
from scipy import stats, integrate

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
        

def plot_date_count(df, fts, fmt='%Y-%m', vsize=1.8):
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
        

def plot_int_dist(df, fts, log=False, vsize=1.8):
    """ Plot group counts (optionally logged) for ints """

    if len(fts) == 0:
        return None

    vert = int(np.ceil(len(fts)))
    f, ax1d = plt.subplots(len(fts), 1, figsize=(14, vert*vsize), squeeze=False)
    for i, ft in enumerate(fts):
        n_nans = pd.isnull(df[ft]).sum()
        mean = df[ft].mean()
        n_zeros = (df[ft] == 0).sum()
        ax = sns.histplot(df.loc[df[ft].notnull(), ft], kde=False, stat='density', 
                    label=f'NaNs: {n_nans}, zeros: {n_zeros}, mean: {mean:.2f}', 
                    color=sns.color_palette()[i%7], ax=ax1d[i][0])
        if log:
            _ = ax.set(yscale='log', title=ft, ylabel='log(count)')
        _ = ax.set(title=ft, ylabel='count', xlabel=None) #'value'
        _ = ax.legend(loc='upper right')
    f.tight_layout(pad=0.8)


def plot_float_dist(df, fts, log=False, sharex=False, sort=True):
    """ Plot distributions for floats, annotate count of nans and zeros """

    def _annotate_facets(data, **kwargs):
        """ Func to be mapped to the dataframe (named `data` by seaborn) 
            used per facet. Assume `data` is the simple result of a melt() 
            and has two fts: variable, value
        """ 
        n_nans = pd.isnull(data['value']).sum()
        n_zeros = (data['value'] == 0).sum()
        mean = data['value'].mean()
        ax = plt.gca()
        ax.text(.993, .93, f'NaNs: {n_nans}, zeros: {n_zeros}, mean: {mean:.2f}', 
                transform=ax.transAxes, ha='right', va='top', 
                backgroundcolor='w', fontsize=10)
    
    if len(fts) == 0:
        return None
    if sort:
        dfm = df[sorted(fts)].melt()
    else:
        dfm = df[fts].melt()
    g = sns.FacetGrid(row='variable', hue='variable', palette=sns.color_palette(),
                      data=dfm, height=1.6, aspect=6, sharex=sharex)
    _ = g.map(sns.violinplot, 'value', order='variable', cut=0, scale='count')
    _ = g.map(sns.pointplot, 'value', order='variable', color='C3', 
                estimator=np.mean, ci=94)
                # https://stackoverflow.com/q/33486613/1165112
                # scatter_kws=(dict(edgecolor='k', edgewidth=100)))
    _ = g.map_dataframe(_annotate_facets)

    if log:
        _ = g.set(xscale='log') #, title=ft, ylabel='log(count)')
    g.fig.tight_layout(pad=0.8)


def plot_joint_ft_x_tgt(df, ft, tgt, subtitle=None, colori=1):
    """ Jointplot of ft vs tgt distributions. Suitable for int or float """
    kde_kws = dict(zorder=0, levels=7, cut=0)
    nsamp = min(len(df), 200)
    g = sns.JointGrid(x=ft, y=tgt, 
                      data=df.sample(nsamp, random_state=RANDOM_SEED), height=6)
    _ = g.plot_joint(sns.kdeplot, **kde_kws, fill=True, color=f'C{colori%5}')
    _ = g.plot_marginals(sns.histplot, color=f'C{colori%5}')
    _ = g.ax_joint.text(.95, .95, 
            f"pearsonr = {stats.pearsonr(df[ft], df[tgt])[0]:.4g}", 
            transform=g.ax_joint.transAxes, ha='right')
    t = ('', 0.0) if subtitle is None else (f'\n{subtitle}', 0.04)
    _ = g.fig.suptitle(f'Joint dist: {ft} x {tgt}{t[0]}', y=1.02 + t[1])


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


def plot_roc_precrec(df):
    """ Plot ROC and PrecRec, also calc and return AUC
        Pass perf df from calc.calc_binary_performance_measures
    """

    roc_auc = integrate.trapezoid(y=df['tpr'], x=df['fpr'])
    prec_rec_auc = integrate.trapezoid(y=df['precision'], x=df['recall'])

    f, axs = plt.subplots(1, 2, figsize=(11.5, 6), sharex=True, sharey=True)
    _ = f.suptitle('ROC and Precision Recall Curves', y=1.0)

    _ = axs[0].plot(df['fpr'], df['tpr'], lw=2, marker='d', alpha=0.8,
                    label=f"ROC (auc={roc_auc:.2f})")
    _ = axs[0].plot((0, 1), (0, 1), '--', c='#cccccc', label='line of equiv')
    _ = axs[0].legend(loc='upper left')
    _ = axs[0].set(title='ROC curve', xlabel='FPR', ylabel='TPR')
    
    _ = axs[1].plot(df['recall'], df['precision'], lw=2, marker='o', alpha=0.8,
                    color='C3', label=f"PrecRec (auc={prec_rec_auc:.2f})")
    _ = axs[1].legend(loc='upper right')
    _ = axs[1].set(title='Precision Recall curve', xlabel='Recall', ylabel='Precision')

    f.tight_layout()

    return roc_auc, prec_rec_auc


def plot_f_measure(df):
    """ Plot F-measures (F0.5, F1, F2) at different percentiles """

    f1_at = df['f1'].argmax()
    dfm = df.reset_index()[['pct', 'f0.5', 'f1', 'f2']].melt(
                id_vars='pct', var_name='f-measure', value_name='f-score')
    f, axs = plt.subplots(1, 1, figsize=(6, 4))
    ax = sns.lineplot(x='pct', y='f-score', hue='f-measure', data=dfm, palette='Greens', lw=2, ax=axs)
    _ = ax.set_ylim(0, 1)
    _ = f.suptitle('F-scores across the percentage range of PPC' + 
            f'\nBest F1 = {df.loc[f1_at, "f1"]:.3f} @ {f1_at} pct', y=1.03)


def plot_accuracy(df):
    """ Plot accuracy at different percentiles """

    acc_at = df['accuracy'].argmax()
    f, axs = plt.subplots(1, 1, figsize=(6, 4))
    ax = sns.lineplot(x='pct', y='accuracy', color='C1', data=df, lw=2, ax=axs)
    _ = ax.set_ylim(0, 1)
    _ = f.suptitle('Accuracy across the percentage range of PPC' + 
            f'\nBest = {df.loc[acc_at, "accuracy"]:.1%} @ {acc_at} pct', y=1.03)


def plot_binary_performance(df, n=1):
    """ Plot ROC, PrecRec, F-score, Accuracy
        Pass perf df from calc.calc_binary_performance_measures
        Return summary stats
    """
    roc_auc = integrate.trapezoid(y=df['tpr'], x=df['fpr'])
    prec_rec_auc = integrate.trapezoid(y=df['precision'], x=df['recall'])

    f, axs = plt.subplots(1, 4, figsize=(18, 5), sharex=False, sharey=False)
    _ = f.suptitle(('Evaluations of Binary Classifier made by sweeping across '+ 
                    f'PPC quantiles\n(requires large n, here n={n})') , y=1.0)

    minpos = np.argmin(np.sqrt(df['fpr']**2 + (1-df['tpr'])**2))
    _ = axs[0].plot(df['fpr'], df['tpr'], lw=2, marker='d', alpha=0.8,
                    label=f"ROC (auc={roc_auc:.2f})")
    _ = axs[0].plot((0, 1), (0, 1), '--', c='#cccccc', label='line of equiv')
    _ = axs[0].plot(df.loc[minpos, 'fpr'], df.loc[minpos, 'tpr'], 
                    lw=2, marker='D', color='w', 
                    markeredgewidth=1, markeredgecolor='b', markersize=9,
                    label=f"Optimum ROC @ {minpos} pct")
    _ = axs[0].legend(loc='lower right')
    _ = axs[0].set(title='ROC curve', xlabel='FPR', ylabel='TPR', ylim=(0,1))
    
    _ = axs[1].plot(df['recall'], df['precision'], lw=2, marker='o', alpha=0.8,
                    color='C3', label=f"PrecRec (auc={prec_rec_auc:.2f})")
    _ = axs[1].legend(loc='upper right')
    _ = axs[1].set(title='Precision Recall curve', ylim=(0,1), 
                    xlabel='Recall', ylabel='Precision')

    f1_at = df['f1'].argmax()
    dfm = df.reset_index()[['pct', 'f0.5', 'f1', 'f2']].melt(
                id_vars='pct', var_name='f-measure', value_name='f-score')

    _ = sns.lineplot(x='pct', y='f-score', hue='f-measure', data=dfm, 
                        palette='Greens', lw=2, ax=axs[2])
    _ = axs[2].plot(f1_at, df.loc[f1_at, 'f1'], 
                    lw=2, marker='D', color='w', 
                    markeredgewidth=1, markeredgecolor='b', markersize=9,
                    label=f"Optimum F1 @ {f1_at} pct")
    _ = axs[2].legend(loc='upper left')
    _ = axs[2].set(title='F-scores across the PPC pcts' +
            f'\nBest F1 = {df.loc[f1_at, "f1"]:.3f} @ {f1_at} pct',
            xlabel='pct', ylabel='F-Score', ylim=(0, 1))

    acc_at = df['accuracy'].argmax()
    _ = sns.lineplot(x='pct', y='accuracy', color='C1', data=df, lw=2, ax=axs[3])
    _ = axs[3].text(x=0.04, y=0.04, 
                    s=('Class imbalance:' + 
                       f'\n0: {df["accuracy"].values[0]:.1%}' + 
                       f'\n1: {df["accuracy"].values[-1]:.1%}'),
                    transform=axs[3].transAxes, ha='left', va='bottom', 
                    backgroundcolor='w', fontsize=10)
    _ = axs[3].set(title='Accuracy across the PPC pcts' +
            f'\nBest = {df.loc[acc_at, "accuracy"]:.1%} @ {acc_at} pct',
            xlabel='pct', ylabel='Accuracy', ylim=(0, 1))

    f.tight_layout()
    return None


def plot_coverage(df, title_add=''):
    """ Convenience plot coverage from mt.calc_ppc_coverage """

    txt_kws = dict(color='#333333', xycoords='data', xytext=(2,-4), 
                textcoords='offset points', fontsize=11, backgroundcolor='w')

    g = sns.lmplot(x='cr', y='coverage', col='method', hue='method', data=df, 
                   fit_reg=False, height=5, scatter_kws={'s':70})
    
    for i, method in enumerate(df['method'].unique()):
        idx = df['method'] == method
        y = df.loc[idx, 'coverage'].values
        x = df.loc[idx, 'cr'].values
        ae = np.abs(y - x)
        auc = integrate.trapezoid(ae, x)

        g.axes[0][i].plot((0,1), (0,1), ls='--', color='#aaaaaa', zorder=-1)
        g.axes[0][i].fill_between(x, y, x, color='#bbbbbb', alpha=0.8, zorder=-1)
        g.axes[0][i].annotate(f'AUC={auc:.3f}', xy=(0, 1), **txt_kws)

    if title_add != '':
        title_add = f': {title_add}'
    g.fig.suptitle((f'PPC Coverage vs CR{title_add}' ), y=1.05)

    return None


def plot_rmse_range(rmse, rmse_pct, lims=(0, 80), yhat_name=''):
    """ Convenience to plot RMSE range with mins """
    dfp = rmse_pct.reset_index()
    dfp = dfp.loc[(dfp['pct'] >= lims[0]) & (dfp['pct'] <= lims[1])].copy()
    min_rmse = rmse_pct.min()
    min_rmse_pct = rmse_pct.index[rmse_pct.argmin()]

    f, axs = plt.subplots(1, 1, figsize=(10, 4))
    ax = sns.lineplot(x='pct', y='rmse', data=dfp, lw=2, ax=axs)
    #     _ = ax.set_yscale('log')
    _ = ax.axhline(rmse, c='r', ls='--', label=f'mean @ {rmse:,.2f}')
    _ = ax.axhline(rmse_pct[50], c='b', ls='--', label=f'median @ {rmse_pct[50]:,.2f}')
    _ = ax.axhline(min_rmse, c='g', ls='--', label=f'min @ pct {min_rmse_pct} @ {min_rmse:,.2f}')
    _ = f.suptitle(f'RMSE ranges {yhat_name}', y=.95)
    _ = ax.legend()
        

def plot_rmse_range_pair(rmse_t, rmse_pct_t, rmse_h, rmse_pct_h, lims=(0, 80), yhat_name=''):
    """ Convenience to plot two rmse pct results """
    f, axs = plt.subplots(1, 2, figsize=(14, 4))
    t = ['train', 'holdout']
    _ = f.suptitle('RMSE ranges {yhat_name}', y=.97)

    for i, (rmse, rmse_pct) in enumerate(zip([rmse_t, rmse_h], [rmse_pct_t, rmse_pct_h])):
        dfp = rmse_pct.reset_index()
        dfp = dfp.loc[(dfp['pct'] >= lims[0]) & (dfp['pct'] <= lims[1])].copy()
        min_rmse = rmse_pct.min()
        min_rmse_pct = rmse_pct.index[rmse_pct.argmin()]
        
        ax = sns.lineplot(x='pct', y='rmse', data=dfp, lw=2, ax=axs[i])
        _ = ax.axhline(rmse, c='r', ls='--', label=f'mean @ {rmse:,.2f}')
        _ = ax.axhline(rmse_pct[50], c='b', ls='--', label=f'median @ {rmse_pct[50]:,.2f}')
        _ = ax.axhline(min_rmse, c='g', ls='--', label=f'min @ pct {min_rmse_pct} @ {min_rmse:,.2f}')
        _ = ax.legend()
        _ = ax.set_title(t[i])
    _ = f.tight_layout()


def plot_r2_range(r2, r2_pct, lims=(0, 80), yhat_name=''):
    """ Convenience to plot R2 range with max 
    """
    dfp = r2_pct.reset_index()
    dfp = dfp.loc[(dfp['pct'] >= lims[0]) & (dfp['pct'] <= lims[1])].copy()
    max_r2 = r2_pct.max()
    max_r2_pct = r2_pct.index[r2_pct.argmax()]
    
    f, axs = plt.subplots(1, 1, figsize=(10, 4))
    ax = sns.lineplot(x='pct', y='r2', data=dfp, lw=2, ax=axs)
    _ = ax.axhline(r2, c='r', ls='--', label=f'mean @ {r2:,.2f}')
    _ = ax.axhline(r2_pct[50], c='b', ls='--', label=f'median @ {r2_pct[50]:,.2f}')
    _ = ax.axhline(max_r2, c='g', ls='--', label=f'max @ pct {max_r2_pct} @ {max_r2:,.2f}')
    _ = f.suptitle(f'$R^{2}$ ranges {yhat_name}', y=.95)
    _ = ax.legend()


def plot_r2_range_pair(r2_t, r2_pct_t, r2_h, r2_pct_h, lims=(0, 80)):
    """ Convenience to plot two r2 pct results (t)raining vs (h)oldout
    """

    f, axs = plt.subplots(1, 2, figsize=(14, 4))
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


def plot_ppc_vs_observed(y, yhat):
    """ Plot (quantile summaries of) yhat_ppc vs y """
    ps = [3, 10, 20, 30, 40, 50, 60, 70, 80, 90, 97]
    df_yhat_qs = pd.DataFrame(np.percentile(yhat, ps, axis=1).T, 
                                  columns=[f'q{p/100}' for p in ps])
    
    f, axs = plt.subplots(1, 1, figsize=(14, 5), sharey=True, sharex=True)
    _ = sns.kdeplot(y, cumulative=True, lw=2, c='g', ax=axs,
            common_norm=False, common_grid=True)

    if (df_yhat_qs.duplicated().sum() == len(df_yhat_qs) - 1):
        # all dupes: model was intercept only
        dfm = df_yhat_qs.iloc[:1].melt(var_name='ppc_q')
        _ = sns.rugplot(x='value', hue='ppc_q', data=dfm, 
                palette='coolwarm', lw=2, ls='-', height=1, 
                ax=axs, zorder=-1)
    else:
        dfm = df_yhat_qs.melt(var_name='ppc_q')
        _ = sns.kdeplot(x='value', hue='ppc_q', data=dfm, 
                cumulative=True, palette='coolwarm', lw=2, ls='-', 
                ax=axs, zorder=-1, common_norm=False, common_grid=True)

    _ = axs.set(xlim=(0, np.ceil(y.max())), ylim=(0, 1))



def plot_bootstrap_lr(dfboot, df, prm='premium', clm='claim', clm_ct='claim_ct',
                      title_add=''):
    """ Plot bootstrapped loss ratio, no grouping """
    
    mn_txt_kws = dict(color='#333333', xycoords='data', xytext=(10,8), 
                    textcoords='offset points', fontsize=8, backgroundcolor='w')
    pest_mn_kws = dict(markerfacecolor='C9', markeredgecolor='#999999', 
                    marker='d', markersize=10) 
    mn_kws = dict(markerfacecolor='w', markeredgecolor='k', marker='d', markersize=16)
    
    mn = dfboot[['lr']].mean().tolist()                          # boot mean
    pest_mn = [np.nan_to_num(df[clm], 0).sum() / df[prm].sum()]  # point est mean

    gd = sns.catplot(x='lr', data=dfboot, kind='violin', cut=0, height=2, aspect=6)
    _ = [gd.ax.plot(v, i%len(mn), **mn_kws) for i, v in enumerate(mn)]
    _ = [gd.ax.annotate(f'{v:.1%}', xy=(v, i%len(mn)), **mn_txt_kws) for i, v in enumerate(mn)]
    _ = [gd.ax.plot(v, i%len(pest_mn), **pest_mn_kws) for i, v in enumerate(pest_mn)]

    elems = [Line2D([0],[0], label='population (bootstrap)', **mn_kws), 
             Line2D([0],[0], label='sample', **pest_mn_kws)]
    gd.ax.legend(handles=elems, loc='lower right', title='Mean LRs')
    
    ypos = 1.34
    if title_add != '':
        ypos = 1.4
        title_add = f'\n{title_add}'

    title = f'Overall Loss Ratio (Population Estimate via Bootstrapping)'
    _ = gd.fig.suptitle((f'{title}{title_add}' + f'\n{len(df)} policies, ' + 
        f"\\${df[prm].sum()/1e6:.1f}M premium, " + 
        f"{df[clm_ct].sum():.0f} claims totalling \\${df[clm].sum()/1e6:.1f}M" + 
        f'\nEst. population mean LR = {mn[0]:.1%}, sample mean LR={pest_mn[0]:.1%}'), y=ypos)
    

def plot_bootstrap_lr_grp(dfboot, df, grp='grp', prm='premium', clm='claim', title_add=''):
    """ Plot bootstrapped loss ratio, grouped by grp """

    mn_txt_kws = dict(color='#333333', xycoords='data', xytext=(10, 8), 
                    textcoords='offset points', fontsize=8, backgroundcolor='w')
    pest_mn_kws = dict(markerfacecolor='C9', markeredgecolor='#999999', 
                    marker='d', markersize=10) 
    mn_kws = dict(markerfacecolor='w', markeredgecolor='k', marker='d', markersize=16)
    if dfboot[grp].dtypes != 'object':
        dfboot = dfboot.copy()
        dfboot[grp] = dfboot[grp].map(lambda x: f's{x}')

    mn = dfboot.groupby(grp)['lr'].mean().tolist()
    pest_mn = df.groupby(grp).apply(lambda g: np.nan_to_num(g[clm], 0).sum() / g[prm].sum()).values

    gd = sns.catplot(x='lr', y=grp, data=dfboot, kind='violin', cut=0, 
                     scale='count', width=0.6, palette='cubehelix_r', 
                     height=2+(len(mn)*.5), aspect=2+(len(mn)*0.05))
   
    _ = [gd.ax.plot(v, i%len(mn), **mn_kws) for i, v in enumerate(mn)]
    _ = [gd.ax.annotate(f'{v:.1%}', xy=(v, i%len(mn)), **mn_txt_kws) for i, v in enumerate(mn)]
    _ = [gd.ax.plot(v, i%len(pest_mn), **pest_mn_kws) for i, v in enumerate(pest_mn)]

    elems = [Line2D([0],[0], label='population (bootstrap)', **mn_kws), 
             Line2D([0],[0], label='sample', **pest_mn_kws)]
    gd.ax.legend(handles=elems, loc='lower right', title='Mean LRs')
    
    ypos = 1.05
    if title_add != '':
        ypos = 1.08
        title_add = f'\n{title_add}'

    title = (f'Gropued Loss Ratios (Population Estimates via Bootstrapping)' + 
            f' - grouped by {grp}')
    _ = gd.fig.suptitle(f'{title}{title_add}', y=ypos)
