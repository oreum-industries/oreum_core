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

# eda.plot.py
"""EDA Plotting"""
import logging
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regex as re
import seaborn as sns
from matplotlib import figure, gridspec, lines, ticker
from scipy import integrate, stats

from .calc import calc_svd
from .describe import get_fts_by_dtype

__all__ = [
    'plot_cat_ct',
    'plot_bool_ct',
    'plot_date_ct',
    'plot_int_dist',
    'plot_float_dist',
    'plot_joint_numeric',
    'plot_mincovdet',
    'plot_roc_precrec',
    'plot_f_measure',
    'plot_accuracy',
    'plot_binary_performance',
    'plot_coverage',
    'plot_rmse_range',
    # 'plot_r2_range',
    # 'plot_r2_range_pair',
    'plot_estimate',
    'plot_bootstrap_lr',
    'plot_bootstrap_lr_grp',
    'plot_bootstrap_grp',
    'plot_bootstrap_delta_grp',
    'plot_smrystat',
    'plot_smrystat_grp',
    'plot_smrystat_grp_year',
    'plot_heatmap_corr',
    'plot_kj_summaries_for_linear_model',
    'plot_grp_ct',
    'plot_cdf_ppc_vs_obs',
    'plot_explained_variance',
]


_log = logging.getLogger(__name__)
RSD = 42
rng = np.random.default_rng(seed=RSD)

sns.set_theme(
    style='darkgrid',
    palette='muted',
    context='notebook',
    rc={'figure.dpi': 72, 'savefig.dpi': 144, 'figure.figsize': (12, 4)},
)


def _get_kws_styling() -> dict:
    """Common styling kws for plots"""
    kws = dict(
        count_txt_kws=dict(
            color='#555555',
            fontsize=8,
            va='center',
            xycoords='data',
            textcoords='offset points',
        ),
        mn_pt_kws=dict(
            markerfacecolor='w', markeredgecolor='#333333', marker='d', markersize=12
        ),
        pest_mn_pt_kws=dict(
            markerfacecolor='#11c8bc',
            markeredgecolor='#eeeeee',
            marker='d',
            markersize=10,
        ),
        mn_txt_kws=dict(
            color='#555555',
            xycoords='data',
            xytext=(9, 11),
            textcoords='offset points',
            fontsize=8,
            backgroundcolor='w',
            ha='left',
        ),
        pest_mn_txt_kws=dict(
            color='#555555',
            xycoords='data',
            xytext=(-9, -12),
            textcoords='offset points',
            fontsize=6,
            backgroundcolor='#c8fdf9',
            ha='right',
        ),
    )
    kws['count_txt_h_kws'] = dict(ha='left', xytext=(4, 0), **kws['count_txt_kws'])
    return kws


def plot_cat_ct(
    df: pd.DataFrame,
    fts: list,
    topn: int = 10,
    vsize: float = 2,
    cat_order=True,
    **kwargs,
) -> figure.Figure:
    """Conv fn: plot group counts for cats"""

    # handle under/over selecting fts
    fts = list(set.intersection(set(df.columns.tolist()), set(fts)))
    if len(fts) == 0:
        return None

    vert = int(np.ceil(len(fts) / 2))
    f, ax2d = plt.subplots(vert, 2, squeeze=False, figsize=(12, 0.5 + vert * vsize))

    for i, ft in enumerate(fts):

        counts_all = df.groupby(ft).size().sort_values(ascending=True)
        if (df[ft].dtype == 'category') & cat_order:
            counts_all = df.groupby(ft).size()

        if df[ft].dtype == bool:
            counts_all = counts_all.sort_index()  # sort so true plots on top

        counts = counts_all.iloc[-topn:]
        ax = counts.plot(
            kind='barh',
            ax=ax2d[i // 2, i % 2],
            title='{}: {} factor levels'.format(ft, len(counts_all)),
            label='{} NaNs'.format(pd.isnull(df[ft]).sum()),
        )
        _ = [
            ax.annotate(
                '{} ({:.0%})'.format(c, c / counts_all.sum()),
                xy=(c, i),
                xycoords='data',
                xytext=(4, -2),
                textcoords='offset points',
                ha='left',
                fontsize=10,
                color='#666666',
            )
            for i, c in enumerate(counts)
        ]

        if df[ft].dtype != bool:
            ax.legend(loc='lower right')
        else:
            _ = ax.set(ylabel=None)

        _ = ax.set_yticklabels([lbl.get_text()[:30] for lbl in ax.get_yticklabels()])

    t = 'Empirical distribution'
    txtadd = kwargs.pop('txtadd', 'cats')
    _ = f.suptitle(' - '.join(filter(None, [t, txtadd])), y=1, fontsize=14)
    _ = f.tight_layout(pad=0.9)
    return f


def plot_bool_ct(
    df: pd.DataFrame, fts: list, vsize: float = 1.5, **kwargs
) -> figure.Figure:
    """Conv fn: plot group counts for bools"""

    # handle under/over selecting fts
    fts = list(set.intersection(set(df.columns.tolist()), set(fts)))
    if len(fts) == 0:
        return None

    vert = int(np.ceil(len(fts) / 2))
    f, ax2d = plt.subplots(vert, 2, squeeze=False, figsize=(12, 0.5 + vert * vsize))

    for i, ft in enumerate(fts):
        counts = df.groupby(ft, dropna=False).size().sort_values(ascending=True)
        counts = counts.sort_index()  # sort so true plots on top
        ax = counts.plot(
            kind='barh',
            ax=ax2d[i // 2, i % 2],
            title='{}: {} boolean levels'.format(ft, len(counts)),
        )
        _ = [
            ax.annotate(
                '{} ({:.0%})'.format(c, c / counts.sum()),
                xy=(c, i),
                xycoords='data',
                xytext=(4, -2),
                textcoords='offset points',
                ha='left',
                fontsize=10,
                color='#666666',
            )
            for i, c in enumerate(counts)
        ]
        _ = ax.set(ylabel=None)
        _ = ax.set_yticklabels([lbl.get_text()[:30] for lbl in ax.get_yticklabels()])

    t = 'Empirical distribution'
    txtadd = kwargs.pop('txtadd', 'bools')
    _ = f.suptitle(' - '.join(filter(None, [t, txtadd])), y=1, fontsize=14)
    f.tight_layout(pad=0.9)
    return f


def plot_date_ct(
    df: pd.DataFrame, fts: list, fmt: str = '%Y-%m', vsize: float = 1.6, **kwargs
) -> figure.Figure:
    """Plot group sizes for dates by strftime format"""

    # handle under/over selecting fts
    fts = list(set.intersection(set(df.columns.tolist()), set(fts)))
    if len(fts) == 0:
        return None

    vert = int(np.ceil(len(fts)))
    f, ax1d = plt.subplots(vert, 1, figsize=(12, 0.5 + vert * vsize), squeeze=True)

    if vert > 1:
        for i, ft in enumerate(fts):
            ax = (
                df[ft]
                .groupby(df[ft].dt.strftime(fmt))
                .size()
                .plot(
                    kind='bar',
                    ax=ax1d[i],
                    title=ft,
                    label='{} NaNs'.format(pd.isnull(df[ft]).sum()),
                )
            )
            ax.legend(loc='upper right')
    else:
        ft = fts[0]
        ax = (
            df[ft]
            .groupby(df[ft].dt.strftime(fmt))
            .size()
            .plot(kind='bar', title=ft, label='{} NaNs'.format(pd.isnull(df[ft]).sum()))
        )
        ax.legend(loc='upper right')
    t = 'Empirical distribution'
    txtadd = kwargs.pop('txtadd', 'dates')
    _ = f.suptitle(' - '.join(filter(None, [t, txtadd])), y=1, fontsize=14)
    _ = f.tight_layout(pad=0.9)
    return f


def plot_int_dist(
    df: pd.DataFrame,
    fts: list,
    log: bool = False,
    vsize: float = 1.5,
    bins: int = None,
    plot_zeros: bool = True,
    **kwargs,
) -> figure.Figure:
    """Plot group counts as histogram (optional log)"""
    # handle under/over selecting fts
    fts = list(set.intersection(set(df.columns.tolist()), set(fts)))
    if len(fts) == 0:
        return None
    if bins is None:
        bins = 'auto'

    vert = int(np.ceil(len(fts)))
    f, ax1d = plt.subplots(len(fts), 1, figsize=(12, 0.5 + vert * vsize), squeeze=False)
    for i, ft in enumerate(fts):
        n_nans = pd.isnull(df[ft]).sum()
        mean = df[ft].mean()
        med = df[ft].median()
        n_zeros = (df[ft] == 0).sum()
        if not plot_zeros:
            df = df.loc[df[ft] != 0].copy()
        ax = sns.histplot(
            df.loc[df[ft].notnull(), ft],
            kde=False,
            stat='count',
            bins=bins,
            label=f'NaNs: {n_nans}, zeros: {n_zeros}, mean: {mean:.2f}, med: {med:.2f}',
            color=sns.color_palette()[i % 7],
            ax=ax1d[i][0],
        )
        if log:
            _ = ax.set(yscale='log', title=ft, ylabel='log(count)')
        _ = ax.set(title=ft, ylabel='count', xlabel=None)  # 'value'
        _ = ax.legend(loc='upper right')
    t = 'Empirical distribution'
    txtadd = kwargs.pop('txtadd', 'ints')
    _ = f.suptitle(' - '.join(filter(None, [t, txtadd])), y=1, fontsize=14)
    _ = f.tight_layout(pad=0.9)
    return f


def plot_float_dist(
    df: pd.DataFrame,
    fts: list,
    log: bool = False,
    sharex: bool = False,
    sort: bool = True,
    **kwargs,
) -> figure.Figure:
    """
    Plot distributions for floats
    Annotate with count of nans, infs (+/-) and zeros
    """

    def _annotate_facets(data, **kwargs):
        """Func to be mapped to the dataframe (named `data` by seaborn)
        used per facet. Assume `data` is the simple result of a melt()
        and has two fts: variable, value
        """
        n_nans = pd.isnull(data['value']).sum()
        n_zeros = (data['value'] == 0).sum()
        n_infs = kwargs.pop('n_infs', 0)
        mean = data['value'].mean()
        med = data['value'].median()
        ax = plt.gca()
        ax.text(
            0.993,
            0.93,
            (
                f'NaNs: {n_nans},  infs+/-: {n_infs},  zeros: {n_zeros},  '
                + f'mean: {mean:.2f},  med: {med:.2f}'
            ),
            transform=ax.transAxes,
            ha='right',
            va='top',
            backgroundcolor='w',
            fontsize=10,
        )

    # handle under/over selecting fts
    fts = list(set.intersection(set(df.columns.tolist()), set(fts)))
    if len(fts) == 0:
        return None

    if sort:
        dfm = df[sorted(fts)].melt(var_name='variable')
    else:
        dfm = df[fts].melt(var_name='variable')

    idx_inf = np.isinf(dfm['value'])
    dfm = dfm.loc[~idx_inf].copy()

    gd = sns.FacetGrid(
        row='variable',
        hue='variable',
        data=dfm,
        palette=sns.color_palette(),
        height=1.6,
        aspect=8,
        sharex=sharex,
    )
    _ = gd.map(sns.violinplot, 'value', order='variable', cut=0, scale='count')
    _ = gd.map(
        sns.pointplot,
        'value',
        order='variable',
        color='C3',
        estimator=np.mean,
        errorbar=('ci', 94),
    )
    # https://stackoverflow.com/q/33486613/1165112
    # scatter_kws=(dict(edgecolor='k', edgewidth=100)))
    _ = gd.map_dataframe(_annotate_facets, n_infs=sum(idx_inf))

    if log:
        _ = gd.set(xscale='log')  # , title=ft, ylabel='log(count)')

    txtadd = kwargs.pop('txtadd', 'floats')
    t = 'Empirical distribution'
    _ = gd.fig.suptitle(' - '.join(filter(None, [t, txtadd])), y=1, fontsize=14)
    _ = gd.fig.tight_layout(pad=0.9)
    return gd.fig


def plot_joint_numeric(
    data: pd.DataFrame,
    ft0: str,
    ft1: str,
    hue: str = None,
    kind: str = 'kde',
    height: int = 6,
    kdefill: bool = True,
    log: Literal['x', 'y', 'both'] = None,
    colori: int = 0,
    nsamp: int = None,
    linreg: bool = True,
    legendpos: str = None,
    palette_type: Literal['q', 'g'] = 'g',
    palette: str = None,
    eq: int = 7,  # equal quantiles. Set higher in the case of extreme values
    **kwargs,
) -> figure.Figure:
    """Jointplot of 2 numeric fts with optional: hue shading, linear regression
    Suitable for int or float"""

    dfp = data.copy()
    ngrps = 1
    kws = dict(color=f'C{colori%7}')  # color rotation max 7

    if nsamp is not None:
        dfp = dfp.sample(nsamp, random_state=RSD).copy()

    nobs = len(dfp)

    if hue is not None:
        ngrps = len(dfp[hue].unique())
        nobs = len(dfp) // ngrps
        if palette is None:
            ftsd = get_fts_by_dtype(dfp)
            linreg = False
            if hue in ftsd['int'] + ftsd['float']:  # bin into n equal quantiles
                dfp[hue] = pd.qcut(dfp[hue].values, q=eq, duplicates='drop')
                kws['palette'] = 'viridis'
                nobs = len(dfp)
            else:
                if palette_type == 'g':
                    kws['palette'] = sns.color_palette(
                        [f'C{i + colori%7}' for i in range(ngrps)]
                    )
                else:  # palette_type == 'q':
                    kws['palette'] = sns.color_palette(
                        palette='RdYlBu_r', n_colors=ngrps
                    )
        elif isinstance(palette, str):
            kws['palette'] = sns.color_palette(palette=palette, n_colors=ngrps)
        else:  # pass a palette directly
            kws['palette'] = palette

    gd = sns.JointGrid(x=ft0, y=ft1, data=dfp, height=height, hue=hue)

    kde_kws = kws | dict(
        zorder=0, levels=7, cut=0, fill=kdefill, legend=True, warn_singular=False
    )
    scatter_kws = kws | dict(
        alpha=0.6, marker='o', linewidths=0.05, edgecolor='#dddddd', s=50
    )
    reg_kws = kws | dict(scatter_kws=scatter_kws, robust=True)
    rug_kws = kws | dict(height=0.1, legend=False)

    if kind == 'kde':
        _ = gd.plot_joint(sns.kdeplot, **kde_kws)
    elif kind == 'scatter':
        _ = gd.plot_joint(sns.scatterplot, **scatter_kws)
        _ = gd.plot_marginals(sns.rugplot, **rug_kws)
    elif kind == 'kde+scatter':
        _ = gd.plot_joint(sns.kdeplot, **kde_kws)
        _ = gd.plot_joint(sns.scatterplot, **scatter_kws)
        _ = gd.plot_marginals(sns.rugplot, **rug_kws)
    elif kind == 'reg':
        _ = gd.plot_joint(sns.regplot, **reg_kws)
        _ = gd.plot_marginals(sns.rugplot, **rug_kws)
    else:
        raise ValueError('kwarg `kind` must be in {kde, scatter, kde+scatter, reg}')

    if legendpos is not None:
        _ = sns.move_legend(gd.ax_joint, legendpos)

    _ = gd.plot_marginals(sns.histplot, kde=True, **kws)

    if linreg:
        r = stats.linregress(x=dfp[ft0], y=dfp[ft1])
        _ = gd.ax_joint.text(
            0.98,
            0.98,
            f"y = {r.slope:.2f}x + {r.intercept:.2f}\nρ = {r.rvalue:.2f}",
            transform=gd.ax_joint.transAxes,
            ha='right',
            va='top',
            fontsize=8,
        )

    if log in ['x', 'both']:
        _ = gd.ax_joint.set_xscale('log')
        _ = gd.ax_marg_x.set_xscale('log')

    if log in ['y', 'both']:
        _ = gd.ax_joint.set_yscale('log')
        _ = gd.ax_marg_y.set_yscale('log')

    t = f'Joint & marginal dists: `{ft0}` vs `{ft1}`, {nobs} obs'
    txtadd = kwargs.pop('txtadd', None)
    _ = gd.fig.suptitle('\n'.join(filter(None, [t, txtadd])), y=1, fontsize=14)
    _ = gd.fig.tight_layout(pad=0.95)
    return gd.fig


def plot_mincovdet(df: pd.DataFrame, mcd, thresh: float = 0.99):
    """Interactive plot of MCD delta results"""

    dfp = df.copy()
    dfp['mcd_delta'] = mcd.dist_
    dfp = dfp.sort_values('mcd_delta')
    dfp['counter'] = np.arange(dfp.shape[0])

    cutoff = np.percentile(dfp['mcd_delta'], thresh * 100)
    dfp['mcd_outlier'] = dfp['mcd_delta'] > cutoff

    f = plt.figure(figsize=(14, 8))
    f.suptitle(
        'Distribution of outliers'
        + '\n(thresh @ {:.1%}, cutoff @ {:.1f}, identified {} outliers)'.format(
            thresh, cutoff, dfp['mcd_outlier'].sum()
        ),
        fontsize=14,
    )

    grd = plt.GridSpec(nrows=1, ncols=2, wspace=0.05, width_ratios=[3, 1])

    # sorted MCD dist plot
    ax0 = plt.subplot(grd[0])
    _ = ax0.scatter(
        dfp['counter'],
        dfp['mcd_delta'],
        c=dfp['counter'] / 1,
        cmap='YlOrRd',
        alpha=0.8,
        marker='o',
        linewidths=0.05,
        edgecolor='#999999',
    )

    _ = ax0.axhline(y=cutoff, xmin=0, xmax=1, linestyle='--', color='#DD0011')
    _ = ax0.annotate(
        'Thresh @ {:.1%}, cutoff @ {:.1f}'.format(thresh, cutoff),
        xy=(0, cutoff),
        xytext=(10, 10),
        textcoords='offset points',
        color='#DD0011',
        style='italic',
        weight='bold',
        size='large',
    )
    _ = ax0.set_xlim((-10, dfp.shape[0] + 10))
    _ = ax0.set_yscale('log')
    _ = ax0.set_ylabel('MCD delta (log)')
    _ = ax0.set_xlabel('Datapoints sorted by increasing MCD delta')

    # summary boxplot
    ax1 = plt.subplot(grd[1], sharey=ax0)
    bx = ax1.boxplot(
        dfp['mcd_delta'],
        sym='k',
        showmeans=True,
        meanprops={
            'marker': 'D',
            'markersize': 10,
            'markeredgecolor': 'k',
            'markerfacecolor': 'w',
            'markeredgewidth': 1,
        },
    )

    _ = ax1.axhline(y=cutoff, xmin=0, xmax=1, linestyle='--', color='#DD0011')
    _ = ax1.set_xlabel('Log MCD distance')
    #     _ = ax1.set_yticklabels([lbl.set_visible(False) for lbl in ax1.get_yticklabels()])
    _ = plt.setp(ax1.get_yticklabels(), visible=False)
    _ = plt.setp(bx['medians'], color='blue')

    return None


def plot_roc_precrec(df: pd.DataFrame) -> tuple[figure.Figure, float, float]:
    """Plot ROC and PrecRec, also calc and return AUC
    Pass perf df from calc.calc_binary_performance_measures
    """

    roc_auc = integrate.trapezoid(y=df['tpr'], x=df['fpr'])
    prec_rec_auc = integrate.trapezoid(y=df['precision'], x=df['recall'])

    f, axs = plt.subplots(1, 2, figsize=(11.5, 6), sharex=True, sharey=True)
    _ = f.suptitle('ROC and Precision Recall Curves', y=1.0)

    _ = axs[0].plot(
        df['fpr'],
        df['tpr'],
        lw=2,
        marker='d',
        alpha=0.8,
        label=f"ROC (auc={roc_auc:.2f})",
    )
    _ = axs[0].plot((0, 1), (0, 1), '--', c='#cccccc', label='line of equiv')
    _ = axs[0].legend(loc='upper left')
    _ = axs[0].set(title='ROC curve', xlabel='FPR', ylabel='TPR')

    _ = axs[1].plot(
        df['recall'],
        df['precision'],
        lw=2,
        marker='o',
        alpha=0.8,
        color='C3',
        label=f"PrecRec (auc={prec_rec_auc:.2f})",
    )
    _ = axs[1].legend(loc='upper right')
    _ = axs[1].set(title='Precision Recall curve', xlabel='Recall', ylabel='Precision')

    f.tight_layout()

    return f, roc_auc, prec_rec_auc


def plot_f_measure(df: pd.DataFrame) -> figure.Figure:
    """Plot F-measures (F0.5, F1, F2) at different percentiles"""

    f1_at = df['f1'].argmax()
    dfm = df.reset_index()[['pct', 'f0.5', 'f1', 'f2']].melt(
        id_vars='pct', var_name='f-measure', value_name='f-score'
    )
    f, axs = plt.subplots(1, 1, figsize=(6, 4))
    ax = sns.lineplot(
        x='pct', y='f-score', hue='f-measure', data=dfm, palette='Greens', lw=2, ax=axs
    )
    _ = ax.set_ylim(0, 1)
    _ = f.suptitle(
        'F-scores across the percentage range of PPC'
        + f'\nBest F1 = {df.loc[f1_at, "f1"]:.3f} @ {f1_at} pct',
        y=1.03,
    )
    return f


def plot_accuracy(df: pd.DataFrame) -> figure.Figure:
    """Plot accuracy at different percentiles"""

    acc_at = df['accuracy'].argmax()
    f, axs = plt.subplots(1, 1, figsize=(6, 4))
    ax = sns.lineplot(x='pct', y='accuracy', color='C1', data=df, lw=2, ax=axs)
    _ = ax.set_ylim(0, 1)
    _ = f.suptitle(
        'Accuracy across the percentage range of PPC'
        + f'\nBest = {df.loc[acc_at, "accuracy"]:.1%} @ {acc_at} pct',
        y=1.03,
    )
    return f


def plot_binary_performance(
    dfperf: pd.DataFrame, nobs: int = 1, **kwargs
) -> figure.Figure:
    """Plot ROC, PrecRec, F-score, Accuracy sweeping across PPC sample quantiles
    Created for perf df from model_pymc.calc.calc_binary_performance_measures
    Return summary stats
    """
    f, axs = plt.subplots(1, 4, figsize=(18, 5), sharex=False, sharey=False)

    # ROC -------------
    roc_auc = integrate.trapezoid(y=dfperf['tpr'], x=dfperf['fpr'])
    r_at = np.argmin(np.sqrt(dfperf['fpr'] ** 2 + (1 - dfperf['tpr']) ** 2))
    r_at = np.round(r_at / 100, 2)

    _ = axs[0].plot(
        dfperf['fpr'],
        dfperf['tpr'],
        lw=2,
        marker='d',
        alpha=0.8,
        label=f"ROC (auc={roc_auc:.2f})",
    )
    _ = axs[0].plot((0, 1), (0, 1), '--', c='#cccccc', label='line of equiv')
    _ = axs[0].plot(
        dfperf.loc[r_at, 'fpr'],
        dfperf.loc[r_at, 'tpr'],
        lw=2,
        marker='D',
        color='w',
        markeredgewidth=1,
        markeredgecolor='b',
        markersize=9,
        label=f"Optimum ROC @ q{r_at}",
    )
    _ = axs[0].legend(loc='lower right')
    _ = axs[0].set(title='ROC curve', xlabel='FPR', ylabel='TPR', ylim=(0, 1))

    # Precision-Recall -------------
    prec_rec_auc = integrate.trapezoid(y=dfperf['precision'], x=dfperf['recall'])
    _ = axs[1].plot(
        dfperf['recall'],
        dfperf['precision'],
        lw=2,
        marker='o',
        alpha=0.8,
        color='C3',
        label=f"PrecRec (auc={prec_rec_auc:.2f})",
    )
    _ = axs[1].legend(loc='upper right')
    _ = axs[1].set(
        title='Precision Recall curve', ylim=(0, 1), xlabel='Recall', ylabel='Precision'
    )

    # F-measure -------------
    f1_at = np.round(dfperf['f1'].argmax() / 100, 2)
    dfm = dfperf.reset_index()[['q', 'f0.5', 'f1', 'f2']].melt(
        id_vars='q', var_name='f-measure', value_name='f-score'
    )

    _ = sns.lineplot(
        x='q', y='f-score', hue='f-measure', data=dfm, palette='Greens', lw=2, ax=axs[2]
    )
    _ = axs[2].plot(
        f1_at,
        dfperf.loc[f1_at, 'f1'],
        lw=2,
        marker='D',
        color='w',
        markeredgewidth=1,
        markeredgecolor='b',
        markersize=9,
        label=f"Optimum F1 @ q{f1_at}",
    )
    _ = axs[2].legend(loc='upper left')
    _ = axs[2].set(
        title='F-measures across the PPC qs'
        + f'\nBest F1 = {dfperf.loc[f1_at, "f1"]:.3f} @ q{f1_at}',
        xlabel='q',
        ylabel='F-Score',
        ylim=(0, 1),
    )

    # Accuracy -------------
    acc_at = np.round(dfperf['accuracy'].argmax() / 100, 2)
    _ = sns.lineplot(x='q', y='accuracy', color='C1', data=dfperf, lw=2, ax=axs[3])
    _ = axs[3].text(
        x=0.04,
        y=0.04,
        s=(
            'Class imbalance:'
            + f'\n0: {dfperf["accuracy"].values[0]:.1%}'
            + f'\n1: {dfperf["accuracy"].values[-1]:.1%}'
        ),
        transform=axs[3].transAxes,
        ha='left',
        va='bottom',
        backgroundcolor='w',
        fontsize=10,
    )
    _ = axs[3].set(
        title='Accuracy across the PPC pcts'
        + f'\nBest = {dfperf.loc[acc_at, "accuracy"]:.1%} @ q{acc_at}',
        xlabel='q',
        ylabel='Accuracy',
        ylim=(0, 1),
    )

    t = (
        'Evaluations of Binary Predictions made by sweeping across PPC '
        + f'sample quantiles (more reliable if nobs>100: here nobs={nobs})'
    )
    txtadd = kwargs.get('txtadd', None)
    _ = f.suptitle('\n'.join(filter(None, [t, txtadd])), y=1.0)
    _ = f.tight_layout()
    return f


def plot_coverage(df: pd.DataFrame, **kwargs) -> figure.Figure:
    """Convenience plot coverage from mt.calc_ppc_coverage"""

    txt_kws = dict(
        color='#555555',
        xycoords='data',
        xytext=(2, -4),
        textcoords='offset points',
        fontsize=11,
        backgroundcolor='w',
    )

    g = sns.lmplot(
        x='cr',
        y='coverage',
        col='method',
        hue='method',
        data=df,
        fit_reg=False,
        height=4,
        scatter_kws={'s': 70},
    )
    txtadd = kwargs.get('txtadd', None)
    for i, method in enumerate(df['method'].unique()):
        idx = df['method'] == method
        y = df.loc[idx, 'coverage'].values
        x = df.loc[idx, 'cr'].values
        ae = np.abs(y - x)
        auc = integrate.trapezoid(ae, x)

        g.axes[0][i].plot((0, 1), (0, 1), ls='--', color='#aaaaaa', zorder=-1)
        g.axes[0][i].fill_between(x, y, x, color='#bbbbbb', alpha=0.8, zorder=-1)
        g.axes[0][i].annotate(f'AUC={auc:.3f}', xy=(0, 1), **txt_kws)

    t = 'PPC Coverage vs CR'
    _ = g.fig.suptitle(' - '.join(filter(None, [t, txtadd])), y=1.05, fontsize=14)

    return g.fig


def plot_rmse_range(
    rmse: float, rmse_qs: pd.Series, yhat: str = 'yhat', y: str = 'y', **kwargs
) -> figure.Figure:
    """Convenience to plot RMSE mean and qs from model_pymc.calc.calc_rmse
    Requires that `rmse_qs` has an entry at the median 0.5"""
    txtadd = kwargs.get('txtadd', None)
    dfp = rmse_qs.to_frame()
    min_rmse_qs = rmse_qs.min()
    min_rmse_qs_q = rmse_qs.idxmin()
    j = max(-int(np.ceil(np.log10(rmse))) + 2, 1)

    f, axs = plt.subplots(1, 1, figsize=(10, 4))
    ax = sns.lineplot(x='q', y='rmse', data=dfp, lw=2, ax=axs)
    _ = ax.axhline(rmse, c='r', ls='-.', label=f'rmse @ mean {rmse:,.{j}f}')
    _ = ax.axhline(
        rmse_qs[0.5], c='b', ls='--', label=f'rmse @ med (q0.5) {rmse_qs[0.5]:,.{j}f}'
    )
    _ = ax.axhline(
        min_rmse_qs,
        c='g',
        ls='--',
        label=f'rmse @ min (q{min_rmse_qs_q:,.{j}f}) {min_rmse_qs:,.{j}f}',
    )
    _ = ax.legend()
    t = f'RMSE range for `{yhat}` vs `{y}`'
    tq = f'qs in [{rmse_qs.index.min()}, {rmse_qs.index.max()}]'
    _ = f.suptitle(' - '.join(filter(None, [t, tq, txtadd])))
    _ = f.tight_layout()
    return f


def plot_estimate(
    df: pd.DataFrame,
    nobs: int,
    yhat: str = 'yhat',
    force_xlim: list = None,
    color: str = None,
    kind: str = 'box',
    arr_overplot: np.array = None,
    **kwargs,
) -> figure.Figure:
    """Plot distribution for univariate estimates, either PPC or bootstrapped
    no grouping. Optionally overplot bootstrapped dfboot"""
    # TODO: Extend this to multivariate grouping
    txtadd = kwargs.pop('txtadd', None)
    sty = _get_kws_styling()
    clr = color if color is not None else sns.color_palette()[0]
    _kws = dict(
        box=dict(
            kind='box',
            sym='',
            orient='h',
            showmeans=True,
            whis=(3, 97),
            meanprops=sty['mn_pt_kws'],
        ),
        violin=dict(kind='violin', cut=0),
        exceedance=dict(kind='ecdf', complementary=True, lw=2, legend=None),
    )
    kws = _kws.get(kind)

    mn = df[[yhat]].mean().tolist()  # estimated mean
    j = max(-int(np.ceil(np.log10(mn[0]))) + 2, 0)

    if kind == 'exceedance':
        qs = kwargs.pop('qs', [0.5, 0.9, 0.95, 0.99])
        txtadd = ' - '.join(filter(None, ['Exceedance Curve', txtadd]))
        gd = sns.displot(x=yhat, data=df, **kws, height=4, aspect=2.5)
        _ = gd.axes[0][0].set(ylabel=f'P({yhat} > x)')  # , xlabel='dollars')
        # _ = gd.axes[0][0].xaxis.set_major_formatter('${x:,.1f}')
        df_qvals = df[[yhat]].quantile(qs)
        df_qvals.index.name = 'quantile'
        df_qvals = df_qvals.reset_index()
        df_qvals['p_gt'] = 1 - df_qvals['quantile'].values
        for q in qs:
            _ = gd.axes[0][0].vlines(
                df_qvals.loc[df_qvals['quantile'] == q, yhat].values,
                0,
                1 - q,
                linestyles='dotted',
                zorder=-1,
                colors='#999999',
            )
        _ = sns.scatterplot(
            x=yhat,
            y='p_gt',
            style='p_gt',
            data=df_qvals,
            markers=['*', 'd', '^', 'o'],
            ax=gd.axes[0][0],
            s=100,
        )
        handles, labels = gd.axes[0][0].get_legend_handles_labels()
        lbls = [str(round(float(lb), 2)) for lb in labels]  # HACK floating point
        gd.axes[0][0].legend(handles, lbls, loc='upper right', title=f'P({yhat}) > x')
        v = re.sub('hat$', '', yhat)
        summary = ',  '.join(
            df_qvals[['p_gt', yhat]]
            .apply(
                lambda r: f'$P(\hat{{{v}}})_{{{r[0]:.2f}}} \geq {{{r[1]:.{j}f}}}$',
                axis=1,
            )
            .tolist()
        )

    elif kind == 'box':
        gd = sns.catplot(x=yhat, data=df, **kws, color=clr, height=2.5, aspect=4)
        _ = [
            gd.ax.annotate(f'{v:,.{1}f}', xy=(v, i % len(mn)), **sty['mn_txt_kws'])
            for i, v in enumerate(mn)
        ]
        elems = [lines.Line2D([0], [0], label=f'mean {yhat}', **sty['mn_pt_kws'])]
        if arr_overplot is not None:
            mn_arr_overplot = arr_overplot.mean()  # estimated mean
            j_arr_overplot = -int(np.ceil(np.log10(mn_arr_overplot))) + 2
            ax = sns.pointplot(
                arr_overplot,
                estimator=np.mean,
                errorbar=('ci', 94),
                color='C1',
                linestyles='-',
                orient='h',
            )
            mn_txt_kws = sty['mn_txt_kws']
            mn_txt_kws['backgroundcolor'] = 'C1'
            _ = ax.annotate(
                f'{mn_arr_overplot:,.{j_arr_overplot}f}',
                xy=(mn_arr_overplot, 0),
                **mn_txt_kws,
            )
            mn_pt_kws = sty['mn_pt_kws']
            mn_pt_kws.update(
                markerfacecolor='C1', markeredgecolor='C1', marker='o', markersize=8
            )
            nm = kwargs.get('arr_overplot_nm', 'overplot')
            elems.append(lines.Line2D([0], [0], label=f'mean {nm}', **mn_pt_kws))

        gd.ax.legend(handles=elems, loc='upper right', fontsize=8)

        if force_xlim is not None:
            _ = gd.ax.set(xlim=force_xlim)

        hdi = df[yhat].quantile(q=[0.03, 0.1, 0.25, 0.5, 0.75, 0.9, 0.97]).values
        summary = (
            f'$\mu = {mn[0]:,.{j}f}$, '
            + f'$q_{{50}} = {hdi[3]:,.{j}f}$, '
            + f'$HDI_{{50}} = [{hdi[2]:,.{j}f}, {hdi[4]:,.{j}f}]$, '
            + f'$HDI_{{80}} = [{hdi[1]:,.{j}f}, {hdi[5]:,.{j}f}]$, '
            + f'$HDI_{{94}} = [{hdi[0]:,.{j}f}, {hdi[6]:,.{j}f}]$'
        )

    t = f'Summary Distribution of {yhat} estimate for {nobs} obs'
    _ = gd.fig.suptitle(' - '.join(filter(None, [t, txtadd])) + f'\n{summary}')
    _ = gd.fig.tight_layout()
    return gd.fig


def plot_bootstrap_lr(
    dfboot: pd.DataFrame,
    df: pd.DataFrame,
    prm: str = 'premium',
    clm: str = 'claim',
    clm_ct: str = 'claim_ct',
    ftname_year: str = 'incept_year',
    pol_summary: bool = True,
    lr_summary: bool = True,
    force_xlim: list = None,
    color: str = None,
    **kwargs,
) -> figure.Figure:
    """Plot bootstrapped loss ratio, no grouping"""

    sty = _get_kws_styling()
    mn = dfboot[['lr']].mean().tolist()  # boot mean
    hdi = dfboot['lr'].quantile(q=[0.03, 0.25, 0.75, 0.97]).values  # boot qs
    pest_mn = [np.nan_to_num(df[clm], 0).sum() / df[prm].sum()]  # point est mean
    clr = color if color is not None else sns.color_palette()[0]

    gd = sns.catplot(
        x='lr', data=dfboot, kind='violin', cut=0, color=clr, height=3, aspect=4
    )
    _ = [gd.ax.plot(v, i % len(mn), **sty['mn_pt_kws']) for i, v in enumerate(mn)]
    _ = [
        gd.ax.annotate(f'{v:.1%}', xy=(v, i % len(mn)), **sty['mn_txt_kws'])
        for i, v in enumerate(mn)
    ]
    _ = [
        gd.ax.plot(v, i % len(pest_mn), **sty['pest_mn_pt_kws'])
        for i, v in enumerate(pest_mn)
    ]
    _ = [
        gd.ax.annotate(f'{v:.1%}', xy=(v, i % len(pest_mn)), **sty['pest_mn_txt_kws'])
        for i, v in enumerate(pest_mn)
    ]
    elems = [
        lines.Line2D([0], [0], label='population LR (bootstrap)', **sty['mn_pt_kws']),
        lines.Line2D([0], [0], label='sample LR', **sty['pest_mn_pt_kws']),
    ]
    gd.ax.legend(handles=elems, loc='upper right', fontsize=8)
    if force_xlim is not None:
        _ = gd.ax.set(xlim=force_xlim)
    gd.ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    # hacky way to deal with year as int or datetime
    pmin = df[ftname_year].min()
    pmax = df[ftname_year].max()
    if np.issubdtype(df[ftname_year].dtype, np.datetime64):
        pmin = pmin.year
        pmax = pmax.year

    summary = ''
    if pol_summary:
        summary += (
            f"Inception {str(pmin)} - {str(pmax)} inclusive, "
            + f'{len(df):,.0f} policies with '
            + f"\\${df[prm].sum()/1e6:.1f}M premium, "
            + f"{df[clm_ct].sum():,.0f} claims totalling "
            + f"\\${df[clm].sum()/1e6:.1f}M"
        )
    if lr_summary:
        summary += (
            f'\nPopulation LR: mean = {mn[0]:.1%}, '
            + f'$HDI_{{50}}$ = [{hdi[1]:.1%}, {hdi[2]:.1%}], '
            + f'$HDI_{{94}}$ = [{hdi[0]:.1%}, {hdi[3]:.1%}]'
        )
    txtadd = kwargs.pop('txtadd', None)
    t = 'Bootstrapped Distribution of Population Loss Ratio'
    t = ' - '.join(filter(None, [t, txtadd]))
    _ = gd.fig.suptitle('\n'.join(filter(None, [t, summary])), y=1, fontsize=14)
    _ = gd.fig.tight_layout()
    return gd.fig


def plot_bootstrap_lr_grp(
    dfboot: pd.DataFrame,
    df: pd.DataFrame,
    grp: str = 'grp',
    prm: str = 'premium',
    clm: str = 'claim',
    clm_ct: str = 'claim_ct',
    ftname_year: str = 'incept_year',
    pol_summary: bool = True,
    force_xlim: list = None,
    annot_pest: bool = False,
    orderby: Literal['ordinal', 'count', 'lr'] = 'ordinal',
    **kwargs,
) -> figure.Figure:
    """Plot bootstrapped loss ratio, grouped by grp"""

    dfboot = dfboot.copy()
    df = df.copy()
    sty = _get_kws_styling()

    # hacky way to deal with year as int or datetime
    pmin = df[ftname_year].min()
    pmax = df[ftname_year].max()
    if np.issubdtype(df[ftname_year].dtype, np.datetime64):
        pmin = pmin.year
        pmax = pmax.year

    # hacky way to convert year as datetime to int
    # if np.issubdtype(df[grp].dtype, np.datetime64): error for categoricals
    if dfboot[grp].dtype == '<M8[ns]':
        dfboot[grp] = dfboot[grp].dt.year.astype(int)
        df[grp] = df[grp].dt.year.astype(int)

    # convert non object type to string
    if not dfboot[grp].dtype in ['object', 'category']:
        dfboot = dfboot.copy()
        dfboot[grp] = dfboot[grp].map(lambda x: f's{x}')
        df = df.copy()
        df[grp] = df[grp].map(lambda x: f's{x}')

    ct = df.groupby(grp, observed=True).size()
    mn = dfboot.groupby(grp, observed=True)['lr'].mean()
    pest_mn = df.groupby(grp, observed=True).apply(
        lambda g: np.nan_to_num(g[clm], 0).sum() / g[prm].sum()
    )

    # create order items / index
    if orderby == 'count':
        order_idx = df.groupby(grp).size().sort_values()[::-1].index
    elif orderby == 'ordinal':
        order_idx = mn.index
    elif orderby == 'lr':
        order_idx = mn.sort_values()[::-1].index
    else:
        return 'choose better'

    # reorder accordingly
    ct = ct.reindex(order_idx).values
    mn = mn.reindex(order_idx).values
    pest_mn = pest_mn.reindex(order_idx).values

    f = plt.figure(figsize=(14, 2.5 + (len(mn) * 0.25)))  # , constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[11, 1], figure=f)
    ax0 = f.add_subplot(gs[0])
    ax1 = f.add_subplot(gs[1], sharey=ax0)

    # add violinplot
    v_kws = dict(kind='violin', cut=0, scale='count', width=0.6, palette='cubehelix_r')
    _ = sns.violinplot(
        x='lr', y=grp, data=dfboot, ax=ax0, order=order_idx.values, **v_kws
    )

    _ = [ax0.plot(v, i % len(mn), **sty['mn_pt_kws']) for i, v in enumerate(mn)]
    _ = [
        ax0.annotate(f'{v:.1%}', xy=(v, i % len(mn)), **sty['mn_txt_kws'])
        for i, v in enumerate(mn)
    ]
    _ = [
        ax0.plot(v, i % len(pest_mn), **sty['pest_mn_pt_kws'])
        for i, v in enumerate(pest_mn)
    ]
    if annot_pest:
        _ = [
            ax0.annotate(f'{v:.1%}', xy=(v, i % len(pest_mn)), **sty['pest_mn_txt_kws'])
            for i, v in enumerate(pest_mn)
        ]
    ax0.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    elems = [
        lines.Line2D([0], [0], label='population LR (bootstrap)', **sty['mn_pt_kws']),
        lines.Line2D([0], [0], label='sample LR', **sty['pest_mn_pt_kws']),
    ]
    _ = ax0.legend(handles=elems, loc='upper right', fontsize=8)
    if force_xlim is not None:
        _ = ax0.set(xlim=force_xlim)

    # add countplot
    _ = sns.countplot(
        y=grp, data=df, order=order_idx.values, ax=ax1, palette='cubehelix_r'
    )
    _ = [
        ax1.annotate(f'{v}', xy=(v, i % len(ct)), **sty['count_txt_h_kws'])
        for i, v in enumerate(ct)
    ]

    summary = ''
    if pol_summary:
        summary += (
            f"Inception {str(pmin)} - {str(pmax)} inclusive, "
            + f'{len(df):,.0f} policies with '
            + f"\\${df[prm].sum()/1e6:.1f}M premium, "
            + f"{df[clm_ct].sum():,.0f} claims totalling "
            + f"\\${df[clm].sum()/1e6:.1f}M"
        )

    txtadd = kwargs.pop('txtadd', None)
    t = f'Bootstrapped Distributions of Population Loss Ratio, grouped by {grp}'
    t = ' - '.join(filter(None, [t, txtadd]))
    _ = f.suptitle('\n'.join(filter(None, [t, summary])), y=1, fontsize=14)
    _ = f.tight_layout()
    return f


def plot_bootstrap_grp(
    dfboot: pd.DataFrame,
    df: pd.DataFrame,
    grp: str = 'grp',
    val: str = 'y_eloss',
    title_add: str = '',
    force_xlim=None,
) -> figure.Figure:
    """Plot bootstrapped value, grouped by grp"""
    sty = _get_kws_styling()

    if not dfboot[grp].dtypes in ['object', 'category']:
        dfboot = dfboot.copy()
        dfboot[grp] = dfboot[grp].map(lambda x: f's{x}')

    mn = dfboot.groupby(grp)[val].mean().tolist()
    pest_mn = df.groupby(grp)[val].mean().values

    f = plt.figure(figsize=(14, 1.5 + (len(mn) * 0.25)))  # , constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[11, 1], figure=f)
    ax0 = f.add_subplot(gs[0])
    ax1 = f.add_subplot(gs[1], sharey=ax0)

    _ = sns.violinplot(
        x=val,
        y=grp,
        data=dfboot,
        kind='violin',
        cut=0,
        scale='count',
        width=0.6,
        palette='cubehelix_r',
        ax=ax0,
    )

    _ = [ax0.plot(v, i % len(mn), **sty['mn_pt_kws']) for i, v in enumerate(mn)]
    _ = [
        ax0.annotate(f'{v:,.0f}', xy=(v, i % len(mn)), **sty['mn_txt_kws'])
        for i, v in enumerate(mn)
    ]
    _ = [
        ax0.plot(v, i % len(pest_mn), **sty['pest_mn_pt_kws'])
        for i, v in enumerate(pest_mn)
    ]
    _ = [
        ax0.annotate(f'{v:,.0f}', xy=(v, i % len(pest_mn)), **sty['pest_mn_txt_kws'])
        for i, v in enumerate(pest_mn)
    ]

    elems = [
        lines.Line2D([0], [0], label='population (bootstrap)', **sty['mn_pt_kws']),
        lines.Line2D([0], [0], label='sample', **sty['pest_mn_pt_kws']),
    ]
    _ = ax0.legend(handles=elems, loc='lower right', title='Mean Val')

    if force_xlim is not None:
        _ = ax0.set(xlim=force_xlim)

    _ = sns.countplot(y=grp, data=df, ax=ax1, palette='cubehelix_r')
    ct = df.groupby(grp).size().tolist()
    _ = [
        ax1.annotate(f'{v}', xy=(v, i % len(ct)), **sty['count_txt_h_kws'])
        for i, v in enumerate(ct)
    ]

    if title_add != '':
        title_add = f'\n{title_add}'
    title = (
        'Grouped Mean Value (Population Estimates via Bootstrapping)'
        + f' - grouped by {grp}'
    )
    _ = f.suptitle(f'{title}{title_add}')
    _ = plt.tight_layout()
    return f


def plot_bootstrap_delta_grp(dfboot, df, grp, force_xlim=None, title_add=''):
    """Plot delta between boostrap results, grouped"""

    sty = _get_kws_styling()
    if dfboot[grp].dtypes != 'object':
        dfboot = dfboot.copy()
        dfboot[grp] = dfboot[grp].map(lambda x: f's{x}')

    mn = dfboot.groupby(grp).size()

    f = plt.figure(figsize=(14, 2 + (len(mn) * 0.2)))  # , constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[11, 1], figure=f)
    ax0 = f.add_subplot(gs[0])
    ax1 = f.add_subplot(gs[1], sharey=ax0)

    _ = sns.boxplot(
        x='lr_delta',
        y=grp,
        data=dfboot,
        palette='cubehelix_r',
        sym='',
        whis=[3, 97],
        showmeans=True,
        notch=True,
        ax=ax0,
    )
    _ = ax0.axvline(0, ls='--', lw=2, c='#555555', zorder=-1)

    if force_xlim is not None:
        _ = ax0.set(xlim=force_xlim)

    _ = sns.countplot(y=grp, data=df, ax=ax1, palette='cubehelix_r')
    ct = df.groupby(grp).size().tolist()
    _ = [
        ax1.annotate(f'{v}', xy=(v, i % len(ct)), **sty['count_txt_h_kws'])
        for i, v in enumerate(ct)
    ]

    if title_add != '':
        title_add = f'\n{title_add}'

    title = f'2-sample bootstrap test - grouped by {grp}'
    _ = f.suptitle(f'{title}{title_add}')

    _ = plt.tight_layout()  # prefer over constrained_layout
    return gs


def plot_smrystat(
    df: pd.DataFrame,
    val: str = 'y_eloss',
    smry: Literal['sum', 'mean'] = 'sum',
    plot_outliers: bool = True,
    palette: sns.palettes._ColorPalette = None,
    **kwargs,
) -> figure.Figure:
    """Plot diagnostics (smrystat, dist) of numeric value `val`"""
    sty = _get_kws_styling()
    idx = df[val].notnull()
    dfp = df.loc[idx].copy()

    f = plt.figure(figsize=(12, 2))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], figure=f)
    ax0 = f.add_subplot(gs[0])
    ax1 = f.add_subplot(gs[1])

    ax0.set_title(f'Distribution of bootstrapped {smry}')
    ax1.set_title('Distribution of indiv. values')

    if palette is None:
        palette = 'viridis'

    estimator = np.sum if smry == 'sum' else np.mean
    _ = sns.pointplot(
        x=val,
        data=dfp,
        palette=palette,
        estimator=estimator,
        errorbar=('ci', 94),
        ax=ax0,
    )

    sym = 'k' if plot_outliers else ''
    _ = sns.boxplot(
        x=val,
        data=dfp,
        palette=palette,
        sym=sym,
        whis=[3, 97],
        showmeans=True,
        meanprops=sty['mn_pt_kws'],
        ax=ax1,
    )

    txtadd = kwargs.pop('txtadd', None)
    t = f'Diagnostic 1D plots of `{val}`'
    _ = f.suptitle('\n'.join(filter(None, [t, txtadd])), y=1, fontsize=14)

    if sum(idx) > 0:
        t = (
            f'Note: {sum(~idx):,.0f} NaNs found in value,'
            f'\nplotted non-NaN dataset of {sum(idx):,.0f}'
        )
        _ = ax1.annotate(
            t, xy=(0.94, 0.94), xycoords='figure fraction', ha='right', fontsize=8
        )

    ax0.xaxis.label.set_visible(False)
    ax1.xaxis.label.set_visible(False)
    _ = plt.tight_layout()
    return f


def plot_smrystat_grp(
    df: pd.DataFrame,
    grp: str = 'grp',
    grpkind: str = None,
    val: str = 'y_eloss',
    smry: Literal['sum', 'mean'] = 'sum',
    plot_outliers: bool = True,
    plot_compact: bool = True,
    plot_grid: bool = True,
    pal: sns.palettes._ColorPalette = None,
    orderby: Literal['ordinal', 'count', 'smrystat', None] = 'ordinal',
    **kwargs,
) -> figure.Figure:
    """Plot diagnostics (smrystat, dist, count) of numeric value `val`
    grouped by categorical value `grp`, with group, ordered by count desc
    """
    sty = _get_kws_styling()
    est = np.sum if smry == 'sum' else np.mean
    idx = df[val].notnull()
    dfp = df.loc[idx].copy()

    if grpkind == 'year':
        dfp[grp] = dfp[grp].dt.year

    if not dfp[grp].dtypes in ['object', 'category', 'string']:
        dfp[grp] = dfp[grp].map(lambda x: f's{x}')

    ct = dfp.groupby(grp, observed=True).size()
    smrystat = dfp.groupby(grp, observed=True)[val].apply(est)

    # create order items / index
    if orderby == 'count':
        ct = ct.sort_values()[::-1]
    elif orderby == 'ordinal':
        pass  # ct == ct already
    elif orderby == 'smrystat':
        ct = ct.reindex(smrystat.sort_values()[::-1].index)
    else:
        pass  # accept the default ordering as passed into func

    f = plt.figure(figsize=(16, 2 + (len(ct) * 0.25)))  # , constrained_layout=True)
    gs = gridspec.GridSpec(1, 3, width_ratios=[5, 5, 1], figure=f)
    ax0 = f.add_subplot(gs[0])
    ax1 = f.add_subplot(gs[1], sharey=ax0)
    ax2 = f.add_subplot(gs[2], sharey=ax0)

    if plot_compact:
        ax1.yaxis.label.set_visible(False)
        ax2.yaxis.label.set_visible(False)
        plt.setp(ax1.get_yticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)

    ax0.set_title(f'Distribution of bootstrapped {smry}')
    ax1.set_title('Distribution of indiv. values')
    ax2.set_title('Count')

    if pal is None:
        pal = 'viridis'

    kws = dict(y=grp, order=ct.index.values, data=dfp, palette=pal)
    kws_point = {**kws, **dict(estimator=est, errorbar=('ci', 94))}
    sym = 'k' if plot_outliers else ''
    kws_box = {**kws, **dict(sym=sym, whis=[3, 97], meanprops=sty['mn_pt_kws'])}

    _ = sns.pointplot(**kws_point, x=val, ax=ax0)
    _ = sns.boxplot(**kws_box, x=val, showmeans=True, ax=ax1)
    _ = sns.countplot(**kws, ax=ax2)
    _ = [
        ax2.annotate(
            f'{c} ({c/ct.sum():.0%})', xy=(c, i % len(ct)), **sty['count_txt_h_kws']
        )
        for i, c in enumerate(ct)
    ]

    if plot_grid:
        ax0.yaxis.grid(True)
        ax1.yaxis.grid(True)
        ax2.yaxis.grid(True)

    t = f'Diagnostic 1D plots of `{val}` grouped by `{grp}`'
    txtadd = kwargs.pop('txtadd', None)
    _ = f.suptitle('\n'.join(filter(None, [t, txtadd])), y=1, fontsize=14)

    if sum(idx) > 0:
        t = (
            f'Note: {sum(~idx):,.0f} NaNs found in value,'
            f'\nplotted non-NaN dataset of {sum(idx):,.0f}'
        )
        _ = ax2.annotate(
            t, xy=(0.96, 0.96), xycoords='figure fraction', ha='right', fontsize=8
        )

    f = plt.gcf()
    _ = f.tight_layout()
    return f


def plot_smrystat_grp_year(
    df: pd.DataFrame,
    grp: str = 'grp',
    val: str = 'y_eloss',
    year: str = 'uw_year',
    smry: Literal['sum', 'mean'] = 'sum',
    plot_outliers: bool = True,
    plot_compact: bool = True,
    plot_grid: bool = True,
    yorder_count: bool = True,
    pal: sns.palettes._ColorPalette = None,
    **kwargs,
) -> figure.Figure:
    """Plot diagnostics (smrystat, dist, count) of numeric value `val`
    grouped by categorical value `grp`, grouped by `year`
    """

    sty = _get_kws_styling()
    lvls = df.groupby(grp).size().index.tolist()
    yrs = df.groupby(year).size().index.tolist()

    f = plt.figure(figsize=(16, len(yrs) * 2 + (len(lvls) * 0.25)))
    gs = gridspec.GridSpec(len(yrs), 3, width_ratios=[5, 5, 1], figure=f)
    ax0d, ax1d, ax2d = {}, {}, {}

    for i, yr in enumerate(yrs):  # ugly loop over years
        dfs = df.loc[df[year] == yr].copy()
        grpsort = sorted(dfs[grp].unique())[::-1]

        if not dfs[grp].dtypes in ['object', 'category', 'string']:
            dfs[grp] = dfs[grp].map(lambda x: f's{x}')
            grpsort = [f's{x}' for x in grpsort]

        sz = dfs.groupby(grp).size()
        ct = sz.sort_values()[::-1] if yorder_count else sz.reindex(grpsort)

        if i == 0:
            ax0d[i] = f.add_subplot(gs[i, 0])
            ax1d[i] = f.add_subplot(gs[i, 1], sharey=ax0d[i])
            ax2d[i] = f.add_subplot(gs[i, 2], sharey=ax0d[i])
        else:
            ax0d[i] = f.add_subplot(gs[i, 0], sharey=ax0d[0], sharex=ax0d[0])
            ax1d[i] = f.add_subplot(gs[i, 1], sharey=ax0d[i], sharex=ax1d[0])
            ax2d[i] = f.add_subplot(gs[i, 2], sharey=ax0d[i], sharex=ax2d[0])

        if plot_compact:
            ax1d[i].yaxis.label.set_visible(False)
            ax2d[i].yaxis.label.set_visible(False)
            plt.setp(ax1d[i].get_yticklabels(), visible=False)
            plt.setp(ax2d[i].get_yticklabels(), visible=False)

        ax0d[i].set_title(f'Distribution of bootstrapped {smry} [{yr:"%Y"}]')
        ax1d[i].set_title(f'Distribution of indiv. values [{yr:"%Y"}]')
        ax2d[i].set_title(f'Count [{yr:"%Y"}]')

        if pal is None:
            pal = 'viridis'
        est = np.sum if smry == 'sum' else np.mean
        kws = dict(y=grp, data=dfs, order=ct.index.values, palette=pal)
        kws_point = {**kws, **dict(estimator=est, errorbar=('ci', 94))}
        sym = 'k' if plot_outliers else ''
        kws_box = {**kws, **dict(sym=sym, whis=[3, 97], meanprops=sty['mn_pt_kws'])}

        _ = sns.pointplot(**kws_point, x=val, linestyles='-', ax=ax0d[i])
        _ = sns.boxplot(**kws_box, x=val, showmeans=True, ax=ax1d[i])
        _ = sns.countplot(**kws, ax=ax2d[i])
        _ = [
            ax2d[i].annotate(f'{v}', xy=(v, j % len(ct)), **sty['count_txt_h_kws'])
            for j, v in enumerate(ct)
        ]

        if plot_grid:
            ax0d[i].yaxis.grid(True)
            ax1d[i].yaxis.grid(True)
            ax2d[i].yaxis.grid(True)

    t = f'Diagnostic 1D plots of `{val}` grouped by `{grp}` split by {year}'
    txtadd = kwargs.pop('txtadd', None)
    _ = f.suptitle('\n'.join(filter(None, [t, txtadd])), y=1, fontsize=14)
    _ = plt.tight_layout()
    f = plt.gcf()
    return f


def plot_heatmap_corr(dfx_corr: pd.DataFrame, **kwargs) -> figure.Figure:
    """Convenience plot correlation as heatmap"""
    f, axs = plt.subplots(
        1, 1, figsize=(6 + 0.25 * len(dfx_corr), 4 + 0.25 * len(dfx_corr))
    )
    _ = sns.heatmap(
        dfx_corr,
        mask=np.triu(np.ones_like(dfx_corr), k=0),
        cmap='RdBu_r',
        square=True,
        ax=axs,
        annot=True,
        fmt='.2f',
        linewidths=0.5,
        vmin=-1,
        vmax=1,
        center=0,
    )
    _ = axs.set_xticklabels(axs.get_xticklabels(), rotation=40, ha='right')
    txtadd = kwargs.pop('txtadd', None)
    t = 'Feature correlations'
    _ = f.suptitle(' - '.join(filter(None, [t, txtadd])), y=1, fontsize=14)
    _ = f.tight_layout()
    return f


def plot_kj_summaries_for_linear_model(dfp, policy_id, title_add='psi'):
    """Convenience: plot summary of kj components for a linear model coeffs
    Highly coupled to summarise_kj_components_for_linear_model
    """

    idx = ~dfp['ft_mapped'].isnull()
    gd = sns.FacetGrid(
        hue='component', data=dfp.loc[idx], palette='vlag', height=4, aspect=1.5
    )
    _ = gd.map(
        sns.barplot,
        'component',
        'ft_mapped',
        order=dfp.loc[idx, 'ft_mapped'],
        lw=3,
        zorder=1,
    )
    _ = gd.axes.flat[0].axvline(0, color='#dddddd', lw=3, zorder=2)
    _ = gd.axes.flat[0].set(xlabel=None, ylabel=None)  # , xticklabels=[])
    _ = gd.fig.suptitle(
        f'Components of linear submodel predictions: {title_add}\nfor policy {policy_id}',
        y=1.08,
    )

    rhs_lbls = dfp.loc[idx, 'input_val_as_label'].values[::-1]

    axr = gd.axes.flat[0].twinx()
    _ = axr.plot(np.zeros(len(rhs_lbls)), np.arange(len(rhs_lbls)) + 0.5, lw=0)
    # _ = axr.set_ylim((-1,len(rhs_lbls)))
    _ = axr.set_yticks([x for x in np.arange(len(rhs_lbls)) + 0.5])
    _ = axr.set_yticklabels(rhs_lbls)
    _ = axr.yaxis.grid(False)
    _ = axr.xaxis.grid(False)
    # _ = axr.spines['top'].set_visible(False)
    # _ = axr.spines['right'].set_visible(False)
    return gd


def plot_grp_ct(
    df: pd.DataFrame, grp: str = 'grp', title_add: str = ''
) -> figure.Figure:
    """Simple countplot for factors in grp, label with percentages
    Works nicely with categorical too
    """

    sty = _get_kws_styling()
    if not df[grp].dtypes in ['object', 'category']:
        raise TypeError('grp must be Object (string) or Categorical')

    ct = df[grp].value_counts(dropna=False)

    f, axs = plt.subplots(1, 1, figsize=(14, 2 + (len(ct) * 0.25)))
    _ = sns.countplot(y=grp, data=df, order=ct.index, ax=axs, palette='viridis')
    _ = [
        axs.annotate(
            f'{v:.0f} ({v/len(df):.0%})', xy=(v, i % len(ct)), **sty['count_txt_h_kws']
        )
        for i, v in enumerate(ct)
    ]

    _ = axs.set(ylabel=None)

    if title_add != '':
        title_add = f'\n{title_add}'

    title = f'Countplot: {len(df)} obs, grouped by {grp}'
    _ = f.suptitle(f'{title}{title_add}')

    _ = plt.tight_layout()

    return f


def plot_cdf_ppc_vs_obs(
    y: np.ndarray, yhat: np.ndarray, xlim_max_override=None
) -> figure.Figure:
    """Plot (quantile summaries of) yhat_ppc vs y
    NOTE:
    y shape: (nobs,)
    yhat shape: (nobs, nsamples)
    """
    ps = [3, 10, 20, 30, 40, 50, 60, 70, 80, 90, 97]
    df_yhat_qs = pd.DataFrame(
        np.percentile(yhat, ps, axis=1).T, columns=[f'q{p/100}' for p in ps]
    )

    f, axs = plt.subplots(1, 1, figsize=(14, 5), sharey=True, sharex=True)
    _ = sns.kdeplot(
        y, cumulative=True, lw=2, c='g', ax=axs, common_norm=False, common_grid=True
    )

    if df_yhat_qs.duplicated().sum() == len(df_yhat_qs) - 1:
        # all dupes: model was intercept only
        dfm = df_yhat_qs.iloc[:1].melt(var_name='ppc_q')
        _ = sns.rugplot(
            x='value',
            hue='ppc_q',
            data=dfm,
            palette='coolwarm',
            lw=2,
            ls='-',
            height=1,
            ax=axs,
            zorder=-1,
        )
    else:
        dfm = df_yhat_qs.melt(var_name='ppc_q')
        _ = sns.kdeplot(
            x='value',
            hue='ppc_q',
            data=dfm,
            cumulative=True,
            palette='coolwarm',
            lw=2,
            ls='-',
            ax=axs,
            zorder=-1,
            common_norm=False,
            common_grid=True,
        )

    if xlim_max_override is not None:
        _ = axs.set(xlim=(0, xlim_max_override), ylim=(0, 1))
    else:
        _ = axs.set(xlim=(0, np.ceil(y.max())), ylim=(0, 1))

    _ = f.suptitle('Cumulative density plot of the posterior predictive vs actual')

    return f


def plot_explained_variance(
    df: pd.DataFrame, k: int = 10, topn: int = 3
) -> figure.Figure:
    """Calculate Truncated SVD and plot explained variance curve, with optional
    vline for the topn components Related to eda.calc.get_svd"""

    _, svd_fit = calc_svd(df, k)
    evr = pd.Series(
        svd_fit.explained_variance_ratio_.cumsum(), name='explained_variance_csum'
    )
    evr.index = np.arange(1, len(evr) + 1)
    evr.index.name = 'component'

    f, axs = plt.subplots(1, 1, figsize=(12, 5))
    _ = sns.pointplot(
        x='component', y='explained_variance_csum', data=evr.reset_index(), ax=axs
    )
    _ = axs.vlines(topn - 1, 0, 1, 'orange', '-.')
    _ = axs.annotate(
        '{:.1%}'.format(evr[topn]),
        xy=(topn - 1, evr[topn]),
        xycoords='data',
        xytext=(-10, 10),
        textcoords='offset points',
        color='orange',
        ha='right',
        fontsize=12,
    )

    _ = axs.set_ylim(0, 1.001)
    _ = f.suptitle(
        f'Explained variance @ top {topn} components ~ {evr[topn]:.1%}', fontsize=14
    )
    _ = f.tight_layout()
    return f
