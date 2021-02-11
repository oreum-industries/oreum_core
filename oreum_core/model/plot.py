# model.plot.py
# copyright 2021 Oreum OÜ
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy as pt
import seaborn as sns


def facetplot_azid_dist(azid, rvs, rvs_hack_extra=0, group='posterior', ref_vals=None):
    """Convenience: plot Krushke style in facets """
    # TODO unpack the compressed rvs from the azid
    
    # m, n = 2, (len(rvs) // 2) + (len(rvs) % 2)
    m, n = 2, ((len(rvs)+rvs_hack_extra) // 2) + ((len(rvs)+rvs_hack_extra) % 2)
    f, ax1d = plt.subplots(n, m, figsize=(m*6, 2.2*n))
    kw = {}
    if ref_vals is not None:
        kw['ref_vals'] = ref_vals
    _ = az.plot_posterior(azid, group=group, ax=ax1d, var_names=rvs, **kw)
    f.suptitle(group, y=0.9 + n*0.005)
    f.tight_layout()


def plot_dist_fns_over_x(dfpdf, dfcdf, dfinvcdf, **kwargs):
    """Convenience to plot results of calc_dist_fns_over_x() """

    name = kwargs.get('name', 'unknown_dist')
    islog = kwargs.get('log', False)
    l = 'log ' if islog else ''
    f, axs = plt.subplots(1, 3, figsize=(16, 4), sharex=False, sharey=False)
    f.suptitle(f'Comparisons manual vs scipy for {l}{name}', y=1.02)
    n = len(dfpdf)

    dfm = dfpdf.reset_index().melt(id_vars='x', value_name='density', var_name='method')
    ax0 = sns.lineplot(x='x', y='density', hue='method', style='method',data=dfm, ax=axs[0])
    _ = ax0.set_title(f"{l}PDF: match {np.sum(np.isclose(dfpdf['manual'], dfpdf['scipy'])) / n :.1%}")

    dfm = dfcdf.reset_index().melt(id_vars='x', value_name='density', var_name='method')
    ax1 = sns.lineplot(x='x', y='density', hue='method', style='method', data=dfm, ax=axs[1])
    _ = ax1.set_title(f"{l}CDF: match {np.sum(np.isclose(dfcdf['manual'], dfcdf['scipy'])) / n :.1%}")

    dfm = dfinvcdf.reset_index().melt(id_vars='u', value_name='density', var_name='method')
    ax2 = sns.lineplot(x='u', y='density', hue='method', style='method', data=dfm, ax=axs[2])
    _ = ax2.set_title(f"{l}InvCDF: match {np.sum(np.isclose(dfinvcdf['manual'], dfinvcdf['scipy'])) / n:.1%}")