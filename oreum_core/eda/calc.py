# eda.calc.py
# copyright 2021 Oreum OÜ
import re
import string
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.base import DataError
import seaborn as sns
from scipy import stats

RANDOM_SEED = 42
rng = np.random.default_rng(seed=RANDOM_SEED)

# TODO see issue #2
def fit_and_plot_fn(obs, tail_kind='right', title_insert=None):
    """
    Fit dists to 1d array of `obs`, report RMSE and plot the fits
    # see https://stackoverflow.com/a/37616966 
    """

    if tail_kind not in set(['right', 'both']):
        raise ValueError("tail_kind must be in {'right', 'both'}")

    import warnings
    # warnings.filterwarnings("error") # handle RuntimeWarning as error so can catch
    # warnings.simplefilter(action='ignore', category='RuntimeWarning')

    dists_discrete = {'poisson': stats.poisson}

    dists_cont_right_tail = {'expon': stats.expon,
                            'gamma':stats.gamma, 
                            'invgamma':stats.invgamma, 
                            # 'invgauss': stats.invgauss,
                            'halfnorm': stats.halfnorm,
                            'halfcauchy': stats.halfcauchy,
                            'lognorm': stats.lognorm,
                            'gumbel': stats.gumbel_r,
                            'invweibull': stats.invweibull}
    # NOTE: not quite true since gumbel and invweibull can go neg

    dists_cont_centered = {'norm': stats.norm,
                           'cauchy': stats.cauchy}
   
    obs_is_discrete = sum(obs == (obs // 1)) == len(obs)
    nbins = 50
    dist_kind = 'Continuous'
    params = {}
    f, ax1d = plt.subplots(1, 1, figsize=(15, 6))
    hist_kws = dict(kde=False, label='data', ax=ax1d, alpha=0.5,
                    color='#aaaaaa', zorder=-1)
    line_kws = dict(lw=2, ls='--', ax=ax1d)
    
    if obs_is_discrete:
        dist_kind = 'Discrete'
        nbins = np.int(obs.max())
        obs_count, bin_edges = np.histogram(obs, bins=nbins, density=False)
        bin_centers = (bin_edges + np.roll(bin_edges, -1))[:-1] / 2.0
        bin_centers_int = np.round(bin_centers)
        dists = dists_discrete

        ax0 = sns.histplot(x=obs, bins=nbins, stat='count', **hist_kws)

        # TODO fix this hack: only works for poisson right now
        for i, (d, dist) in enumerate(dists.items()):
            mle_pois = obs.mean()
            params[d] = dict(shape=mle_pois, loc=0, scale=None)
            pmf = stats.poisson.pmf(k=bin_centers_int, mu=mle_pois) * len(obs)

            # rmse not necessarily good for discrete count models
            # https://stats.stackexchange.com/questions/48811/cost-function-for-validating-poisson-regression-models
            rmse = np.sqrt(np.sum(np.power(obs_count - pmf, 2.0)) / len(obs))
            ax1 = sns.lineplot(x=bin_centers_int, y=pmf, 
                               label=f'{d}: {rmse:.2g}', **line_kws)

    else:
        obs_density, bin_edges = np.histogram(obs, bins=nbins, density=True)
        bin_centers = (bin_edges + np.roll(bin_edges, -1))[:-1] / 2.0
        
        if tail_kind == 'both':
            dists = dists_cont_centered
        elif tail_kind == 'right':
            dists = dists_cont_right_tail
     
        ax0 = sns.histplot(x=obs, bins=nbins, stat='density', **hist_kws)

        for i, (d, dist) in enumerate(dists.items()):
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore')
                ps = dist.fit(obs, floc=0)   # can throw RuntimeWarnings which we ignore
            shape, loc, scale = ps[:-2], ps[-2], ps[-1]
            params[d] = dict(shape=shape, loc=loc, scale=scale)
            pdf = dist.pdf(bin_centers, loc=loc, scale=scale, *shape)

            # rmse not necessarily good for discrete count models
            # https://stats.stackexchange.com/questions/48811/cost-function-for-validating-poisson-regression-models
            rmse = np.sqrt(np.sum(np.power(obs_density - pdf, 2.0)) / len(obs))
            ax1 = sns.lineplot(x=bin_centers, y=pdf, 
                               label=f'{d}: {rmse:.2g}', **line_kws)

    title = (f'{dist_kind} function approximations to `{title_insert}`')
    _ = f.suptitle(title, y=0.97)
    _ = f.axes[0].legend(title='dist: RMSE', title_fontsize=10)
    return f, params


def get_gini(r, n):
    """ For array r, return estimate of gini co-efficient over n
        g = A / (A+B)
    """
    return 1 - sum(r.sort_values().cumsum() * (2 / n))


def bootstrap_lr(df, prm='premium', clm='claim', nboot=1000):
    """ Calc vectorised bootstrap loss ratios for df
        Pass a dataframe or group. fts named `'premium', 'claim'`
        Accept nans in clm
    """
    # vectorise via numpy broadcasting random indexs to a larger shape
    rng = np.random.default_rng(42)
    sample_idx = rng.integers(0, len(df), size=(len(df), nboot))
    premium_amt_boot = df[prm].values[sample_idx]
    claim_amt_boot = np.nan_to_num(df[clm], 0)[sample_idx]
    
    dfboot = pd.DataFrame({
                'premium_sum': premium_amt_boot.sum(axis=0),
                'claim_sum': claim_amt_boot.sum(axis=0)})

    dfboot['lr'] = dfboot['claim_sum'] / dfboot['premium_sum']
    
    return dfboot   