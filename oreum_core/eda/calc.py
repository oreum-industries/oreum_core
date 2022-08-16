# eda.calc.py
# copyright 2022 Oreum Industries

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import figure
from scipy import stats

RSD = 42
rng = np.random.default_rng(seed=RSD)


# TODO see issue #2
def fit_and_plot_fn(
    obs: pd.Series, tail_kind: str = 'right'
) -> tuple[figure.Figure, dict]:
    """Fit dists to 1d array of `obs`, report RMSE and plot the fits
    see https://stackoverflow.com/a/37616966
    """

    if tail_kind not in set(['right', 'both']):
        raise ValueError("tail_kind must be one of {'right', 'both'}")

    # warnings.filterwarnings("error") # handle RuntimeWarning as error so can catch
    # warnings.simplefilter(action='ignore', category='RuntimeWarning')

    dists_discrete = {'poisson': stats.poisson}

    dists_cont_right_tail = {
        'expon': stats.expon,
        'gamma': stats.gamma,
        'gumbel': stats.gumbel_r,
        'halfnorm': stats.halfnorm,
        'halfcauchy': stats.halfcauchy,
        'invgamma': stats.invgamma,
        # 'invgauss': stats.invgauss,
        'invweibull': stats.invweibull,
        'lognorm': stats.lognorm,
        'fisk': stats.fisk
    }
    # NOTE: not quite true since gumbel and invweibull can go neg

    dists_cont_centered = {'norm': stats.norm, 'cauchy': stats.cauchy}

    obs_is_discrete = sum(obs == (obs // 1)) == len(obs)
    nbins = 50
    dist_kind = 'Continuous'
    params = {}
    f, ax1d = plt.subplots(1, 1, figsize=(14, 6))
    hist_kws = dict(
        kde=False, label='data', ax=ax1d, alpha=0.5, color='#aaaaaa', zorder=-1
    )
    line_kws = dict(lw=2, ls='--', ax=ax1d)

    def _annotate_facets():
        """Convenience to annotate, based on eda.plots.plot_float_dist"""
        n_nans = pd.isnull(obs).sum()
        n_zeros = (obs == 0).sum()
        n_infs = np.isinf(obs).sum()
        mean = obs.mean()
        med = obs.median()
        ax = plt.gca()
        ax.text(
            0.5,
            0.97,
            (
                f'NaNs: {n_nans},  infs+/-: {n_infs},  zeros: {n_zeros},  '
                + f'mean: {mean:.2f},  med: {med:.2f}'
            ),
            transform=ax.transAxes,
            ha='center',
            va='top',
            backgroundcolor='w',
            fontsize=10,
        )


    if obs_is_discrete:
        dist_kind = 'Discrete'
        nbins = np.int(obs.max())
        obs_count, bin_edges = np.histogram(obs, bins=nbins, density=False)
        bin_centers = (bin_edges + np.roll(bin_edges, -1))[:-1] / 2.0
        bin_centers_int = np.round(bin_centers)
        dists = dists_discrete

        _ = sns.histplot(x=obs, bins=nbins, stat='count', **hist_kws)

        # TODO fix this hack: only works for poisson right now
        for i, (d, dist) in enumerate(dists.items()):
            mle_pois = obs.mean()
            params[d] = dict(shape=mle_pois, loc=0, scale=None)
            pmf = stats.poisson.pmf(k=bin_centers_int, mu=mle_pois) * len(obs)

            # rmse not necessarily good for discrete count models
            # https://stats.stackexchange.com/questions/48811/cost-function-for-validating-poisson-regression-models
            rmse = np.sqrt(np.sum(np.power(obs_count - pmf, 2.0)) / len(obs))
            _ = sns.lineplot(
                x=bin_centers_int, y=pmf, label=f'{d}: {rmse:#.2g}', **line_kws
            )

    else:
        obs_density, bin_edges = np.histogram(obs, bins=nbins, density=True)
        bin_centers = (bin_edges + np.roll(bin_edges, -1))[:-1] / 2.0

        if tail_kind == 'both':
            dists = dists_cont_centered
        elif tail_kind == 'right':
            dists = dists_cont_right_tail

        _ = sns.histplot(x=obs, bins=nbins, stat='density', **hist_kws)

        for i, (d, dist) in enumerate(dists.items()):
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore')
                ps = dist.fit(obs, floc=0)  # can throw RuntimeWarnings which we ignore
            shape, loc, scale = ps[:-2], ps[-2], ps[-1]
            params[d] = dict(shape=shape, loc=loc, scale=scale)
            pdf = dist.pdf(bin_centers, loc=loc, scale=scale, *shape)

            # rmse not necessarily good for discrete count models
            # https://stats.stackexchange.com/questions/48811/cost-function-for-validating-poisson-regression-models
            rmse = np.sqrt(np.sum(np.power(obs_density - pdf, 2.0)) / len(obs))
            _ = sns.lineplot(
                x=bin_centers, y=pdf, label=f'{d}: {rmse:#.2g}', **line_kws
            )

    title = f'{dist_kind} function approximations to {obs.name}'
    _ = f.suptitle(title, y=0.97)
    _ = f.axes[0].legend(title='dist: RMSE', title_fontsize=10)
    _annotate_facets()

    return f, params


def get_gini(r, n):
    """For array r, return estimate of gini co-efficient over n
    g = A / (A+B)
    """
    return 1 - sum(r.sort_values().cumsum() * (2 / n))


def bootstrap(a, nboot=1000, summary_fn=np.mean):
    """Calc vectorised bootstrap sample of array of observations
    By default return the mean value of the observations per sample
    I.e if len(a)=20 and nboot=100, this returns 100 bootstrap resampled
    mean estimates of those 20 observations
    """
    # vectorise via numpy broadcasting random indexs to a 2D shape
    rng = np.random.default_rng(seed=RSD)
    sample_idx = rng.integers(0, len(a), size=(len(a), nboot))

    # hack allow for passing a series
    if type(a) == pd.Series:
        a = a.values

    samples = a[sample_idx]
    if summary_fn is not None:
        return np.apply_along_axis(summary_fn, 0, samples)
    else:
        return samples


def bootstrap_lr(df, prm='premium', clm='claim', nboot=1000):
    """Calc vectorised bootstrap loss ratios for df
    Pass a dataframe or group. fts named `'premium', 'claim'`
    Accept nans in clm
    """
    dfboot = pd.DataFrame(
        {
            'premium_sum': bootstrap(df[prm], nboot, np.sum),
            'claim_sum': bootstrap(np.nan_to_num(df[clm], 0), nboot, np.sum),
        }
    )

    dfboot['lr'] = dfboot['claim_sum'] / dfboot['premium_sum']
    return dfboot


def calc_geometric_cv(lognormal_yhat):
    """Calculate geometric coefficient of variation for log-normally
    distributed samples.
    Expect 2D array shape (nobs, nsamples)
    https://en.wikipedia.org/wiki/Coefficient_of_variation#Log-normal_data
    """
    return np.sqrt(np.exp(np.std(np.log(lognormal_yhat), axis=1) ** 2) - 1)


def _ecdf(a):
    """Empirical CDF of array
    Return sorted array and ecdf values
    """
    x = np.sort(a)
    n = len(a)
    y = np.arange(1, n + 1) / n
    return x, y


def calc_location_in_ecdf(baseline_arr, test_arr):
    """Calculate the position of each element in test_arr
    relative to the ECDF described by baseline_arr
    (len(baseline_arr) === len(test_arr)) = False
    """
    sorted_baseline, cdf_prop = _ecdf(baseline_arr)
    idxs = np.argmax((test_arr < sorted_baseline[:, None]), axis=0)
    return cdf_prop[idxs]


def month_diff(a: pd.DataFrame, b: pd.DataFrame):
    """https://stackoverflow.com/a/40924041/1165112

    In recent pandas can equally use to_period(), though it's unwieldy
    e.g
    [x.n for x in (df['obs_date'].dt.to_period('M') -
                    df['reported_date'].dt.to_period('M'))]
    """
    return 12 * (a.dt.year - b.dt.year) + (a.dt.month - b.dt.month)
