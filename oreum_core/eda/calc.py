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

# eda.calc.py
"""Calculations to help EDA"""
import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import figure
from scipy import stats
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from umap.umap_ import UMAP

RSD = 42
rng = np.random.default_rng(seed=RSD)

_log = logging.getLogger(__name__)

__all__ = [
    'fit_and_plot_fn',
    'get_gini',
    'bootstrap',
    'bootstrap_lr',
    'calc_geometric_cv',
    'calc_location_in_ecdf',
    'month_diff',
    'tril_nan',
    'calc_svd',
    'calc_umap',
]


# TODO see issue #2
def fit_and_plot_fn(obs: pd.Series) -> tuple[figure.Figure, dict]:
    """Fit dists to 1d array of `obs`, report RMSE and plot the fits
    see https://stackoverflow.com/a/37616966
    """

    dists_discrete_support_gte_zero = {
        'poisson': stats.poisson,
        # 'binom': stats.binom,  # placeholder
        # 'nbinom': stats.nbinom,  # placeholder
    }

    dists_cont_support_real = {
        'norm': stats.norm,
        'cauchy': stats.cauchy,
        'gumbel': stats.gumbel_r,
        'invweibull': stats.invweibull,
    }

    dists_cont_support_gte_zero = {
        'expon': stats.expon,
        'halfnorm': stats.halfnorm,
        'halfcauchy': stats.halfcauchy,
        'fisk': stats.fisk,
    }

    dists_cont_support_gt_zero = {
        'gamma': stats.gamma,
        'invgamma': stats.invgamma,
        'lognorm': stats.lognorm,
    }

    nbins = 50
    dist_kind = 'Continuous'
    params = {}
    f, ax1d = plt.subplots(1, 1, figsize=(14, 6))
    hist_kws = dict(
        kde=False, label='data', ax=ax1d, alpha=0.5, color='#aaaaaa', zorder=-1
    )
    line_kws = dict(lw=2, ls='--', ax=ax1d)

    # handle nans and infs
    n_nans = pd.isnull(obs).sum()
    n_infs = np.isinf(obs).sum()
    idx = pd.isnull(obs) | np.isinf(obs)
    obs = obs.loc[~idx]

    obs_is_discrete = sum(obs == (obs // 1)) == len(obs)
    n_zeros = (obs == 0).sum()
    n_negs = (obs < 0).sum()

    def _annotate_facets(n_nans, n_infs, n_zeros):
        """Convenience to annotate, based on eda.plots.plot_float_dist"""
        n_zeros = (obs == 0).sum()
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
        dists = dists_discrete_support_gte_zero

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

        dists = dists_cont_support_real
        if n_negs == 0:
            dists.update(dists_cont_support_gte_zero)
        if n_zeros == 0:
            dists.update(dists_cont_support_gt_zero)

        _ = sns.histplot(x=obs, bins=nbins, stat='density', **hist_kws)

        for i, (d, dist) in enumerate(dists.items()):
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore')
                ps = dist.fit(obs, floc=0)  # can throw RuntimeWarnings which we ignore
            shape, loc, scale = ps[:-2], ps[-2], ps[-1]
            params[d] = dict(shape=shape, loc=loc, scale=scale)
            pdf = dist.pdf(bin_centers, loc=loc, scale=scale, *shape)
            rmse = np.sqrt(np.sum(np.power(obs_density - pdf, 2.0)) / len(obs))
            _ = sns.lineplot(
                x=bin_centers, y=pdf, label=f'{d}: {rmse:#.2g}', **line_kws
            )

    title = f'{dist_kind} function approximations to {obs.name}'
    _ = f.suptitle(title, y=0.97)
    _ = f.axes[0].legend(title='dist: RMSE', title_fontsize=10)
    _annotate_facets(n_nans, n_infs, n_zeros)

    return f, params


def get_gini(r: np.ndarray, n: np.ndarray) -> np.ndarray:
    """For array r, return estimate of gini co-efficient over n
    g = A / (A+B)
    """
    return 1 - sum(r.sort_values().cumsum() * (2 / n))


def bootstrap(
    a: np.ndarray | pd.Series, nboot: int = 1000, summary_fn=np.mean, idx_only=False
) -> np.ndarray:
    """Calc vectorised bootstrap sample of ndarray of observations
    By default return the mean value of the observations per sample
    i.e. if len(a)=20 and nboot=100, this returns 100 bootstrap resampled
    mean estimates of those 20 observations
    Vectorised sampling via numpy broadcasting random indexes to a 2D shape
    """
    rng = np.random.default_rng(seed=RSD)
    sample_idx = rng.integers(0, len(a), size=(len(a), nboot))

    if idx_only:
        return sample_idx

    # hack allow for passing a series, need a ndarray to broadcast 2D properly
    if isinstance(a, pd.Series):
        a = a.values

    samples = a[sample_idx]
    if summary_fn is not None:
        return np.apply_along_axis(summary_fn, 0, samples)
    else:
        return samples


def bootstrap_lr(
    df: pd.DataFrame, prm: str = 'premium', clm: str = 'claim', nboot: int = 1000
) -> pd.DataFrame:
    """Calc vectorised bootstrap loss ratios for df
    Pass dataframe or group, accept nans in clm
    Use the same index for prem and claims
    """
    idx = bootstrap(df[prm], nboot, idx_only=True)
    s_prm = df[prm].values[idx]
    s_clm = df[clm].values[idx]

    dfboot = pd.DataFrame(
        {
            'premium_sum': np.apply_along_axis(np.sum, 0, s_prm),
            'claim_sum': np.apply_along_axis(np.sum, 0, np.nan_to_num(s_clm, 0)),
        }
    )

    dfboot['lr'] = dfboot['claim_sum'] / dfboot['premium_sum']
    return dfboot


def calc_geometric_cv(lognormal_yhat: np.ndarray) -> np.ndarray:
    """Calculate geometric coefficient of variation for log-normally
    distributed samples.
    Expect 2D array shape (nobs, nsamples)
    https://en.wikipedia.org/wiki/Coefficient_of_variation#Log-normal_data
    """
    return np.sqrt(np.exp(np.std(np.log(lognormal_yhat), axis=1) ** 2) - 1)


def _ecdf(a: np.ndarray) -> tuple:
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


def month_diff(
    a: pd.Series, b: pd.Series, series_name: str = 'month_diff'
) -> pd.Series:
    """Integer month count between dates a to b

    https://stackoverflow.com/a/40924041/1165112

    In recent pandas can equally use to_period(), though it's unwieldy
    e.g
    [x.n for x in (df['obs_date'].dt.to_period('M') -
                    df['reported_date'].dt.to_period('M'))]
    """
    s = 12 * (b.dt.year - a.dt.year) + (b.dt.month - a.dt.month)
    s.name = series_name
    return s


def tril_nan(m: np.ndarray, k: int = 0) -> np.ndarray:
    """Copy of np.tril but mask with np.nans not zeros
    Example usage, to set tril to np.nan in DataFrame.corr()
    corr = df_b0.corr()
    mask = eda.tril_nan(corr, k=-1)
    corr.mask(np.isnan(mask))

    """

    m = np.asanyarray(m)  # numpy.core.numeric
    mask = np.tri(*m.shape[-2:], k=k, dtype=bool)

    # return np.where(mask, m, np.ones(1, m.dtype) * np.nan)
    return np.where(mask, m, np.nan)


def calc_svd(df: pd.DataFrame, k: int = 10) -> tuple[pd.DataFrame, TruncatedSVD]:
    """Calc SVD for k components (and preprocess to remove nulls and zscore),
    report degeneracy, return transformed df and fitted TruncatedSVD object"""

    # protect SVD from nulls
    idx_nulls = df.isnull().sum(axis=1) > 0
    if sum(idx_nulls) > 0:
        df = df.loc[~idx_nulls].copy()
        _log.info(f'Excluding {sum(idx_nulls)} rows containing a null, prior to SVD')

    # standardize
    scaler = StandardScaler().fit(df)
    dfs = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)

    # use scikit learn's TruncatedSVD with randomized
    k = min(k, dfs.shape[1] - 1)
    svd = TruncatedSVD(n_components=k, random_state=RSD)
    svd_fit = svd.fit(dfs)

    # Are any eigenvalues NaN or really small?
    n_null = sum(np.isnan(svd_fit.singular_values_))
    assert n_null == 0, f'{n_null} Singular Values are NaN'
    n_tiny = sum(svd_fit.singular_values_ < 1e-12)
    assert n_tiny == 0, f'{n_tiny} Singular Values are < 1e-12'

    dfx = svd_fit.transform(dfs)

    return dfx, svd_fit


def calc_umap(df: pd.DataFrame) -> tuple[pd.DataFrame, UMAP]:
    """Calc 2D UMAP (and preprocess to remove nulls and zscore), return
    transformed df and fitted UMAP object"""

    # protect UMAP from nulls
    idx_nulls = df.isnull().sum(axis=1) > 0
    if sum(idx_nulls) > 0:
        df = df.loc[~idx_nulls].copy()
        _log.info(f'Excluding {sum(idx_nulls)} rows containing a null, prior to UMAP')

    umapper = UMAP(n_neighbors=5)
    umap_fit = umapper.fit(df)
    dfx = pd.DataFrame(umap_fit.transform(df), columns=['c0', 'c1'], index=df.index)

    return dfx, umap_fit
