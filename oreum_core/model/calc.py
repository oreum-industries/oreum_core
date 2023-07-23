# Copyright 2023 Oreum Industries
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

# model.calc.py
"""Common Calculations for Model Evaluation"""
import sys

import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
from pytensor.gradient import grad

__all__ = [
    'calc_f_measure',
    'calc_binary_performance_measures',
    'calc_mse',
    'calc_rmse',
    'calc_r2',
    'calc_bayesian_r2',
    'calc_ppc_coverage',
    'expand_packed_triangular',
    'log_jcd',
    'calc_2_sample_delta_prop',
    'numpy_invlogit',
]


RSD = 42
rng = np.random.default_rng(seed=RSD)


def calc_f_measure(precision, recall, b=1):
    """Choose b such that recall is b times more important than precision"""
    return (1 + b**2) * (precision * recall) / ((b**2 * precision) + recall)


def calc_binary_performance_measures(y, yhat):
    """Calculate tpr (recall), fpr, precision, accuracy for binary target,
    using all samples from PPC, use vectorised calcs
    shapes y: (nsamples,), yhat: (nsamples, nobservations)
    """
    yhat_pct = np.percentile(yhat, np.arange(0, 101, 1), axis=0).T
    y_mx = np.tile(y.reshape(-1, 1), 101)

    # calc tp, fp, tn, fn vectorized
    tp = np.nansum(np.where(yhat_pct == 1, y_mx, np.nan), axis=0)
    fp = np.nansum(np.where(yhat_pct == 1, 1 - y_mx, np.nan), axis=0)
    tn = np.nansum(np.where(yhat_pct == 0, 1 - y_mx, np.nan), axis=0)
    fn = np.nansum(np.where(yhat_pct == 0, y_mx, np.nan), axis=0)

    # calc tpr (recall), fpr, precision etc
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    tpr = recall = tp / (tp + fn)
    fpr = fp / (tn + fp)
    precision = np.nan_to_num(tp / (tp + fp), nan=1)  # beware of divide by zero

    perf = pd.DataFrame(
        {
            'accuracy': accuracy,
            'tpr': tpr,
            'fpr': fpr,
            'recall': recall,
            'precision': precision,
            'f0.5': calc_f_measure(precision, recall, b=0.5),
            'f1': calc_f_measure(precision, recall, b=1),
            'f2': calc_f_measure(precision, recall, b=2),
        },
        index=np.arange(101),
    )
    perf.index.set_names('pct', inplace=True)

    return perf


def calc_mse(y: np.ndarray, yhat: np.ndarray) -> tuple[np.ndarray, pd.Series]:
    r""" Convenience: Calculate MSE using all samples
        y shape: (nobs, )
        yhat shape: (nsamples, nobs)

    Mean-Squared Error of prediction vs observed
    $$\frac{1}{n}\sum_{i=1}^{i=n}(\hat{y}_{i}-y_{i})^{2}$$

    \begin{align}
    \text{Method A, compress to mean of samples: } \text{MSE} &= \frac{1}{n} \sum_{i=1}^{i=n}(\mu_{j}(\hat{y}_{i}) - y_{i})^{2} \\
    \text{Method B, use all samples then mean: } \text{MSE} &= \frac{1}{n} \sum_{i=1}^{i=n}(\mu_{j}(\hat{y}_{ij} - y_{i})^{2})
    \end{align}

    WARNING:
        the 'samples' approach squares outliers pushing farther from the mean
        # se_samples = np.power(yhat - yobs, 2)      # (nsamp, nobs)
        # mse_samples = np.mean(se_samples, axis=1)  # (nsamp,)
        i.e.
        take mean across samples then square the differences is usually smaller than
        square the differences each sample and preserve samples
        https://en.wikipedia.org/wiki/Generalized_mean

        I can only think to calc summary stats and then calc MSE for them
    """
    # collapse samples to mean then calc error
    se = np.power(yhat.mean(axis=0) - y, 2)  # (nobs, )
    mse = np.mean(se, axis=0)  # 1

    # collapse samples to a range of summary stats then calc error
    smry = np.arange(0, 101, 2)
    se_pct = np.power(np.percentile(yhat, smry, axis=0) - y, 2)  # (len(smry), nobs)
    mse_pct = np.mean(se_pct, axis=1)  # len(smry)

    s_mse_pct = pd.Series(mse_pct, index=smry, name='mse')
    s_mse_pct.index.rename('pct', inplace=True)
    return mse, s_mse_pct


def calc_rmse(y: np.ndarray, yhat: np.ndarray) -> tuple[np.ndarray, pd.Series]:
    """Convenience: Calculate RMSE using all samples
    shape (nsamples, nobs)
    """
    mse, s_mse_pct = calc_mse(y, yhat)
    s_rmse_pct = s_mse_pct.map(np.sqrt)
    s_rmse_pct._set_name('rmse', inplace=True)
    return np.sqrt(mse), s_rmse_pct


def calc_r2(y: np.ndarray, yhat: np.ndarray) -> tuple[np.ndarray, pd.Series]:
    """Calculate R2,
    return mean r2 and via summary stats of yhat
    NOTE: shape (nsamples, nobservations)
    $$R^{2} = 1 - \frac{\sum e_{model}^{2}}{\sum e_{mean}^{2}}$$
    R2 normal range [0, 1]
    """
    sse_mean = np.sum((y - y.mean(axis=0)) ** 2)

    # Collapse samples to mean then calc error
    sse_model_mean = np.sum((y - yhat.mean(axis=0)) ** 2)
    r2_mean = 1 - (sse_model_mean / sse_mean)

    # calc summary stats of yhat
    smry = np.arange(0, 101, 5)
    sse_model = np.sum(
        (y - np.percentile(yhat, smry, axis=0)) ** 2, axis=1
    )  # (len(smry), nobs)
    r2_pct = pd.Series(1 - (sse_model / sse_mean), index=smry, name='r2')
    r2_pct.index.rename('pct', inplace=True)

    return r2_mean, r2_pct


def calc_bayesian_r2(y: np.ndarray, yhat: np.ndarray) -> pd.DataFrame:
    """Calculate R2 across all samples
    NOTE:
        y shape: (nobs,)
        yhat shape: (nobs, nsamples)
        return shape: (nsamples, )
    """

    var_yhat = np.var(yhat, axis=0)
    var_residuals = np.var(y.reshape(-1, 1) - yhat, axis=0)
    r2 = var_yhat / (var_yhat + var_residuals)
    return pd.DataFrame(r2, columns=['r2'])


def calc_ppc_coverage(y: np.ndarray, yhat: np.ndarray) -> pd.DataFrame:
    """Calc the proportion of coverage from full yhat ppc
    shapes: y (nobservations), yhat (nsamples, nobservations)
    """

    crs = np.arange(0, 1.01, 0.02)
    bounds = dict(
        pin_left=dict(
            lower=np.tile(np.percentile(yhat, 0.0, axis=0), reps=(len(crs), 1)),
            upper=np.percentile(yhat, 100.0 * crs, axis=0),
        ),
        middle_out=dict(
            lower=np.percentile(yhat, 50.0 - (50.0 * crs), axis=0),
            upper=np.percentile(yhat, 50.0 + (50.0 * crs), axis=0),
        ),
        # pin_right=dict(       ##just a rotation of pin_left
        #     lower=np.percentile(yhat, 100. - (100 * crs), axis=0),
        #     upper=np.tile(np.percentile(yhat, 100., axis=0), reps=(len(crs), 1)))
    )

    cov = []
    for k, v in bounds.items():
        for i, cr in enumerate(crs):
            cov.append(
                (
                    k,
                    cr,
                    np.sum(np.int64(y >= v['lower'][i]) * np.int64(y <= v['upper'][i]))
                    / len(y),
                )
            )

    return pd.DataFrame(cov, columns=['method', 'cr', 'coverage'])


# TODO fix this at source
# Minor edit to a math fn to prevent annoying deprecation warnings
# Jon Sedar 2020-03-31
# Users/jon/anaconda/envs/instechex/lib/python3.6/site-packages/theano/tensor/subtensor.py:2339:
# FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated;
# use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index,
# `arr[np.array(seq)]`, which will result either in an error or a different result.
#   out[0][inputs[2:]] = inputs[1]


def expand_packed_triangular(n, packed, lower=True, diagonal_only=False):
    R"""Convert a packed triangular matrix into a two dimensional array.
    Triangular matrices can be stored with better space efficiancy by
    storing the non-zero values in a one-dimensional array. We number
    the elements by row like this (for lower or upper triangular matrices):
        [[0 - - -]     [[0 1 2 3]
         [1 2 - -]      [- 4 5 6]
         [3 4 5 -]      [- - 7 8]
         [6 7 8 9]]     [- - - 9]
    Parameters
    ----------
    n: int
        The number of rows of the triangular matrix.
    packed: theano.vector
        The matrix in packed format.
    lower: bool, default=True
        If true, assume that the matrix is lower triangular.
    diagonal_only: bool
        If true, return only the diagonal of the matrix.
    """
    if packed.ndim != 1:
        raise ValueError('Packed triagular is not one dimensional.')
    if not isinstance(n, int):
        raise TypeError('n must be an integer')

    if diagonal_only and lower:
        diag_idxs = np.arange(1, n + 1).cumsum() - 1
        return packed[diag_idxs]
    elif diagonal_only and not lower:
        diag_idxs = np.arange(2, n + 2)[::-1].cumsum() - n - 1
        return packed[diag_idxs]
    elif lower:
        out = pt.zeros((n, n), dtype=pytensor.config.floatX)
        idxs = np.tril_indices(n)
        return pt.set_subtensor(out[idxs], packed)
    elif not lower:
        out = pt.zeros((n, n), dtype=pytensor.config.floatX)
        idxs = np.triu_indices(n)
        return pt.set_subtensor(out[idxs], packed)


def log_jcd(f_inv_x, x):
    """Calc log of Jacobian determinant.
    Helps log-likelihood min of copula marginals, see JPL:
    + https://github.com/junpenglao/advance-bayesian-modelling-with-pymc/blob/master/Advance_topics/Box-Cox%20transformation.ipynb  # noqa: W505
    + https://github.com/junpenglao/Planet_Sakaar_Data_Science/blob/e39072eb65535adf743c6f0cd319fdf941cb2798/PyMC3QnA/Box-Cox%20transformation.ipynb  # noqa: W505
    + https://discourse.pymc.io/t/mixture-model-with-boxcox-transformation/988
    Used in oreum_lab copula experiments
    """
    return pt.log(pt.abs(pt.reshape(grad(pt.sum(f_inv_x), [x]), x.shape)))


def calc_2_sample_delta_prop(a, aref, a_index=None, fully_vectorised=False):
    r"""Calculate 2-side sample delta difference between arrays row-wise
    so that we can make a statement about the difference between a test array
    a reference array how different a is from aref

    Basic algo
    ----------

    for each row i in a:
        for each row j in aref:
            do: d = a[i] - aref[j]
            do: q = quantiles[0.03, 0.97](d)
            do: b_i = q > 0
            if: sum(b_i) == 2:
                we state "97% of a[i] > aref[j], substantially larger"
            elif: sum(b_i) == 1:
                we state "not different"
            else (sum(b_i) == 0):
                we state "97% of a[i] < aref[j], substantially smaller"
        do: prop = unique_count(b) / len(b)

        we state "prop a[i] larger | no different | smaller than aref"

    Parameters
    ----------
    a: 2D numpy array shape (nobs, nsamples), as returned by sample_ppc()
        This will be tested against the reference array arr_ref
        This is typically the prediction set (1 or more policies)

    aref: 2D numpy array shape (nobs, nsamples), as returned by sample_ppc()
        This is typically the training set (1 or more policies)

    a_index: Pandas series index, default None
        If a_index is not None, then prop_delta is returned as a DataFrame

    fully_vectorised : bool, default=False
        arr is not limited to a single policy
        If True, we use a numpy broadcasting to create a 3D array of deltas
            and perform both loops i, j in vectorised fashion
            see https://stackoverflow.com/a/43412438
            This is clever but consumes a lot of RAM (GBs)
        If False we loop the outer loop i
            This is more memory efficient and in testing seems faster!

    Returns
    -------
    prop_delta : numpy array of proportions of arr that are
        [0, 1, 2](substantially lower, no difference, substantially higher)
        than aref
        has shape (len(a), 3)
    """

    def _bincount_pad(a, maxval=2):
        b = np.bincount(a)
        return np.pad(
            b, (0, np.maximum(0, maxval + 1 - len(b))), 'constant', constant_values=(0)
        )

    # silently deal with the common mistake of sending a 1D array for testing
    # must be a horizontal slice
    if a.ndim == 1:
        a = a[np.newaxis, :]

    rope = np.array([0.03, 0.97])  # ROPE limits, must be len 2

    if fully_vectorised:
        delta = a[:, np.newaxis] - aref  # (len(a), len(aref), width(aref))
        delta_gt0 = 1 * (
            np.quantile(delta, rope, axis=2) > 0
        )  # (len(rope), len(a), len(aref))
        n_intersects = np.sum(delta_gt0, axis=0)  # (len(a), len(aref))

    else:
        n_intersects = np.empty(shape=(len(a), len(aref)))  # (len(a), len(aref))
        for i in range(len(a)):
            delta = a[i] - aref  # (len(aref), width(aref))
            delta_gt0 = 1 * (
                np.quantile(delta, rope, axis=1) > 0
            )  # (len(rope), len(aref))
            n_intersects[i] = np.sum(delta_gt0, axis=0)  # (len(aref))
        n_intersects = n_intersects.astype(np.int)

    prop_intersects_across_aref = np.apply_along_axis(
        lambda r: _bincount_pad(r, len(rope)), 1, n_intersects
    ) / len(
        aref
    )  # (len(a), [0, 1, 2])

    if a_index is not None:
        prop_intersects_across_aref = pd.DataFrame(
            prop_intersects_across_aref,
            columns=['subs_lower', 'no_difference', 'subs_higher'],
            index=a_index,
        )

    return prop_intersects_across_aref


def numpy_invlogit(x, eps=sys.float_info.epsilon):
    """The inverse of the logit function, 1 / (1 + exp(-x))."""
    return (1.0 - 2.0 * eps) / (1.0 + np.exp(-x)) + eps
