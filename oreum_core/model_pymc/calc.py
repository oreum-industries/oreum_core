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

# model.calc.py
"""Common Calculations for Model Evaluation"""
import sys
from typing import Any, Dict, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd
import pymc
import pytensor
import pytensor.gradient as tg
import pytensor.tensor as pt
from arviz import InferenceData, dict_to_dataset

# from fastprogress import progress_bar
from pymc.backends.arviz import _DefaultTrace, coords_and_dims_for_inferencedata
from pymc.model import Model, modelcontext
from pymc.pytensorf import PointFunc
from pymc.util import dataset_to_point_list

__all__ = [
    'get_log_jcd_scalar',
    'get_log_jcd_scan',
    'calc_f_beta',
    'calc_binary_performance_measures',
    'calc_rmse',
    'calc_r2',
    'calc_bayesian_r2',
    'calc_ppc_coverage',
    'expand_packed_triangular',
    'calc_2_sample_delta_prop',
    'numpy_invlogit',
    'compute_log_likelihood_for_potential',
]


RSD = 42
rng = np.random.default_rng(seed=RSD)


def get_log_jcd_scalar(
    f_inv_x: pt.TensorVariable, x: pt.TensorVariable
) -> pt.TensorVariable:
    """Calc log of determinant of Jacobian where f_inv_x and x are both (n,)
    dimensional tensors, and `f_inv_x` is a direct, element-wise transformation
    of `x` without cross-terms (diagonals). Add this Jacobian adjustment to
    models where observed is a transformation, to handle change in coords / volume.
    NOTE:
    + Slight abuse of terminology because for a 1D f_inv_x, we can sum to scalar
    + Initially developed for a model with 1D f_inv_x

    Also see detail from Stan docs:
    + https://mc-stan.org/docs/2_25/reference-manual/change-of-variables-section.html#multivariate-changes-of-variables
    + https://mc-stan.org/documentation/case-studies/mle-params.html
    "The absolute derivative of the inverse transform measures how the scale of
    the transformed variable changes with respect to the underlying variable."

    Example from jungpenglao:
    + https://github.com/junpenglao/Planet_Sakaar_Data_Science/blob/4366de036cc608c942fdebb930e96f2cc8b83d71/Ports/Jacobian%20Adjustment%20in%20PyMC3.ipynb#L163  # noqa: W505
    + https://github.com/junpenglao/Planet_Sakaar_Data_Science/blob/main/WIP/vector_transformation_copulas.ipynb

    More discussion:
    + https://www.tamaspapp.eu/post/jacobian-chain/
    + https://discourse.pymc.io/t/jacobian-adjustment/1711
    + https://github.com/junpenglao/All-that-likelihood-with-PyMC3/blob/master/Notebooks/Neals_funnel.ipynb
    + https://discourse.pymc.io/t/how-do-i-implement-an-upper-limit-log-normal-distribution/1337/4
    + https://github.com/junpenglao/advance-bayesian-modelling-with-PyMC3/blob/master/Advance_topics/Box-Cox%20transformation.ipynb  # noqa: W505
    + https://slideslive.com/38907842/session-3-model-parameterization-and-coordinate-system-neals-funnel
    + https://discourse.pymc.io/t/mixture-model-with-boxcox-transformation/988
    + https://pytensor.readthedocs.io/en/latest/library/gradient.html#pytensor.gradient.grad
    + https://www.pymc.io/projects/docs/en/latest/_modules/pymc/math.html#
    + https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/slides/lec10.pdf
    + https://github.com/pymc-devs/pymc-examples/blob/1428f1b4e0d352a88667776b3ec612db93e032d9/examples/case_studies/copula-estimation.ipynb  # noqa: W505
    + https://discourse.pymc.io/t/when-to-adjust-for-jacobian-and-when-to-skip-in-pymc3/2830
    """
    return pt.log(pt.abs(tg.grad(cost=pt.sum(f_inv_x), wrt=[x])))


def get_log_jcd_scan(
    f_inv_x: pt.TensorVariable,
    x: pt.TensorVariable,
    upstream_rvs: list[pt.TensorVariable] = None,
) -> pt.TensorVariable:
    """Calc log of determinant of Jacobian where f_inv_x and x are both (n, k)
    dimensional tensors, and `f_inv_x` is a direct, element-wise transformation
    of `x` without cross-terms (diagonals). Add this Jacobian adjustment to
    models where observed is a transformation, to handle change in coords / volume.
    NOTE:
    + Initially developed for a model where k = 2
    + Use explicit scan and strict, make sure to pass in any RVs upstream of
      f_inv_x in `upstream_rvs`: scan needs to see these in non_sequences,
      although we dont actually use them inside the inner function get_grads
    + Break into two scan calls to handle each k separately
    + Scales linearly with n, so expect to take a while for n > 200
    + Based on https://github.com/pymc-devs/pytensor/blob/7bb18f3a3590d47132245b7868b3a4a6587a4667/pytensor/gradient.py#L1984  # noqa W505
    + Also see https://discourse.pymc.io/t/something-changed-in-pytensor-2-12-3-and-thus-pymc-5-6-1-that-makes-my-pytensor-gradient-grad-call-get-stuck-any-ideas/13100/4  # noqa W505
    + Also see https://discourse.pymc.io/t/hitting-a-weird-error-to-do-with-rngs-in-scan-in-a-custom-function-inside-a-potential/13151/14  # noqa W505
    """
    n = f_inv_x.shape[0]
    idx = pt.arange(n, dtype="int32")
    non_seq = [f_inv_x, x]
    if upstream_rvs is not None:
        non_seq += upstream_rvs

    def _grads(i, s, c, w, *args):
        """Inner function allows for extra args that we dont actually use:
        because scan needs to know about upstream_rvs in non_sequences and also
        wants to passes them into the inner funciton _grad.
        See https://github.com/pymc-devs/pytensor/pull/191

        """
        return tg.grad(cost=c[i, s], wrt=[w])  # shape (n, n, k)

    kws = dict(fn=_grads, sequences=idx, n_steps=n, name="_grads", strict=True)
    grads0, _ = pytensor.scan(**kws, non_sequences=[0, *non_seq])
    grads1, _ = pytensor.scan(**kws, non_sequences=[1, *non_seq])
    grads = grads0.sum(axis=0) + grads1.sum(axis=0)  # shape (n, k)
    log_jcd = pt.sum(pt.log(pt.abs(grads)), axis=1)  # shape (n)
    return log_jcd


def calc_f_beta(precision: np.array, recall: np.array, beta: float = 1.0) -> np.array:
    """Set beta such that recall is beta times more important than precision"""
    with np.errstate(divide='ignore', invalid='ignore'):
        fb = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    return np.nan_to_num(fb, nan=0, posinf=0, neginf=0)


def calc_binary_performance_measures(y: np.array, yhat: np.array) -> pd.DataFrame:
    """Calculate tpr (recall), fpr, precision, accuracy for binary target,
    using quantiles of all samples from PPC, use vectorised calcs
    shapes y: (nsamples,), yhat: (nsamples, nobservations)
    """
    qs = np.round(np.arange(0, 1.01, 0.01), 2)
    yhat_q = np.quantile(yhat, qs, axis=0, method='linear').T
    y_mx = np.tile(y.reshape(-1, 1), 101)

    # calc tp, fp, tn, fn vectorized
    tp = np.nansum(np.where(yhat_q == 1, y_mx, np.nan), axis=0)
    fp = np.nansum(np.where(yhat_q == 1, 1 - y_mx, np.nan), axis=0)
    tn = np.nansum(np.where(yhat_q == 0, 1 - y_mx, np.nan), axis=0)
    fn = np.nansum(np.where(yhat_q == 0, y_mx, np.nan), axis=0)

    # calc tpr (recall), fpr, precision etc
    with np.errstate(divide='ignore', invalid='ignore'):
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
            'f0.5': calc_f_beta(precision, recall, beta=0.5),
            'f1': calc_f_beta(precision, recall, beta=1),
            'f2': calc_f_beta(precision, recall, beta=2),
        },
        index=qs,
    )
    perf.index.set_names('q', inplace=True)

    return perf.round(6)


def calc_rmse(
    dfhat: pd.DataFrame,
    oid: str = 'oid',
    yhat: str = 'yhat',
    y: str = 'y',
    method: str = 'a',
    qs: bool = False,
    mse_only: bool = False,
) -> float | tuple[float, pd.Series]:
    r"""Calculate (R)MSE using the mean PPC samples:

    Require that dfhat is a long-format left join of PPC samples `yhat` and
    observed `y` values (which of course will contain nsamples duplicates)
    ie.: len(dfhat) == (nobs * nsamples)
    and: dfhat has index [oid, chain, draw] and fts [yhat, y]

    Optionally set mse_only = True to get just the MSE (default is RMSE)
    Optionally set qs = True to also return a Series of quantiles

    For the SE calc, we have two options:
    + A: compress samples to a summary statistic before calculate MSE
    + B: calculate MSE for all samples and then take summary statistic

    \begin{align}
    \text{A:} \text{MSE} &= \frac{1}{n} \sum_{i=1}^{i=n}(\mu_{j}(\hat{y}_{i}) - y_{i})^{2} \\
    \text{B:} \text{MSE} &= \frac{1}{n} \sum_{i=1}^{i=n}(\mu_{j}(\hat{y}_{ij} - y_{i})^{2})
    \end{align}

    Option B can create huge MSE values because outliers for each obs are
    squared and push farther from the mean. This might be useful for a very
    sensitive analysis.

    i.e.
    take mean across samples then square the differences is usually smaller
    than square the differences each sample and preserve samples
    https://en.wikipedia.org/wiki/Generalized_mean

    But here we will default to the easier-to-understand Option A
    """
    if method not in ['a', 'b']:
        raise AttributeError("method must be in `a` or `b`")

    def _grp_se_at_mn(g):
        if method == 'a':
            return np.power(g[yhat].mean() - g[y].max(), 2)
        else:
            return np.power(g[yhat] - g[y], 2).mean()

    def _grp_se_at_qs(g):
        qs = np.round(np.linspace(0, 1, 100 + 1), 2)
        if method == 'a':
            se_at_qs = np.power(np.quantile(g[yhat], qs) - g[y].max(), 2)
        else:
            se_at_qs = np.quantile(np.power(g[yhat] - g[y], 2), qs)

        s_se_at_qs = pd.Series(se_at_qs, index=qs, name='se')
        s_se_at_qs.index.rename('q', inplace=True)
        return s_se_at_qs

    mse_at_mn = dfhat.groupby(level=oid).apply(_grp_se_at_mn).mean(axis=0)
    if not qs:
        if mse_only:
            return mse_at_mn
        else:
            return np.sqrt(mse_at_mn)
    else:
        mse_at_qs = dfhat.groupby(level=oid).apply(_grp_se_at_qs).mean(axis=0)
        if mse_only:
            return mse_at_mn, mse_at_qs
        else:
            rmse_at_qs = mse_at_qs.map(np.sqrt)
            rmse_at_qs._set_name('rmse', inplace=True)
            return np.sqrt(mse_at_mn), rmse_at_qs


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


def expand_packed_triangular(n, packed, lower=True, diagonal_only=False):
    r"""Convert a packed triangular matrix into a two dimensional array.
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


def compute_log_likelihood_for_potential(
    idata: InferenceData,
    *,
    var_names: Optional[Sequence[str]] = None,
    extend_inferencedata: bool = True,
    model: Optional[Model] = None,
    sample_dims: Sequence[str] = ("chain", "draw"),
    progressbar=True,
):
    """Hackidy hack JS @ 20230919
    Copied and modified from pymc compute_log_likelihood to allow a Potential
    Differences NOTED with inline comments
    orig: https://github.com/pymc-devs/pymc/blob/92278278d4a8b78f17ed0f101eb29d0d9982eb45/pymc/stats/log_likelihood.py#L29C1-L128C31
    discussion: https://discourse.pymc.io/t/using-a-random-variable-as-observed/7184/10

    IMPORTANT NOTE 2024-08-04 in the intervening time, the source function that
    this copies / modifies has changed hugely - it's going to cause substantial
    pain to update :S

    ---

    Compute elemwise log_likelihood of model given InferenceData with posterior group

    Parameters
    ----------
    idata : InferenceData
        InferenceData with posterior group
    var_names : sequence of str, optional
        List of Potential variable names for which to compute log_likelihood
    extend_inferencedata : bool, default True
        Whether to extend the original InferenceData or return a new one
    model : Model, optional
    sample_dims : sequence of str, default ("chain", "draw")
    progressbar : bool, default True

    Returns
    -------
    idata : InferenceData
        InferenceData with log_likelihood group

    """

    posterior = idata["posterior"]

    model = modelcontext(model)

    observed_vars = [model.named_vars[name] for name in var_names]

    if var_names is None:
        observed_vars = model.observed_RVs
        var_names = tuple(rv.name for rv in observed_vars)
    else:
        observed_vars = [model.named_vars[name] for name in var_names]
        if not set(observed_vars).issubset(
            model.observed_RVs + model.potentials
        ):  # NOTE MODIFIED JS
            raise ValueError(
                f"var_names must refer to observed_RVs in the model. Got: {var_names}"
            )

    # We need to temporarily disable transforms, because the InferenceData only keeps the untransformed values
    # pylint: disable=used-before-assignment
    try:
        original_rvs_to_values = model.rvs_to_values
        original_rvs_to_transforms = model.rvs_to_transforms

        model.rvs_to_values = {
            rv: rv.clone() if rv not in model.observed_RVs else value
            for rv, value in model.rvs_to_values.items()
        }
        model.rvs_to_transforms = {rv: None for rv in model.basic_RVs}

        elemwise_loglike_fn = model.compile_fn(
            inputs=model.value_vars,
            outs=model.logp(vars=observed_vars, sum=False),
            on_unused_input="ignore",
        )
        elemwise_loglike_fn = cast(PointFunc, elemwise_loglike_fn)
    finally:
        model.rvs_to_values = original_rvs_to_values
        model.rvs_to_transforms = original_rvs_to_transforms
    # pylint: enable=used-before-assignment

    # Ignore Deterministics
    posterior_values = posterior[[rv.name for rv in model.free_RVs]]
    posterior_pts, stacked_dims = dataset_to_point_list(posterior_values, sample_dims)
    n_pts = len(posterior_pts)
    loglike_dict = _DefaultTrace(n_pts)
    indices = range(n_pts)
    # if progressbar:
    #     indices = progress_bar(indices, total=n_pts, display=progressbar)

    for idx in indices:
        loglikes_pts = elemwise_loglike_fn(posterior_pts[idx])
        for rv_name, rv_loglike in zip(var_names, loglikes_pts):
            loglike_dict.insert(rv_name, rv_loglike, idx)

    loglike_trace = loglike_dict.trace_dict
    for key, array in loglike_trace.items():
        loglike_trace[key] = array.reshape(
            (*[len(coord) for coord in stacked_dims.values()], *array.shape[1:])
        )

    coords, dims = coords_and_dims_for_inferencedata(model)
    loglike_dataset = dict_to_dataset(
        loglike_trace,
        library=pymc,
        dims=dims,
        coords=coords,
        default_dims=list(sample_dims),
        skip_event_dims=True,
    )

    if extend_inferencedata:
        idata.add_groups(dict(log_likelihood=loglike_dataset))
        return idata
    else:
        return loglike_dataset
