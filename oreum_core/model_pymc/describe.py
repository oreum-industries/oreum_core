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

# model.describe.py
"""Model Descriptions"""
import re

import arviz as az
import numpy as np
import pandas as pd
import patsy as pat

from ..model_pymc import BasePYMCModel

__all__ = [
    'model_desc',
    'extract_yobs_yhat',
    'describe_dist',
    'get_summary',
    'print_rvs',
]

RSD = 42
rng = np.random.default_rng(seed=RSD)


def model_desc(fml: str) -> str:
    """Convenience: return patsy modeldesc
    NOTE: `.describe()` doesn't return the `1 +` (intercept) term in the
        case that it's present. check and add if needed
    """
    fmls = fml.split(' ~ ')
    add_intercept = False if re.match(r'1 \+', fml) is None else True
    r = pat.ModelDesc.from_formula(fml).describe()
    if len(fmls) == 2:
        rs = r.split(' ~ ')
        if add_intercept:
            r = f'{rs[0]} ~ 1 + {rs[1]}'
    elif len(fmls) == 1:
        if add_intercept:
            r = f'1 + {r[2:]}'
    else:
        raise ValueError('fml must have only a single tilde `~`')
    return f'patsy linear model desc:\n{r}\n'


def extract_yobs_yhat(
    idata: az.InferenceData, obs: str = 'y', pred: str = 'yhat'
) -> tuple:
    """Convenience: extract y_obs, y_hat from idata
    get yhat in the shape (nsamples, nobs)
    """
    nsamp = np.product(idata.posterior_predictive[pred].shape[:-1])
    yobs = idata.constant_data[obs].values  # (nobs,)
    yhat = idata.posterior_predictive[pred].values.reshape(nsamp, -1)  # (nsamp, nobs)
    return yobs, yhat


def describe_dist(mdl: BasePYMCModel, log: bool = False, inc_summary: bool = False):
    """Convenience: get distribution descriptions from distributions.DistNumpy
    and return for printing or Markdown
    NOTE: consider deprecating
    """
    title = f'{mdl.name}: Natural Distributions'
    dist = mdl.dist_natural
    if log:
        title = f'{mdl.name}: Logged Distributions'
        dist = mdl.dist_log

    if inc_summary:
        return title, {**mdl.notation, **dist, **mdl.conditions, **mdl.summary_stats}
    return title, {**dist}


def get_summary(mdl: BasePYMCModel, rvs: list, group='posterior') -> pd.DataFrame:
    """Convenience fn to get arviz summary of idata posteriors"""
    df = az.summary(mdl.idata, var_names=rvs, group=group)
    return df


def print_rvs(mdl: BasePYMCModel) -> list[str]:
    """Convenience to provide RV strings to display in Notebook
    Use as _ = [display(Markdown(s for s in mt.print_rvs(mdl)))]
    """
    r = []
    for k, rvs in mdl.get_rvs().items():
        if k in ['free', 'potentials', 'deterministics']:
            for rv in rvs:
                try:
                    r.append(rv.str_repr(formatting='string', include_params=True))
                except AttributeError:
                    # initially developed as a bit of a hack to deal with
                    # 'TensorVariable' object has no attribute 'str_repr'
                    # in the case of e.g. autoimputed x_mv_unobserved
                    pass
    return r


# def print_rvs(rvs: list[pt.TensorVariable]) -> None:
#     """Display rvs to Notebook using latex, post sub underscores
#     Hack to replace print(mdl.model) see https://github.com/pymc-devs/pymc/issues/6869
#     """
#     rx_name_usc = re.compile(r"(?:text\{[^\_\}]+?)(\_+)(?:[^\_\}]+?)(?:(\_+)(?:[^\_\}]+?))*?(?:\})", re.I)
#     for rv in rvs:
#         s = rv.str_repr(formatting='latex', include_params=True)
#         t = rx_name_usc.sub(r"\\_", s)
#         display(Markdown(t))
