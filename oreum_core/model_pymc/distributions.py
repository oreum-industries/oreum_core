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

# model.distributions.py
"""Handful of additional useful distributions / transforms"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc.distributions.dist_math import check_icdf_parameters, check_icdf_value
from scipy import stats

__all__ = ['sanity_check_lognorm', 'normal_icdf', 'lognormal_icdf', 'mv_dist']

# NOTE hack to clip values away from {0, 1} for invcdfs
# Whilst value = {0, 1} is theoretically allowed, is seems to cause a
# numeric compuational issue somewhere in pt.erfcinv which throws infs.
# This screws up the downstream, so clip slightly away from {0, 1}
CLIP = 1e-15  # NOTE 1e-18 too small


def sanity_check_lognorm(mu: float = 0.0, sigma: float = 1.0) -> None:
    """Sanity checker to confirm parameterisation of lognorm dists
    between scipy and pymc"""
    n = 1000
    x = np.linspace(0, 100, n)
    fd = stats.lognorm(scale=np.exp(mu), s=sigma)
    y_scipy = fd.pdf(x)
    rv = pm.LogNormal.dist(mu=mu, sigma=sigma)
    y_pymc = np.exp(pm.logp(rv, x).eval())
    assert sum(np.isclose(y_scipy, y_pymc)) == n


def normal_icdf(
    x: pt.TensorVariable, mu: pt.TensorVariable = 0.0, sigma: pt.TensorVariable = 1.0
) -> pt.TensorVariable:
    """Normal iCDF aka InverseCDF aka PPF aka Probit. Default mean-center 1sd
    NOTE:
    + Modified from pymc.distributions.continuous.Normal.icdf
    + Slow implementation of erfcinv not in C
    + See also https://stackoverflow.com/questions/60472139/computing-the-inverse-of-the-complementary-error-function-erfcinv-in-c
    + Hack to clip values away from edges [0., 1.]:
        Whilst value = {0, 1} is theoretically allowed, it seems to cause a
        numeric computational issue somewhere in pt.erfcinv which throws infs.
        This screws up the downstream, so clip slightly away from edges [0, 1]
    + Used in oreum_lab.src.model.copula.model_a
        NOTE: Possibly after pymc > 5.5 will change to use
        y_cop_u_rv = pm.Normal.dist(mu=0., sigma=1.)
        pm.icdf(y_cop_u_rv, pt.stack([y_m1u, y_m2u], axis=1)),
    """
    x = pt.clip(x, CLIP, 1 - CLIP)
    r = check_icdf_value(mu - sigma * pt.sqrt(2.0) * pt.erfcinv(2.0 * x), x)
    return check_icdf_parameters(r, sigma > 0.0, msg="sigma > 0")


def lognormal_icdf(
    x: pt.TensorVariable, mu: pt.TensorVariable = 0.0, sigma: pt.TensorVariable = 1.0
) -> pt.TensorVariable:
    """LogNormal icdf, defaulted to mean-centered, 1sd
    NOTE:
    + Modified from pymc.distributions.continuous.LogNormal.icdf in pymc > v5.5
    """
    return pt.exp(normal_icdf(x=x, mu=mu, sigma=sigma))


def mv_dist(
    chol: pt.TensorVariable,
    n: int,
    size: pt.TensorVariable,  # required by CustomDist, but we ignore it for clarity
) -> pt.TensorVariable:
    """Hack to wrap a conventional MvNormal inside a CustomDist
    So that we can hack it to use an observed that contains a FreeRV"""
    return pm.MvNormal.dist(mu=pt.zeros(2), chol=chol, shape=(n, 2))
