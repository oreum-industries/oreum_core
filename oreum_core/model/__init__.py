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

# model/
"""Various classes & functions for modelling, primarily using pymc"""
from oreum_core.model.base import BasePYMCModel
from oreum_core.model.calc import (
    calc_2_sample_delta_prop,
    calc_bayesian_r2,
    calc_binary_performance_measures,
    calc_dist_fns_over_x,
    calc_dist_fns_over_x_manual_only,
    calc_mse,
    calc_ppc_coverage,
    calc_r2,
    calc_rmse,
    log_jcd,
    numpy_invlogit,
)
from oreum_core.model.describe import (
    describe_dist,
    extract_yobs_yhat,
    get_posterior_summary,
    model_desc,
)

# from oreum_core.model.distributions import (
#     InverseWeibull,
#     InverseWeibullNumpy,
#     Kumaraswamy,
#     Lognormal,
#     LognormalNumpy,
#     Normal,
#     NormalNumpy,
#     ZeroInflatedInverseWeibull,
#     ZeroInflatedLogNormal,
#     boundzero_numpy,
#     boundzero_theano,
# )
from oreum_core.model.model_io import ModelIO
from oreum_core.model.plot import (
    facetplot_krushke,
    forestplot_multiple,
    forestplot_single,
    pairplot_corr,
    plot_dist_fns_over_x,
    plot_dist_fns_over_x_manual_only,
    plot_energy,
    plot_ppc_loopit,
    plot_trace,
)
