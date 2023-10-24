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
    calc_f_beta,
    calc_mse,
    calc_ppc_coverage,
    calc_r2,
    calc_rmse,
    compute_log_likelihood_for_potential,
    get_log_jcd_scalar,
    get_log_jcd_scan,
    numpy_invlogit,
)
from oreum_core.model.describe import (
    describe_dist,
    extract_yobs_yhat,
    get_summary,
    model_desc,
    print_rvs,
)
from oreum_core.model.distributions import (
    lognormal_icdf,
    mv_dist,
    normal_icdf,
    sanity_check_lognorm,
)
from oreum_core.model.model_io import ModelIO
from oreum_core.model.plot import (
    facetplot_krushke,
    forestplot_multiple,
    forestplot_single,
    pairplot_corr,
    plot_compare,
    plot_energy,
    plot_loo_pit,
    plot_ppc,
    plot_trace,
)
