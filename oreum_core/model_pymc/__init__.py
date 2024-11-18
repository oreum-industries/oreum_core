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

# model_pymc/
"""Various classes & functions for modelling, using PyMC"""
from .base import BasePYMCModel
from .calc import (
    calc_2_sample_delta_prop,
    calc_bayesian_r2,
    calc_binary_performance_measures,
    calc_f_beta,
    calc_ppc_coverage,
    calc_r2,
    calc_rmse,
    compute_log_likelihood_for_potential,
    get_log_jcd_scalar,
    get_log_jcd_scan,
    numpy_invlogit,
)
from .describe import (
    describe_dist,
    extract_yobs_yhat,
    get_summary,
    model_desc,
    print_rvs,
)
from .distributions import lognormal_icdf, mv_dist, normal_icdf, sanity_check_lognorm
from .plot import (
    facetplot_krushke,
    forestplot_multiple,
    forestplot_single,
    pairplot_corr,
    plot_compare,
    plot_energy,
    plot_lkjcc_corr,
    plot_loo_pit,
    plot_ppc,
    plot_trace,
    plot_yhat_vs_y,
)
from .pymc_io import PYMCIO
