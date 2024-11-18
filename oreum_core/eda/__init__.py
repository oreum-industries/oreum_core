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

# eda/
"""Various classes & functions for exploratory data analysis"""
from .calc import (
    bootstrap,
    bootstrap_lr,
    calc_geometric_cv,
    calc_location_in_ecdf,
    calc_svd,
    calc_umap,
    fit_and_plot_fn,
    get_gini,
    month_diff,
    tril_nan,
)
from .describe import describe, display_fw, display_ht, get_fts_by_dtype
from .eda_io import FigureIO, output_data_dict
from .plot import (  # plot_umap,; plot_r2_range,; plot_r2_range_pair,
    plot_accuracy,
    plot_binary_performance,
    plot_bool_ct,
    plot_bootstrap_delta_grp,
    plot_bootstrap_grp,
    plot_bootstrap_lr,
    plot_bootstrap_lr_grp,
    plot_cat_ct,
    plot_cdf_ppc_vs_obs,
    plot_coverage,
    plot_date_ct,
    plot_estimate,
    plot_explained_variance,
    plot_f_measure,
    plot_float_dist,
    plot_grp_ct,
    plot_heatmap_corr,
    plot_int_dist,
    plot_joint_numeric,
    plot_kj_summaries_for_linear_model,
    plot_mincovdet,
    plot_rmse_range,
    plot_roc_precrec,
    plot_smrystat,
    plot_smrystat_grp,
    plot_smrystat_grp_year,
)
