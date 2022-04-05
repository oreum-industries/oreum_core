# eda/
# Various classes & functions for exploratory data analysis
# copyright 2022 Oreum Industries
from oreum_core.eda.describe import (
    custom_describe,
    display_fw,
    display_ht,
    get_fts_by_dtype,
    output_data_dict,
)
from oreum_core.eda.plot import (
    plot_cat_count,
    plot_bool_count,
    plot_date_count,
    plot_int_dist,
    plot_float_dist,
    plot_joint_ft_x_tgt,
    plot_mincovdet,
    plot_roc_precrec,
    plot_f_measure,
    plot_accuracy,
    plot_binary_performance,
    plot_coverage,
    plot_rmse_range,
    plot_rmse_range_pair,
    plot_r2_range,
    plot_r2_range_pair,
    plot_ppc_vs_observed,
    plot_bootstrap_lr,
    plot_bootstrap_lr_grp,
    plot_bootstrap_grp,
    plot_bootstrap_delta_grp,
    plot_grp_sum_dist_count,
    plot_grp_year_sum_dist_count,
    plot_heatmap_corr,
    plot_kj_summaries_for_linear_model,
    plot_grp_count,
)
from oreum_core.eda.calc import (
    fit_and_plot_fn,
    get_gini,
    bootstrap,
    bootstrap_lr,
    calc_geometric_cv,
    calc_location_in_ecdf,
    month_diff,
)
from oreum_core.eda.utils import (
    display_image_file,
)
