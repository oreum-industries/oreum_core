# copyright 2021 Oreum OÜ
from oreum_core.eda.describe import (
    custom_describe, 
    display_fw, 
    get_fts_by_dtype
    )
from oreum_core.eda.plot import (
    plot_cat_count,
    plot_date_count,
    plot_int_dist,
    plot_float_dist,
    plot_joint_ft_x_tgt,    
    plot_mincovdet,
    plot_rmse_range,
    plot_rmse_range_pair,
    plot_r2_range,
    plot_r2_range_pair,
    )
from oreum_core.eda.calc import (
    fit_fn,
    get_gini
    )