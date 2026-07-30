select
    feature,
    n_compared,
    max_abs_diff,
    mean_abs_diff,
    pct_within_tol
from main_marts.mart_feature_validation
order by max_abs_diff desc
