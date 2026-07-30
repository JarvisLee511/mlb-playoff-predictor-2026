-- Reliability curve per model: predicted probability vs realised win rate per bin.
-- calibration_gap = fraction_won - mean_predicted (0 = perfectly calibrated).
select
    model,
    row_number() over (partition by model order by mean_predicted) as bin_index,
    mean_predicted,
    fraction_won,
    calibration_gap,
    abs(calibration_gap) as abs_gap
from {{ ref('stg_calibration') }}
