-- Reliability bins: for each model and probability bin, the mean predicted
-- probability vs the actual fraction of games won. Perfect calibration => equal.
with source as (
    select * from {{ source('mlb_pipeline', 'calibration') }}
)
select
    model,
    cast(mean_predicted as double) as mean_predicted,
    cast(fraction_won as double)   as fraction_won,
    cast(fraction_won as double) - cast(mean_predicted as double) as calibration_gap
from source
