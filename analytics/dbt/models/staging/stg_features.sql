-- The Python-built modeling table (data/processed/features.csv). Used ONLY as the
-- reference "answer key" for mart_feature_validation and as the source of the two
-- features NOT re-expressed in SQL (Elo rating + bullpen snapshots, which come from
-- separate engines). Not an input to the SQL feature reimplementation.
with source as (
    select * from {{ source('mlb_pipeline', 'features') }}
)
select
    cast(game_id as bigint)          as game_id,
    cast(home_id as integer)         as home_id,
    cast(away_id as integer)         as away_id,
    -- Elo engine output (src/models/elo.py) — consumed as-is
    try_cast(elo_diff as double)     as elo_diff,
    try_cast(elo_prob_home as double) as elo_prob_home,
    -- bullpen engine output (src/features.py::build_bullpen_snapshots)
    try_cast(bp_fip_diff as double)      as bp_fip_diff,
    try_cast(bp_fatigue_diff as double)  as bp_fatigue_diff,
    -- reference values for validation of the SQL reimplementation
    try_cast(park_hfa as double)         as ref_park_hfa,
    try_cast(home_winpct_30 as double)   as ref_home_winpct_30,
    try_cast(home_rundiff_30 as double)  as ref_home_rundiff_30,
    try_cast(home_season_winpct as double) as ref_home_season_winpct,
    try_cast(home_rest_days as double)   as ref_home_rest_days,
    try_cast(home_ops as double)         as ref_home_ops,
    try_cast(home_era as double)         as ref_home_era,
    try_cast(home_fip as double)         as ref_home_fip,
    try_cast(home_whip as double)        as ref_home_whip,
    try_cast(home_pyth as double)        as ref_home_pyth,
    try_cast(home_off_bbk as double)     as ref_home_off_bbk,
    try_cast(home_sp_fip as double)      as ref_home_sp_fip,
    try_cast(home_sp_kbb as double)      as ref_home_sp_kbb,
    try_cast(home_sp_fip5 as double)     as ref_home_sp_fip5
from source
