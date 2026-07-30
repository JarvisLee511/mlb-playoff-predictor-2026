-- Out-of-sample track record: one row per predicted game. The five p_home_* columns
-- are each model's pre-game probability that the home team wins. Later-added models
-- (ens, skl) are null on older rows. status flips pending -> final once scored.
with source as (
    select * from {{ source('mlb_pipeline', 'predictions_log') }}
)
select
    cast(date as date)        as game_date,
    cast(game_id as bigint)   as game_id,
    cast(home_id as integer)  as home_id,
    home_name,
    cast(away_id as integer)  as away_id,
    away_name,
    game_time_et,
    home_pitcher,
    away_pitcher,
    status,
    (lower(status) = 'final')       as is_final,
    try_cast(home_score as integer) as home_score,
    try_cast(away_score as integer) as away_score,
    try_cast(home_win as integer)   as home_win,
    try_cast(p_home_elo as double)  as p_home_elo,
    try_cast(p_home_lr as double)   as p_home_lr,
    try_cast(p_home_xgb as double)  as p_home_xgb,
    try_cast(p_home_ens as double)  as p_home_ens,
    try_cast(p_home_skl as double)  as p_home_skl
from source
