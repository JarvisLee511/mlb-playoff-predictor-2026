-- One row per pitcher-appearance. is_start flags a game the pitcher started.
with source as (
    select * from {{ source('mlb_pipeline', 'pitcher_logs') }}
)
select
    cast(season as integer)     as season,
    cast(pitcher_id as bigint)  as pitcher_id,
    cast(team_id as integer)    as team_id,
    cast(game_id as bigint)     as game_id,
    cast(date as date)          as game_date,
    cast(gs as integer)         as games_started,
    (cast(gs as integer) = 1)   as is_start,
    cast(ip as double)          as innings_pitched,
    cast(er as integer)         as earned_runs,
    cast(bb as integer)         as walks,
    cast(so as integer)         as strike_outs,
    cast(hr as integer)         as home_runs,
    cast(hbp as integer)        as hit_by_pitch,
    cast(bf as integer)         as batters_faced
from source
