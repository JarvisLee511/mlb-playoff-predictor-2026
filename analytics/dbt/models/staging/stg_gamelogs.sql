-- One row per team-game. Typed + renamed passthrough of the raw hitting/pitching
-- box lines. h_* = this team's offense, p_* = pitching allowed.
with source as (
    select * from {{ source('mlb_pipeline', 'gamelogs') }}
)
select
    cast(season as integer)              as season,
    cast(team_id as integer)             as team_id,
    cast(game_id as bigint)              as game_id,
    cast(date as date)                   as game_date,

    -- hitting
    cast(h_atBats as integer)            as at_bats,
    cast(h_hits as integer)              as hits,
    cast(h_doubles as integer)           as doubles,
    cast(h_triples as integer)           as triples,
    cast(h_homeRuns as integer)          as home_runs,
    cast(h_baseOnBalls as integer)       as walks,
    cast(h_hitByPitch as integer)        as hit_by_pitch,
    cast(h_sacFlies as integer)          as sac_flies,
    cast(h_strikeOuts as integer)        as strike_outs,
    cast(h_plateAppearances as integer)  as plate_appearances,
    cast(h_runs as integer)              as runs_scored,

    -- pitching allowed
    cast(p_ip as double)                 as innings_pitched,
    cast(p_earnedRuns as integer)        as earned_runs,
    cast(p_runs as integer)              as runs_allowed,
    cast(p_hits as integer)              as hits_allowed,
    cast(p_baseOnBalls as integer)       as walks_allowed,
    cast(p_strikeOuts as integer)        as strike_outs_pitched,
    cast(p_homeRuns as integer)          as home_runs_allowed,
    cast(p_hitByPitch as integer)        as hit_by_pitch_allowed
from source
