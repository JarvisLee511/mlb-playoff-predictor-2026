-- Monte-Carlo (10k-sim) playoff odds per team for the current season.
with source as (
    select * from {{ source('mlb_pipeline', 'playoff_odds') }}
)
select
    cast(team_id as integer)          as team_id,
    cast(current_wins as integer)     as current_wins,
    cast(proj_wins as double)         as projected_wins,
    cast(make_playoffs as double)     as p_make_playoffs,
    cast(win_division as double)      as p_win_division,
    cast(first_round_bye as double)   as p_first_round_bye,
    cast(win_pennant as double)       as p_win_pennant,
    cast(win_world_series as double)  as p_win_world_series,
    team_name,
    abbrev,
    league,
    division
from source
