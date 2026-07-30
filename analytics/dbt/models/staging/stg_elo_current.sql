-- Current Elo rating per team.
with source as (
    select * from {{ source('mlb_pipeline', 'elo_current') }}
)
select
    cast(team_id as integer) as team_id,
    cast(elo as double)      as elo,
    team_name,
    abbrev,
    league,
    division
from source
