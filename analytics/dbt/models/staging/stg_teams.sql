-- Team dimension.
with source as (
    select * from {{ source('mlb_pipeline', 'teams') }}
)
select
    cast(team_id as integer) as team_id,
    team_name,
    abbrev,
    league,
    division
from source
