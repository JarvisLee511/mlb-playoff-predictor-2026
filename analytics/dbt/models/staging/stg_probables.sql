-- One row per game_id: probable starting pitcher ids.
with source as (
    select * from {{ source('mlb_pipeline', 'probables') }}
)
select
    cast(season as integer)      as season,
    cast(game_id as bigint)      as game_id,
    try_cast(home_sp_id as bigint) as home_sp_id,
    try_cast(away_sp_id as bigint) as away_sp_id
from source
