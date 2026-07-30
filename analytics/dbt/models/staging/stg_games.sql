-- One row per game_id: final scores and the home_win label.
with source as (
    select * from {{ source('mlb_pipeline', 'games') }}
)
select
    cast(season as integer)   as season,
    cast(game_id as bigint)   as game_id,
    cast(date as date)        as game_date,
    status,
    cast(home_id as integer)  as home_id,
    home_name,
    cast(away_id as integer)  as away_id,
    away_name,
    try_cast(home_score as integer) as home_score,
    try_cast(away_score as integer) as away_score,
    try_cast(home_win as integer)   as home_win,
    (lower(status) = 'final') as is_final
from source
