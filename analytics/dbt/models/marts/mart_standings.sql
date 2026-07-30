-- Team power board: current Elo, current-season record, and Monte-Carlo playoff odds.
-- One row per team. Feeds the standings / odds pages of the BI layer.
with odds as (select * from {{ ref('stg_playoff_odds') }}),
elo as (select team_id, elo from {{ ref('stg_elo_current') }}),

-- current-season games played, from the latest season present in the schedule
current_season as (
    select max(season) as season from {{ ref('stg_games') }}
),
record as (
    select
        team_id,
        count(*) as games_played,
        sum(wins) as wins,
        count(*) - sum(wins) as losses
    from (
        select home_id as team_id, home_win as wins
        from {{ ref('stg_games') }}
        where is_final and season = (select season from current_season)
        union all
        select away_id as team_id, 1 - home_win as wins
        from {{ ref('stg_games') }}
        where is_final and season = (select season from current_season)
    )
    group by team_id
)

select
    o.team_id,
    o.team_name,
    o.abbrev,
    o.league,
    o.division,
    e.elo,
    rank() over (order by e.elo desc)                    as elo_rank,
    coalesce(r.wins, o.current_wins)                     as wins,
    r.losses,
    r.games_played,
    round(o.projected_wins, 1)                           as projected_wins,
    o.p_make_playoffs,
    o.p_win_division,
    o.p_first_round_bye,
    o.p_win_pennant,
    o.p_win_world_series,
    rank() over (order by o.p_win_world_series desc)     as world_series_rank
from odds o
left join elo e using (team_id)
left join record r using (team_id)
