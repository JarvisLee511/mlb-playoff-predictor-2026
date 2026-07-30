-- Per team-game rolling form, knowable BEFORE the game. SQL re-expression of the
-- deque/dict state machine in src/features.py::build_features. Key points matched:
--   * the 30-game window carries ACROSS seasons (deque is keyed by team only)
--   * season win% and rest days reset / key per season and per team respectively
--   * the exact Python defaults when there isn't enough history (0.5 / 0.0 / 3 days)
{% set roll_window = 30 %}
{% set min_games = 10 %}

with logs as (
    select
        team_id,
        season,
        game_id,
        game_date,
        case when runs_scored > runs_allowed then 1 else 0 end as win,
        runs_scored - runs_allowed as run_diff
    from {{ ref('stg_gamelogs') }}
),

windowed as (
    select
        team_id,
        season,
        game_id,
        game_date,
        -- last up-to-30 prior games (across seasons, like the deque)
        count(*)      over w30 as n_prior_30,
        avg(win)      over w30 as winpct_30_raw,
        avg(run_diff) over w30 as rundiff_30_raw,
        -- last up-to-10 prior games
        count(*)      over w10 as n_prior_10,
        avg(win)      over w10 as winpct_10_raw,
        -- season-to-date (resets per season)
        count(*)      over wseason as n_prior_season,
        avg(win)      over wseason as season_winpct_raw,
        -- rest days = gap since this team's previous game (any season)
        date_diff('day',
                  lag(game_date) over (partition by team_id order by game_date, game_id),
                  game_date) as rest_gap
    from logs
    window
        w30 as (partition by team_id order by game_date, game_id
                rows between {{ roll_window }} preceding and 1 preceding),
        w10 as (partition by team_id order by game_date, game_id
                rows between 10 preceding and 1 preceding),
        wseason as (partition by team_id, season order by game_date, game_id
                    rows between unbounded preceding and 1 preceding)
)

select
    team_id,
    season,
    game_id,
    game_date,
    n_prior_30,
    (n_prior_30 >= {{ min_games }})                                          as enough_history,
    case when n_prior_30 >= {{ min_games }} then winpct_30_raw  else 0.5 end as winpct_30,
    case when n_prior_30 >= {{ min_games }} then rundiff_30_raw else 0.0 end as rundiff_30,
    case when n_prior_10 >= 5             then winpct_10_raw   else 0.5 end as winpct_10,
    case when n_prior_season > 0          then season_winpct_raw else 0.5 end as season_winpct,
    -- default 3 days when no prior game; clip to [0, 10]
    least(greatest(coalesce(rest_gap, 3), 0), 10)                           as rest_days
from windowed
