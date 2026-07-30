-- Season-to-date advanced rate stats, knowable BEFORE each game.
-- SQL re-expression of src/features.py::build_advanced_pregame. The Python version
-- does `grp[counting].cumsum() - g[counting]` (cumulative up to but EXCLUDING the
-- current game); here that is a window frame of ROWS UNBOUNDED PRECEDING .. 1 PRECEDING
-- partitioned by (team_id, season), ordered by (date, game_id).
{% set adv_min_games = 15 %}
{% set pyth_exp = 1.83 %}

with logs as (
    select * from {{ ref('stg_gamelogs') }}
),

pre as (
    select
        team_id,
        game_id,
        game_date,
        season,
        count(*) over w                          as games_before,
        -- hitting counting stats (strictly-prior cumulative)
        sum(at_bats) over w                      as ab,
        sum(hits) over w                         as h,
        sum(doubles) over w                      as b2,
        sum(triples) over w                      as b3,
        sum(home_runs) over w                    as hr,
        sum(walks) over w                        as bb,
        sum(hit_by_pitch) over w                 as hbp,
        sum(sac_flies) over w                    as sf,
        sum(strike_outs) over w                  as so,
        sum(plate_appearances) over w            as pa,
        sum(runs_scored) over w                  as rs,
        -- pitching-allowed counting stats
        sum(innings_pitched) over w              as p_ip,
        sum(earned_runs) over w                  as p_er,
        sum(runs_allowed) over w                 as ra,
        sum(hits_allowed) over w                 as p_h,
        sum(walks_allowed) over w                as p_bb,
        sum(strike_outs_pitched) over w          as p_so,
        sum(home_runs_allowed) over w            as p_hr,
        sum(hit_by_pitch_allowed) over w         as p_hbp
    from logs
    window w as (
        partition by team_id, season
        order by game_date, game_id
        rows between unbounded preceding and 1 preceding
    )
),

calc as (
    select
        team_id,
        game_id,
        game_date,
        season,
        games_before,
        -- OBP + SLG
        (h + bb + hbp) / nullif(ab + bb + hbp + sf, 0)
            + (h + b2 + 2 * b3 + 3 * hr) / nullif(ab, 0)        as ops,
        9.0 * p_er / nullif(p_ip, 0)                            as era,
        (13 * p_hr + 3 * (p_bb + p_hbp) - 2 * p_so) / nullif(p_ip, 0) + 3.10 as fip,
        (p_bb + p_h) / nullif(p_ip, 0)                          as whip,
        pow(rs, {{ pyth_exp }})
            / nullif(pow(rs, {{ pyth_exp }}) + pow(ra, {{ pyth_exp }}), 0) as pyth,
        (bb - so) / nullif(pa, 0)                               as off_bbk
    from pre
)

select
    team_id,
    game_id,
    game_date,
    season,
    games_before,
    -- advanced rate stats are noise below ADV_MIN_GAMES prior games -> null (matches
    -- the Python mask), so downstream model-input comparisons line up exactly.
    case when games_before >= {{ adv_min_games }} then ops     end as ops,
    case when games_before >= {{ adv_min_games }} then era     end as era,
    case when games_before >= {{ adv_min_games }} then fip     end as fip,
    case when games_before >= {{ adv_min_games }} then whip    end as whip,
    case when games_before >= {{ adv_min_games }} then pyth    end as pyth,
    case when games_before >= {{ adv_min_games }} then off_bbk end as off_bbk
from calc
