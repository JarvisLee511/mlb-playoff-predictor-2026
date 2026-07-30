-- Pre-game feature table, one row per game_id — the SQL reimplementation of the
-- model's FEATURE_COLS. Box-score-derived features (rolling form, season-to-date rate
-- stats, starting-pitcher snapshots, park HFA) are rebuilt here in pure SQL from the
-- intermediate window/ASOF models. Elo (stateful rating engine) and bullpen snapshots
-- are consumed from stg_features as-is. mart_feature_validation checks the rebuilt
-- columns against the Python features.csv.
{% set park_prior_n = 200 %}
{% set park_prior_rate = 0.54 %}

with games as (
    select game_id, season, game_date, home_id, away_id, home_name, away_name, status
    from {{ ref('stg_games') }}
),

park as (
    select
        game_id,
        (coalesce(sum(home_win) over w, 0) + {{ park_prior_rate }} * {{ park_prior_n }})
            / (coalesce(count(home_win) over w, 0) + {{ park_prior_n }}) as park_hfa
    from {{ ref('stg_games') }}
    window w as (
        partition by home_id
        order by game_date, game_id
        rows between unbounded preceding and 1 preceding
    )
),

form as (select * from {{ ref('int_team_rolling_form') }}),
adv  as (select * from {{ ref('int_team_season_to_date') }}),
prob as (select * from {{ ref('stg_probables') }}),
snaps as (select * from {{ ref('int_pitcher_snapshots') }}),
ref  as (select * from {{ ref('stg_features') }}),

joined as (
    select
        g.game_id, g.season, g.game_date, g.status,
        g.home_id, g.home_name, g.away_id, g.away_name,
        p.park_hfa,

        -- rolling form
        hf.winpct_30 as home_winpct_30, af.winpct_30 as away_winpct_30,
        hf.rundiff_30 as home_rundiff_30, af.rundiff_30 as away_rundiff_30,
        hf.winpct_10 as home_winpct_10, af.winpct_10 as away_winpct_10,
        hf.season_winpct as home_season_winpct, af.season_winpct as away_season_winpct,
        hf.rest_days as home_rest_days, af.rest_days as away_rest_days,

        -- season-to-date advanced rate stats
        ha.ops as home_ops, aa.ops as away_ops,
        ha.era as home_era, aa.era as away_era,
        ha.fip as home_fip, aa.fip as away_fip,
        ha.whip as home_whip, aa.whip as away_whip,
        ha.pyth as home_pyth, aa.pyth as away_pyth,
        ha.off_bbk as home_off_bbk, aa.off_bbk as away_off_bbk,

        -- starting pitcher snapshots (ASOF: latest snapshot strictly before game date)
        hp.sp_fip as home_sp_fip, ap.sp_fip as away_sp_fip,
        hp.sp_kbb as home_sp_kbb, ap.sp_kbb as away_sp_kbb,
        hp.sp_fip5 as home_sp_fip5, ap.sp_fip5 as away_sp_fip5,

        -- consumed from the Python engines (not re-expressed in SQL)
        r.elo_diff, r.elo_prob_home, r.bp_fip_diff, r.bp_fatigue_diff
    from games g
    left join park p on g.game_id = p.game_id
    left join form hf on hf.team_id = g.home_id and hf.game_id = g.game_id
    left join form af on af.team_id = g.away_id and af.game_id = g.game_id
    left join adv  ha on ha.team_id = g.home_id and ha.game_id = g.game_id
    left join adv  aa on aa.team_id = g.away_id and aa.game_id = g.game_id
    left join prob pr on pr.game_id = g.game_id
    asof left join snaps hp
        on hp.pitcher_id = pr.home_sp_id and hp.season = g.season and hp.snapshot_date < g.game_date
    asof left join snaps ap
        on ap.pitcher_id = pr.away_sp_id and ap.season = g.season and ap.snapshot_date < g.game_date
    left join ref r on r.game_id = g.game_id
)

select
    *,
    -- FEATURE_COLS differentials (home minus away). Advanced/pitcher diffs default to 0
    -- when a component is null, matching the Python .fillna(0).
    home_winpct_30 - away_winpct_30                as winpct30_diff,
    home_rundiff_30 - away_rundiff_30              as rundiff30_diff,
    home_season_winpct - away_season_winpct        as season_winpct_diff,
    home_rest_days - away_rest_days                as rest_diff,
    home_winpct_10 - away_winpct_10                as winpct10_diff,
    coalesce(home_ops - away_ops, 0)               as ops_diff,
    coalesce(home_era - away_era, 0)               as era_diff,
    coalesce(home_fip - away_fip, 0)               as fip_diff,
    coalesce(home_whip - away_whip, 0)             as whip_diff,
    coalesce(home_pyth - away_pyth, 0)             as pyth_diff,
    coalesce(home_off_bbk - away_off_bbk, 0)       as off_bbk_diff,
    coalesce(home_sp_fip - away_sp_fip, 0)         as sp_fip_diff,
    coalesce(home_sp_kbb - away_sp_kbb, 0)         as sp_kbb_diff,
    coalesce(home_sp_fip5 - away_sp_fip5, 0)       as sp_fip5_diff
from joined
