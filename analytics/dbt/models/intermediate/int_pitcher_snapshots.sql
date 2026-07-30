-- Post-appearance cumulative pitcher stats, one row per (pitcher_id, game_id).
-- SQL re-expression of src/features.py::build_pitcher_snapshots. Cumulative sums are
-- INCLUSIVE of the current appearance (a post-game snapshot); the ASOF join downstream
-- picks the latest snapshot strictly BEFORE a game date, keeping everything pre-game.
-- Shrinkage: _shrunk_fip(num, ip, prior_ip) = (num + (4.20-3.10)*prior_ip)/(ip+prior_ip) + 3.10
--   season-to-date prior_ip = 30  -> +33.0 numerator, +30 denominator
--   last-5-starts   prior_ip = 15  -> +16.5 numerator, +15 denominator
-- (K-BB)/BF prior: (so - bb + 0.14*120) / (bf + 120)
with appearances as (
    select
        pitcher_id,
        season,
        game_id,
        game_date,
        is_start,
        innings_pitched as ip,
        batters_faced,
        strike_outs,
        walks,
        13 * home_runs + 3 * (walks + hit_by_pitch) - 2 * strike_outs as fip_num
    from {{ ref('stg_pitcher_logs') }}
),

cumulative as (
    select
        *,
        (sum(fip_num) over wcum + 33.0) / nullif(sum(ip) over wcum + 30, 0) + 3.10 as sp_fip,
        (sum(strike_outs) over wcum - sum(walks) over wcum + 0.14 * 120)
            / nullif(sum(batters_faced) over wcum + 120, 0)                        as sp_kbb
    from appearances
    window wcum as (
        partition by pitcher_id, season
        order by game_date, game_id
        rows between unbounded preceding and current row
    )
),

-- last-5-starts shrunk FIP, computed over the STARTS subset only
starts_roll as (
    select
        pitcher_id,
        game_id,
        (sum(fip_num) over wstart + 16.5) / nullif(sum(ip) over wstart + 15, 0) + 3.10 as sp_fip5_at_start
    from appearances
    where is_start
    window wstart as (
        partition by pitcher_id, season
        order by game_date, game_id
        rows between 4 preceding and current row
    )
)

select
    c.pitcher_id,
    c.season,
    c.game_id,
    c.game_date                                as snapshot_date,
    c.game_date                                as last_app,
    c.sp_fip,
    c.sp_kbb,
    -- carry the most recent start's last-5 value forward across relief appearances
    -- (Python ffill), then fall back to season-to-date sp_fip before the first start
    coalesce(
        last_value(sr.sp_fip5_at_start ignore nulls) over (
            partition by c.pitcher_id, c.season
            order by c.game_date, c.game_id
            rows between unbounded preceding and current row
        ),
        c.sp_fip
    )                                          as sp_fip5
from cumulative c
left join starts_roll sr
    on c.pitcher_id = sr.pitcher_id and c.game_id = sr.game_id
