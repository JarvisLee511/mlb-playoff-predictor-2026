-- Proves the SQL feature reimplementation matches the Python pipeline. For each
-- rebuilt column it compares mart_team_game_features against the reference values from
-- data/processed/features.csv (stg_features), over games present in both, and reports
-- the worst absolute difference plus the share of rows within a 1e-6 tolerance.
-- A healthy build shows max_abs_diff ~ 0 and pct_within_tol = 1.0 for every feature.
{% set tol = 0.000001 %}
{% set pairs = [
    ('park_hfa',           'ref_park_hfa'),
    ('home_winpct_30',     'ref_home_winpct_30'),
    ('home_rundiff_30',    'ref_home_rundiff_30'),
    ('home_season_winpct', 'ref_home_season_winpct'),
    ('home_rest_days',     'ref_home_rest_days'),
    ('home_ops',           'ref_home_ops'),
    ('home_era',           'ref_home_era'),
    ('home_fip',           'ref_home_fip'),
    ('home_whip',          'ref_home_whip'),
    ('home_pyth',          'ref_home_pyth'),
    ('home_off_bbk',       'ref_home_off_bbk'),
    ('home_sp_fip',        'ref_home_sp_fip'),
    ('home_sp_kbb',        'ref_home_sp_kbb'),
    ('home_sp_fip5',       'ref_home_sp_fip5')
] %}

with compared as (
    select
        f.game_id,
        f.park_hfa, f.home_winpct_30, f.home_rundiff_30, f.home_season_winpct,
        f.home_rest_days, f.home_ops, f.home_era, f.home_fip, f.home_whip,
        f.home_pyth, f.home_off_bbk, f.home_sp_fip, f.home_sp_kbb, f.home_sp_fip5,
        r.ref_park_hfa, r.ref_home_winpct_30, r.ref_home_rundiff_30, r.ref_home_season_winpct,
        r.ref_home_rest_days, r.ref_home_ops, r.ref_home_era, r.ref_home_fip, r.ref_home_whip,
        r.ref_home_pyth, r.ref_home_off_bbk, r.ref_home_sp_fip, r.ref_home_sp_kbb, r.ref_home_sp_fip5
    from {{ ref('mart_team_game_features') }} f
    inner join {{ ref('stg_features') }} r using (game_id)
)

{% for rebuilt, reference in pairs %}
select
    '{{ rebuilt }}'                                      as feature,
    count(*) filter (where {{ reference }} is not null and {{ rebuilt }} is not null) as n_compared,
    max(abs({{ rebuilt }} - {{ reference }}))            as max_abs_diff,
    avg(abs({{ rebuilt }} - {{ reference }}))            as mean_abs_diff,
    count(*) filter (
        where {{ reference }} is not null and {{ rebuilt }} is not null
          and abs({{ rebuilt }} - {{ reference }}) <= {{ tol }}
    )::double
    / nullif(count(*) filter (where {{ reference }} is not null and {{ rebuilt }} is not null), 0)
                                                         as pct_within_tol
from compared
{% if not loop.last %}union all{% endif %}
{% endfor %}
