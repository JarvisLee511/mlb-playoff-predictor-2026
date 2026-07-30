-- Backtest metrics live in a nested JSON object ({model: {metrics}, best_model: ...}).
-- read_json_auto lands it as one row of struct columns; we unpivot to one row per model.
with raw as (
    select * from read_json_auto('{{ var("mlb_root") }}/outputs/metrics.json')
),
tidy as (
    select 'elo_baseline'        as model, elo_baseline        as s from raw
    union all select 'logistic_regression', logistic_regression      from raw
    union all select 'xgboost',             xgboost                  from raw
    union all select 'ensemble',            ensemble                 from raw
    union all select 'skellam_poisson',     skellam_poisson          from raw
)
select
    t.model,
    cast(t.s['log_loss'] as double) as log_loss,
    cast(t.s['brier'] as double)    as brier,
    cast(t.s['auc'] as double)      as auc,
    cast(t.s['accuracy'] as double) as accuracy,
    cast(t.s['n_games'] as integer) as n_games,
    (t.model = (select best_model from raw)) as is_best_model
from tidy t
