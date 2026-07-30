-- Out-of-sample scorecard: one row per model over the games actually scored in the
-- live tracker, plus two reference baselines. This is the headline BI table — it shows
-- how each model performs against the always-pick-home floor and the coin-flip ceiling.
{% set eps = 0.000001 %}

with finals as (
    select game_id, game_date, home_win,
           p_home_elo, p_home_lr, p_home_xgb, p_home_ens, p_home_skl
    from {{ ref('stg_predictions_log') }}
    where is_final and home_win is not null
),

long as (
    unpivot finals
    on p_home_elo, p_home_lr, p_home_xgb, p_home_ens, p_home_skl
    into name prob_col value p
),

per_model as (
    select
        case prob_col
            when 'p_home_elo' then 'elo'
            when 'p_home_lr'  then 'lr'
            when 'p_home_xgb' then 'xgb'
            when 'p_home_ens' then 'ens'
            when 'p_home_skl' then 'skl'
        end                                                        as model,
        false                                                      as is_baseline,
        count(*)                                                   as n_scored,
        avg(case when (p > 0.5) = (home_win = 1) then 1.0 else 0.0 end) as accuracy,
        avg(pow(p - home_win, 2))                                  as brier,
        avg(-(home_win * ln(greatest(least(p, 1 - {{ eps }}), {{ eps }}))
              + (1 - home_win) * ln(1 - greatest(least(p, 1 - {{ eps }}), {{ eps }})))) as log_loss
    from long
    where p is not null
    group by 1
),

baselines as (
    -- always pick the home team: accuracy = share of home wins
    select 'baseline_always_home' as model, true as is_baseline,
           count(*) as n_scored, avg(home_win::double) as accuracy,
           null::double as brier, null::double as log_loss
    from finals
    union all
    -- coin flip: 0.5 every game -> brier 0.25, log loss ln(2)
    select 'baseline_coinflip', true,
           (select count(*) from finals), 0.5, 0.25, ln(2.0)
)

select
    model, is_baseline, n_scored,
    round(accuracy, 4)  as accuracy,
    round(brier, 4)     as brier,
    round(log_loss, 4)  as log_loss
from per_model
union all
select model, is_baseline, n_scored,
       round(accuracy, 4), round(brier, 4), round(log_loss, 4)
from baselines
order by is_baseline, log_loss nulls last
