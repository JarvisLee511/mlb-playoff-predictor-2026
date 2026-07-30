-- Out-of-sample scorecard: one row per model plus two reference baselines. This is the
-- headline BI table — it shows how each model performs against the always-pick-home floor
-- and the coin-flip ceiling.
--
-- Every metric is computed on the games *all* models predicted. Models added later in the
-- season have no prediction for earlier games, so scoring each one over its own non-null
-- rows and then ordering by log loss would rank them on different samples. n_available
-- keeps that coverage difference visible instead of hiding it.
{% set eps = 0.000001 %}

with finals as (
    select game_id, game_date, home_win,
           p_home_elo, p_home_lr, p_home_xgb, p_home_ens, p_home_skl
    from {{ ref('stg_predictions_log') }}
    where is_final and home_win is not null
),

-- the intersection: every model called these games
common as (
    select * from finals
    where p_home_elo is not null
      and p_home_lr  is not null
      and p_home_xgb is not null
      and p_home_ens is not null
      and p_home_skl is not null
),

-- per-model coverage over the full log, reported next to the comparable metrics
coverage as (
    select
        case prob_col
            when 'p_home_elo' then 'elo'
            when 'p_home_lr'  then 'lr'
            when 'p_home_xgb' then 'xgb'
            when 'p_home_ens' then 'ens'
            when 'p_home_skl' then 'skl'
        end        as model,
        count(*)   as n_available
    from (
        unpivot finals
        on p_home_elo, p_home_lr, p_home_xgb, p_home_ens, p_home_skl
        into name prob_col value p
    )
    where p is not null
    group by 1
),

long as (
    unpivot common
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
    group by 1
),

baselines as (
    -- always pick the home team: accuracy = share of home wins
    select 'baseline_always_home' as model, true as is_baseline,
           count(*) as n_scored, avg(home_win::double) as accuracy,
           null::double as brier, null::double as log_loss
    from common
    union all
    -- coin flip: 0.5 every game -> brier 0.25, log loss ln(2)
    select 'baseline_coinflip', true,
           (select count(*) from common), 0.5, 0.25, ln(2.0)
),

elo_log_loss as (select log_loss from per_model where model = 'elo')

select
    m.model,
    m.is_baseline,
    m.n_scored,
    c.n_available,
    round(m.accuracy, 4) as accuracy,
    round(m.brier, 4)    as brier,
    round(m.log_loss, 4) as log_loss,
    -- negative = better than the unfitted Elo baseline
    case when m.model = 'elo' then null
         else round(m.log_loss - (select log_loss from elo_log_loss), 5)
    end as delta_log_loss_vs_elo
from per_model m
left join coverage c using (model)

union all

select
    b.model, b.is_baseline, b.n_scored, b.n_scored as n_available,
    round(b.accuracy, 4), round(b.brier, 4), round(b.log_loss, 4), null
from baselines b

order by is_baseline, log_loss nulls last
