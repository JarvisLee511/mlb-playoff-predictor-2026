---
title: Model Performance
---

Out-of-sample results from the live prediction tracker (`data/predictions_log.csv`),
aggregated in the `mart_model_performance` dbt model. Each model's probability is logged
**before** first pitch, then scored once the game is final.

```sql models
select model, n_scored, accuracy, brier, log_loss
from mlb.model_performance
where not is_baseline
order by log_loss
```

```sql floor
select accuracy as always_home from mlb.model_performance where model = 'baseline_always_home'
```

```sql ceiling
select log_loss as coinflip_ll from mlb.model_performance where model = 'baseline_coinflip'
```

<BigValue data={floor} value=always_home fmt="0.0%" title="Always-pick-home accuracy (floor)"/>
<BigValue data={ceiling} value=coinflip_ll fmt="0.000" title="Coin-flip log loss (ln 2)"/>

## Log loss by model

Lower is better. The coin-flip reference is ln(2) ≈ 0.693.

<BarChart data={models} x=model y=log_loss yFmt="0.000" title="Log loss (lower = better)">
    <ReferenceLine y={0.6931} label="coin flip" color=negative/>
</BarChart>

## Accuracy by model

<BarChart data={models} x=model y=accuracy yFmt="0.0%" title="Out-of-sample accuracy"/>

## Full table

<DataTable data={models}>
    <Column id=model/>
    <Column id=n_scored title="Games"/>
    <Column id=accuracy fmt="0.0%"/>
    <Column id=brier fmt="0.000"/>
    <Column id=log_loss title="Log loss" fmt="0.000"/>
</DataTable>

> Single-game MLB prediction has a low ceiling — even Vegas closing lines land near
> 58–60% accuracy. Read these against the always-home floor and coin-flip reference,
> not in isolation.
