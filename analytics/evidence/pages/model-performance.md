---
title: Model Performance
---

Out-of-sample results from the live prediction tracker (`data/predictions_log.csv`),
aggregated in the `mart_model_performance` dbt model. Each model's probability is logged
**before** first pitch, then scored once the game is final.

All models are scored on the same games — the ones every model called — so the ranking
below compares like with like. See the [scorecard note](/) for why that matters.

```sql models
select model, n_scored, n_available, accuracy, brier, log_loss, delta_log_loss_vs_elo
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
    <Column id=n_scored title="Scored on"/>
    <Column id=n_available title="Available"/>
    <Column id=accuracy fmt="0.0%"/>
    <Column id=brier fmt="0.000"/>
    <Column id=log_loss title="Log loss" fmt="0.000"/>
    <Column id=delta_log_loss_vs_elo title="Δ vs Elo" fmt="0.0000"/>
</DataTable>

> Single-game MLB prediction has a low ceiling — even Vegas closing lines land near
> 58–60% accuracy. Read these against the always-home floor and coin-flip reference,
> not in isolation.
>
> The Δ column is the point that matters: the fitted models sit within a thousandth of a
> log loss of the unfitted Elo baseline, i.e. inside the noise. The ML features are not
> adding measurable signal on top of Elo, and that is the finding.
