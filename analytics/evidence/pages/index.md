---
title: MLB 2026 — Analytics Engineering
description: A modern-data-stack layer (DuckDB + dbt) over the MLB win-probability model.
---

This dashboard is built directly on the **DuckDB warehouse** produced by a **dbt**
project that re-expresses the MLB prediction pipeline's feature engineering as tested
SQL models. Every number below is a query against a dbt mart — no hand-built extracts.

```sql best_model
select model, accuracy, log_loss, n_scored, delta_log_loss_vs_elo
from mlb.model_performance
where not is_baseline
order by log_loss
limit 1
```

```sql scored
select max(n_available) as n_available, max(n_scored) as n_scored
from mlb.model_performance
```

<BigValue data={best_model} value=model title="Best model (lowest log loss)"/>
<BigValue data={best_model} value=delta_log_loss_vs_elo fmt="0.0000" title="Its gap to Elo (log loss)"/>
<BigValue data={scored} value=n_available title="Games scored in tracker"/>

That gap is **{fmt(best_model[0].delta_log_loss_vs_elo, '0.0000')} log loss against a rating
system with no fitted features at all** (negative = better) — smaller than the sampling noise on this many
games. The [prediction site](https://jarvislee511.github.io/mlb-playoff-predictor-2026/)
puts a paired-bootstrap interval on that gap, and every interval still spans zero.
Reporting it as a win would be the mistake, so it is reported as what it is.

## World Series odds

Monte-Carlo (10,000-season simulation) championship probability by team.

```sql top_ws
select team_name, abbrev, league, p_win_world_series
from mlb.standings
order by p_win_world_series desc
limit 12
```

<BarChart
    data={top_ws}
    x=team_name
    y=p_win_world_series
    swapXY=true
    yFmt="0.0%"
    title="Championship probability — top 12"
/>

## Model scorecard

How each model performs on games it predicted **before** first pitch, against the
always-pick-home floor and the coin-flip reference.

Every row is scored on the **same** {scored[0].n_scored} games — the ones every model
called. Models added later in the season have no prediction for earlier games, so ranking
each on its own non-null rows would compare different samples. `Available` shows that
coverage difference rather than hiding it.

```sql scorecard
select model, is_baseline, n_scored, n_available, accuracy, brier, log_loss,
       delta_log_loss_vs_elo
from mlb.model_performance
order by is_baseline, log_loss nulls last
```

<DataTable data={scorecard} rows=8>
    <Column id=model/>
    <Column id=n_scored title="Scored on"/>
    <Column id=n_available title="Available"/>
    <Column id=accuracy fmt="0.0%"/>
    <Column id=brier fmt="0.000"/>
    <Column id=log_loss title="Log loss" fmt="0.000"/>
    <Column id=delta_log_loss_vs_elo title="Δ vs Elo" fmt="0.0000"/>
</DataTable>

Pages: [Model performance](/model-performance) · [Standings & odds](/standings) ·
[Calibration](/calibration) · [Feature validation](/methodology)
