---
title: MLB 2026 — Analytics Engineering
description: A modern-data-stack layer (DuckDB + dbt) over the MLB win-probability model.
---

This dashboard is built directly on the **DuckDB warehouse** produced by a **dbt**
project that re-expresses the MLB prediction pipeline's feature engineering as tested
SQL models. Every number below is a query against a dbt mart — no hand-built extracts.

```sql best_model
select model, accuracy, log_loss, n_scored
from mlb.model_performance
where not is_baseline
order by log_loss
limit 1
```

```sql scored
select max(n_scored) as n_scored from mlb.model_performance
```

<BigValue data={best_model} value=model title="Best model (lowest log loss)"/>
<BigValue data={best_model} value=accuracy fmt="0.0%" title="Its out-of-sample accuracy"/>
<BigValue data={scored} value=n_scored title="Games scored in tracker"/>

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

```sql scorecard
select model, is_baseline, n_scored, accuracy, brier, log_loss
from mlb.model_performance
order by is_baseline, log_loss nulls last
```

<DataTable data={scorecard} rows=8>
    <Column id=model/>
    <Column id=n_scored title="Games"/>
    <Column id=accuracy fmt="0.0%"/>
    <Column id=brier fmt="0.000"/>
    <Column id=log_loss title="Log loss" fmt="0.000"/>
</DataTable>

Pages: [Model performance](/model-performance) · [Standings & odds](/standings) ·
[Calibration](/calibration) · [Feature validation](/methodology)
