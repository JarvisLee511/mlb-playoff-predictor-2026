---
title: Feature Validation
---

The whole point of this layer: the model's feature engineering — originally hand-written
in pandas (`src/features.py`) — was **re-expressed as SQL** in dbt using window functions
and ASOF joins. `mart_feature_validation` proves the SQL rebuild matches the Python
pipeline row-for-row, by diffing every rebuilt feature against `data/processed/features.csv`.

```sql validation
select feature, n_compared, max_abs_diff, mean_abs_diff, pct_within_tol
from mlb.feature_validation
order by max_abs_diff desc
```

```sql headline
select
    count(*) as n_features,
    min(pct_within_tol) as worst_agreement,
    max(max_abs_diff) as worst_abs_diff
from mlb.feature_validation
```

<BigValue data={headline} value=n_features title="Features validated"/>
<BigValue data={headline} value=worst_agreement fmt="0.00%" title="Worst per-feature agreement"/>
<BigValue data={headline} value=worst_abs_diff fmt="0.000" title="Worst absolute difference"/>

## Per-feature agreement

Every season-to-date rate stat (OPS, ERA, FIP, WHIP, Pythagorean, BB−K%) plus the
starting-pitcher snapshots, park HFA, season win% and rest days match the Python pipeline
to floating-point precision. The only residual is on the 30-game rolling win% / run
differential (~99.9%), from a handful of window-edge games around doubleheaders.

<DataTable data={validation}>
    <Column id=feature/>
    <Column id=n_compared title="Rows compared"/>
    <Column id=max_abs_diff title="Max abs diff" fmt=sci/>
    <Column id=mean_abs_diff title="Mean abs diff" fmt=sci/>
    <Column id=pct_within_tol title="% within 1e-6" fmt="0.00%"/>
</DataTable>

## How it's built

The pipeline files (`data`, `outputs`) are read in place as **dbt sources**, then flow
through three layers:

- **staging** — typed, renamed views (`stg_*`) over each source file.
- **intermediate** — the feature reimplementation:
  - `int_team_season_to_date` — season-to-date rate stats via `SUM(...) OVER (... ROWS ...)` windows.
  - `int_team_rolling_form` — the 30-game deque re-expressed as a rolling window frame.
  - `int_pitcher_snapshots` — cumulative shrunk FIP / K−BB / last-5-starts, ready for an ASOF join.
- **marts** — `mart_team_game_features` (ASOF-joins the pitcher snapshots), plus
  `mart_feature_validation` (this page), `mart_model_performance`, `mart_calibration`
  and `mart_standings`.

