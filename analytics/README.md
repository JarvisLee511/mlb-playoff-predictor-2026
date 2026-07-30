# Analytics Engineering Layer

A modern-data-stack layer built **on top of** the MLB win-probability model, without
touching it. The Python pipeline stays the system of record; this layer reads the same
files it writes and re-expresses the feature engineering and tracking as a tested,
documented, orchestrated SQL warehouse — the toolchain analytics-engineering roles run on.

```text
 Python pipeline (src/, daily_update.py)         data/*.csv · data/raw · data/processed · outputs/*
        │  writes                                              │  read in place (no copy)
        ▼                                                       ▼
 ┌───────────────────────────────────────── dbt (dbt-duckdb) ─────────────────────────────────┐
 │  staging  →  intermediate (window fns, ASOF joins)  →  marts  +  tests  +  docs (lineage)   │
 └────────────────────────────────────────────────┬───────────────────────────────────────────┘
                                                   ▼
                        DuckDB warehouse (warehouse/mlb.duckdb)
                                    │                         │
                         Evidence.dev (BI-as-code)      Tableau Public (extracts)
                                    │
                  Dagster asset graph orchestrates pipeline → dbt end-to-end
```

## What's here

| Path | Tool | What it is |
|---|---|---|
| `dbt/` | dbt-duckdb | staging → intermediate → marts, tests, source freshness, docs |
| `orchestration/` | Dagster | one asset graph: `mlb_prediction_pipeline` → dbt sources → models |
| `evidence/` | Evidence.dev | BI-as-code site (SQL + markdown → static site) |
| `tableau/` | Tableau Public | CSV export script + workbook build/publish guide |
| `warehouse/` | DuckDB | `mlb.duckdb`, rebuilt by `dbt build` (git-ignored) |

## The headline: SQL feature parity

`src/features.py` builds the model's inputs in pandas (rolling deques, season-to-date
rate stats, `merge_asof` on pitcher snapshots). The dbt `intermediate/` models rebuild
those in pure SQL — `SUM(...) OVER (... ROWS ...)` windows and DuckDB `ASOF JOIN`s — and
`mart_feature_validation` diffs every rebuilt column against `features.csv`:

- season-to-date OPS/ERA/FIP/WHIP/Pythagorean/BB−K%, starter snapshots, park HFA, season
  win% and rest days: **match to floating-point precision** (≤ 1e-15)
- 30-game rolling win% / run-diff: **~99.9%** (residual is a window-edge game around
  doubleheaders)

Enforced by the singular test `assert_feature_reimplementation_matches`.

## Run it locally

Prereqs: the project `.venv` with `dbt-duckdb`, `dagster`, `dagster-dbt` installed, and
Node 18+ for Evidence. All dbt commands run from `analytics/dbt`.

```bash
# 1. Build + test the warehouse (regenerates warehouse/mlb.duckdb)
cd analytics/dbt
dbt build --profiles-dir .
dbt docs generate --profiles-dir . && dbt docs serve --profiles-dir .   # lineage graph

# 2. Explore the orchestration graph
cd ../orchestration
../../.venv/Scripts/dagster.exe dev -f definitions.py                    # http://localhost:3000

# 3. Run the BI site
cd ../evidence
npm install
npm run sources        # extract marts from DuckDB
npm run dev            # http://localhost:3000

# 4. Refresh Tableau extracts
cd ../tableau
../../.venv/Scripts/python.exe export_tableau.py
```

## CI

- **`.github/workflows/daily.yml`** — after the daily prediction commit, installs
  `requirements-analytics.txt` and runs `dbt build` (all models + tests) on the freshly
  regenerated data. Placed after the commit so an analytics failure never blocks the
  prediction push.
- **`.github/workflows/analytics-site.yml`** — `workflow_dispatch` + weekly; rebuilds the
  warehouse and the Evidence site and publishes it to `docs/analytics/`.

## Design notes

- **Parallel, not invasive.** dbt reads files; it never writes back into the model's
  inputs. Deleting `analytics/` leaves the predictor untouched.
- **Elo and bullpen** features are consumed from `features.csv` as-is (stateful rating
  engine / separate snapshot builder), not re-expressed in SQL — see the comment in
  `mart_team_game_features.sql`.
- **Paths** resolve through the `mlb_root` dbt var (default `../..`), so dbt must be run
  from `analytics/dbt` (or pass `--vars '{mlb_root: <abs repo root>}'`).
