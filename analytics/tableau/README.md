# Tableau Public workbook

A recruiter-searchable Tableau Public dashboard built on the same dbt marts as the
Evidence site. Tableau Public can't connect to DuckDB or auto-refresh, so this folder
holds flat CSV extracts plus the build steps.

## 1. Regenerate the extracts

```bash
# from repo root, after `dbt build`
cd analytics/tableau
../../.venv/Scripts/python.exe export_tableau.py
```

Produces `data/`:

| File | Grain | Use for |
|---|---|---|
| `standings.csv` | one row per team | playoff-odds bars, Elo vs projected-wins scatter, division small-multiples |
| `model_performance.csv` | one row per model (+ baselines) | log-loss / accuracy bars vs the coin-flip and always-home references |
| `calibration.csv` | one row per model × bin | reliability curve (predicted vs actual) |
| `predictions_scored.csv` | one row per scored game × model (long) | accuracy-over-time line, a `correct` calc field |
| `feature_validation.csv` | one row per rebuilt feature | "SQL rebuild matches Python" proof table |

## 2. Build the workbook (Tableau Public Desktop, free)

1. **Connect** → Text file → add all five CSVs (no joins needed; use as separate data sources).
2. Suggested sheets:
   - **Championship odds** — `standings`: bar of `p_win_world_series` by `team_name`, sorted desc, color by `league`.
   - **Elo vs projected wins** — `standings`: scatter `elo` × `projected_wins`, color `league`, label top teams.
   - **Model scorecard** — `model_performance`: bars of `log_loss` and `accuracy`; add a reference line at `log_loss = 0.693` (coin flip).
   - **Reliability** — `calibration`: `mean_predicted` (x) vs `fraction_won` (y) by `model`, with a 45° reference line.
   - **Accuracy over time** — `predictions_scored`: calc field
     `correct = IF (([p_home] > 0.5) = ([home_win] = 1)) THEN 1 ELSE 0 END`,
     plot running `AVG(correct)` by `game_date`, filter to one `prob_col`.
3. Assemble a **dashboard** ("MLB 2026 — Model & Playoff Analytics"); add a title and a
   one-line note that the data comes from a dbt/DuckDB warehouse.

## 3. Publish

`Server → Tableau Public → Save to Tableau Public As…` (needs a free Tableau Public
account). Copy the resulting URL into the résumé and the project README.

> Refresh cadence: manual. Re-run step 1 and re-upload when you want the public workbook
> to reflect the latest tracker. The Evidence site (in `../evidence`) is the
> auto-updating counterpart that ships with CI.
