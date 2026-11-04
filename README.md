# MLB 2026 Playoff Predictor ⚾

**Live site: <https://jarvislee511.github.io/mlb-playoff-predictor-2026/>**

Machine learning pipeline that predicts individual MLB game outcomes and propagates
those predictions into **2026 postseason odds** via Monte Carlo simulation —
FiveThirtyEight-style, built from scratch in Python. A GitHub Actions job retrained
the models and refreshed the site **automatically every morning** of the season.

> **The 2026 season is complete.** All 1,427 scored predictions are final, the site's
> playoff odds are frozen as the pre-playoff forecast, and the postseason results sit next
> to them: the model's favourite, the Milwaukee Brewers (25.8%), won the World Series.
> See [2026 postseason](#2026-postseason).

## Daily automation

Every morning (3:30 AM ET, with a 7:30 AM backup), `.github/workflows/daily.yml` runs
`daily_update.py`. Step 0 is a season gate (`src/season.py`): once every regular-season
game is played and every prediction is scored, the pipeline stops, the site freezes, and
only `docs/data/season.json` (the postseason results) is updated until the World Series
ends. After that the workflow exits in seconds and commits nothing. Setting
`CURRENT_SEASON` in `src/config.py` to the next year re-opens the gate. In season:

1. **Fetch** the latest results from the MLB Stats API.
2. **Score yesterday's predictions** against actual results and append them to
   `data/predictions_log.csv` — predictions are always logged *before* games are
   played, so the site's Model Tracker is a genuine out-of-sample track record.
3. **Retrain** all three win-probability models on the expanded data (continuous learning).
4. **Re-simulate** the season (10,000 runs) for updated playoff odds.
5. **Predict today's games** (with probable starting pitchers) and log them.
6. **Fetch injury-list moves and call-ups** from the transactions API.
7. Export JSON to `docs/data/` and push — GitHub Pages serves the updated site.

## What it does

1. **Data** — pulls 2015–2026 regular-season results (~27,000 games) and the remaining
   2026 schedule from the official [MLB Stats API](https://statsapi.mlb.com), so the
   forecast updates as the real season unfolds.
2. **Elo baseline** — a FiveThirtyEight-style Elo system (K=4, margin-of-victory
   multiplier, 24-point home advantage, 1/3 regression to the mean each offseason).
3. **ML models** — five models compared on pre-game features only:
   Elo difference, last-30-game form, season-to-date win%, rest days,
   **advanced stat differentials** (OPS, ERA, FIP, WHIP, Pythagorean win%, BB−K rate),
   **probable starting pitchers** (shrunk season-to-date FIP / K−BB% / last-5-starts
   form, from 12 seasons of per-pitcher game logs), **bullpen quality + 3-day workload
   fatigue**, and per-park home advantage. Models: Elo baseline, logistic regression,
   XGBoost, an **Elo+LR logistic stack** (fit in logit space on the validation season —
   three parameters, so unlike isotonic regression it cannot overfit a single season),
   and a **Poisson–Skellam** run-distribution model. Every feature is knowable at first
   pitch (as-of joins, no leakage). Time-based split: train 2015–2023, validate 2024,
   test 2025–2026.
4. **Season simulation** — 10,000 Monte Carlo simulations of the remaining 2026 schedule
   and the full 12-team postseason bracket (Wild Card Bo3 → Division Series Bo5 →
   LCS Bo7 → World Series Bo7) produce each team's probability of making the playoffs,
   winning its division, earning a first-round bye, the pennant, and the World Series.
5. **Dashboard** — an interactive Streamlit app with playoff odds, Elo power ratings,
   a model evaluation report (log loss / Brier / calibration), and a head-to-head
   game predictor.

## Why a simulation instead of classifying "playoff team: yes/no"?

Game-level prediction is where the signal is; whether a team makes the playoffs is a
*consequence* of ~80 remaining coin flips with team-specific weights. Simulating the
schedule propagates the uncertainty in each game forward to season-end probabilities,
and lets one model answer many questions (division odds, byes, World Series) at once.

## Results

Two independent out-of-sample evaluations, and they agree: **the ML features do not beat
the Elo baseline by more than noise.** That is the result, not a bug — and the live site
[says so on its own Model Tracker tab](https://jarvislee511.github.io/mlb-playoff-predictor-2026/)
rather than presenting a leaderboard that implies otherwise.

**Backtest** — 4,859 held-out games (2025 + played 2026, never seen in training;
`outputs/metrics.json`):

| Model | Log loss | Brier | AUC | Accuracy |
|---|---|---|---|---|
| Elo baseline | 0.6814 | 0.2442 | 0.571 | 56.2% |
| Logistic regression | **0.6797** | 0.2434 | 0.578 | 55.8% |
| XGBoost | 0.6818 | 0.2444 | 0.573 | 56.2% |
| Elo+LR stack | 0.6800 | 0.2435 | 0.578 | 56.2% |
| Poisson–Skellam | 0.6849 | 0.2459 | 0.576 | 55.6% |

**Live tracker, final** — the whole 2026 regular season: every game predicted *before
first pitch* and scored the next morning, restricted to the 1,412 games every model
called, with a paired bootstrap (5,000 resamples) on each model's log-loss gap to Elo:

| Model | Log loss | Accuracy | Δ vs Elo | 95% CI |
|---|---|---|---|---|
| Elo baseline | 0.6800 | 56.4% | — | — |
| Logistic regression | **0.6772** | 56.5% | −0.0028 | [−0.0070, +0.0013] |
| Elo+LR stack | 0.6774 | 57.0% | −0.0026 | [−0.0056, +0.0003] |
| XGBoost | 0.6797 | 56.8% | −0.0003 | [−0.0043, +0.0039] |
| Poisson–Skellam | 0.6806 | 56.9% | +0.0007 | [−0.0062, +0.0075] |

Every interval still spans zero, the stack's only just. The best model is 0.0028 log loss
ahead of a rating system with no fitted features at all, and a full season of games is not
enough to call that gap real.

**Why that is the expected answer.** Elo already integrates team strength from every game
ever played, and the features layered on top (season-to-date rate stats, 30-game form,
probable-starter quality, bullpen fatigue, rest, park) are largely *functions of the same
history*. Public pre-game information is close to exhausted; what remains is in-game state,
pitch-level detail, and day-of bullpen availability. Reporting a 0.001 improvement as a win
would be the mistake.

**Read log loss, not accuracy.** Home teams won 52.9% of those games. The models' own
probabilities implied 56–57% expected accuracy and they realized 56–57%, with a ±2.6-point
95% margin: well calibrated, and still not separable from each other on accuracy.
Calibration curves are on the site.

> These are the final figures for the 2026 regular season. The conclusion, that the gap
> between the models is smaller than the sampling noise, did not change as the sample grew
> from the 650 games this README first reported to the full season.

## 2026 postseason

The model does not predict postseason games; its last regular-season simulation (after the
final game on 2026-09-27) is the pre-playoff forecast, frozen on the site's Odds tab with
each team's actual result beside it. The model's favourite, the **Milwaukee Brewers**
(25.8% to win the World Series, #1 of 12), beat the Tampa Bay Rays 4–1 in the World
Series; both pennant winners were its favourites in their leagues. Picking at random from
the 12-team field would give 8.3%. One postseason is a single draw, so this is a record,
not a grade: the tracker above is where the model is graded.

### Ablations, kept in the record

- **Starting-lineup wOBA** — the full pipeline was built specifically to test it (starting
  lineups + 12 seasons of per-batter game logs, leak-free as-of snapshots). It moved LR
  test log loss by **+0.00004**: nothing. Lineup-average wOBA is collinear with the team
  season OPS already in the model, and a deviation-from-normal variant (meant to isolate a
  rested star) was flat with a wrong-signed coefficient. Dropped from `FEATURE_COLS`
  instead of kept for show; `python -m experiments.ablation_lineup` reruns the comparison
  (it tells you what to rebuild first, since the 43 MB of lineup data is not committed).

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

python run_pipeline.py           # fetch + train + simulate (~5 min)
streamlit run app/streamlit_app.py
```

`python run_pipeline.py --no-fetch` reuses cached data and skips the API calls.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q          # 96 tests, ~3 s, no network and no data files
```

They guard the claims the rest of this README makes: the Elo scale and its
zero-sum rating transfer, that every rolling stat and both as-of joins exclude the
current game, that the hand-rolled Skellam matches a brute-force Poisson double
sum to 1e-9, the delta-vs-Elo sign convention behind the results table, and the
postseason bracket's structure. `daily.yml` runs them **before** the pipeline.

The suite was itself checked by mutation — plant a plausible bug, confirm the
tests go red — and catches **12 of 12** planted bugs. That exercise found a real
fragility in `bullpen_fatigue` and added two tests that were missing. See
[`tests/README.md`](tests/README.md).

## Project structure

```
├── run_pipeline.py          # one-shot pipeline (fetch -> train -> simulate)
├── daily_update.py          # daily automation entry point (run by CI)
├── .github/workflows/daily.yml  # scheduled retrain + site refresh
├── src/
│   ├── config.py            # paths, Elo + simulation constants
│   ├── data/fetch.py        # MLB Stats API ingestion (retry/backoff)
│   ├── features.py          # leak-free pre-game feature engineering
│   ├── pipeline.py          # reusable pipeline stages
│   ├── predictions.py       # daily game predictions + scoring log
│   ├── transactions.py      # injuries / call-ups / roster moves
│   ├── site_export.py       # JSON export for the static site
│   ├── models/elo.py        # Elo rating system
│   ├── models/train.py      # LR + XGBoost training & evaluation
│   └── simulate.py          # Monte Carlo season + postseason simulation
├── docs/                    # static website (GitHub Pages) + its data JSONs
├── data/predictions_log.csv # permanent out-of-sample prediction record
├── app/streamlit_app.py     # local interactive dashboard
├── analytics/               # dbt + DuckDB + Dagster layer (see below)
├── experiments/             # ablations kept as a record of negative results
└── outputs/                 # odds, metrics, calibration, current Elo
```

## Analytics engineering layer (`analytics/`)

A modern-data-stack layer built **on top of** this model without modifying it: a
**dbt + DuckDB** warehouse re-expresses the pandas feature engineering as tested SQL
(window functions + ASOF joins), with **Dagster** orchestration and two BI front-ends
(**Evidence.dev** and **Tableau Public**).

The centrepiece is a validation harness — `mart_feature_validation` diffs every
SQL-rebuilt feature against the Python `features.csv`: the season-to-date rate stats,
pitcher snapshots, park HFA, season win% and rest days match to floating-point precision,
and 30-game rolling form matches ~99.9%. In CI, `daily.yml` runs `dbt build` (all models
+ tests) on the freshly regenerated data after each daily prediction commit.

**🌐 Live BI site: <https://jarvislee511.github.io/mlb-playoff-predictor-2026/analytics/>** —
five Evidence pages served from the same Pages site, rebuilt weekly by
[`analytics-site.yml`](.github/workflows/analytics-site.yml). The
[feature-validation page](https://jarvislee511.github.io/mlb-playoff-predictor-2026/analytics/methodology)
is the one to look at: per-feature max/mean absolute difference between the SQL rebuild and
the pandas original, most of them at 1e-15 to 1e-18.

See **[`analytics/README.md`](analytics/README.md)** for the full walkthrough.

## Modeling notes & limitations

- **No data leakage**: every feature is computable before first pitch; rolling stats
  are updated only after each game is recorded.
- Seeding ties are broken randomly rather than by MLB's head-to-head tiebreakers;
  over 10,000 simulations the effect on odds is negligible.
- The simulation uses Elo probabilities (future games have unknown rolling stats);
  the ML models power game-level evaluation and the matchup predictor.
- **Probable starting pitchers *are* modeled** (shrunk season-to-date FIP and K−BB%, plus
  last-5-starts form, from 12 seasons of per-pitcher game logs) and so is bullpen quality
  with 3-day workload fatigue. What is *not* in the model: pitch-level / Statcast detail,
  day-of bullpen availability, injuries and trades. Injuries and roster moves are fetched
  and shown on the site as context, but they are not features — the largest known gap and
  the natural next iteration.
- The shortened 2020 season is kept for Elo continuity but its small sample is
  handled by the cross-season rolling windows.

## Data source

All data from the official MLB Stats API via the
[MLB-StatsAPI](https://github.com/toddrob99/MLB-StatsAPI) Python wrapper.
This project is for educational/portfolio purposes and is not affiliated with MLB.
