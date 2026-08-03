# MLB 2026 Playoff Predictor ⚾

**Live site: <https://jarvislee511.github.io/mlb-playoff-predictor-2026/>**

Machine learning pipeline that predicts individual MLB game outcomes and propagates
those predictions into **2026 postseason odds** via Monte Carlo simulation —
FiveThirtyEight-style, built from scratch in Python. A GitHub Actions job retrains
the models and refreshes the site **automatically every morning**.

## Daily automation

Every day at 7:30 AM ET, `.github/workflows/daily.yml` runs `daily_update.py`:

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

**Backtest** — 4,097 held-out games (2025 + played 2026, never seen in training;
`outputs/metrics.json`):

| Model | Log loss | Brier | AUC | Accuracy |
|---|---|---|---|---|
| Elo baseline | 0.6827 | 0.2449 | 0.567 | 55.9% |
| Logistic regression | 0.6818 | 0.2444 | 0.571 | 55.3% |
| XGBoost | 0.6838 | 0.2454 | 0.566 | 55.4% |
| Elo+LR stack | **0.6818** | 0.2444 | 0.572 | 55.7% |
| Poisson–Skellam | 0.6874 | 0.2471 | 0.570 | 55.0% |

**Live tracker** — 650 games predicted *before first pitch* and scored the next morning,
restricted to the games every model called, with a paired bootstrap (5,000 resamples) on
each model's log-loss gap to Elo:

| Model | Log loss | Accuracy | Δ vs Elo | 95% CI |
|---|---|---|---|---|
| Elo baseline | 0.6868 | 54.9% | — | — |
| Elo+LR stack | **0.6863** | 54.2% | −0.0005 | [−0.0051, +0.0041] |
| XGBoost | 0.6872 | 54.2% | +0.0004 | [−0.0058, +0.0068] |
| Logistic regression | 0.6876 | 53.7% | +0.0008 | [−0.0053, +0.0071] |
| Poisson–Skellam | 0.6928 | 54.9% | +0.0060 | [−0.0046, +0.0166] |

Every interval spans zero. The best variant is 0.0005 log loss ahead of a rating system
with no fitted features at all — a gap ten times smaller than the sampling noise.

**Why that is the expected answer.** Elo already integrates team strength from every game
ever played, and the features layered on top (season-to-date rate stats, 30-game form,
probable-starter quality, bullpen fatigue, rest, park) are largely *functions of the same
history*. Public pre-game information is close to exhausted; what remains is in-game state,
pitch-level detail, and day-of bullpen availability. Reporting a 0.001 improvement as a win
would be the mistake.

**Read log loss, not accuracy.** Always picking the home team scores 51.2% on that live
sample. The models' own probabilities imply ~56% expected accuracy and they realize ~54%
with a ±4-point 95% margin — i.e. they are well calibrated, and accuracy at this sample
size cannot separate them. Calibration curves are on the site.

> The live-sample figures above are rescored by the daily CI run, so they drift by a
> few tenths of a point. `outputs/metrics.json` and the site always carry the current
> numbers; the conclusion — that the gap between the models is smaller than the
> sampling noise — has held every run.

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
pytest -q          # 77 tests, ~2 s, no network and no data files
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
