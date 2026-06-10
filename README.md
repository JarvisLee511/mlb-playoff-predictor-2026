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
   XGBoost, an **isotonic-calibrated Elo+LR ensemble**, and a **Poisson–Skellam**
   run-distribution model. Every feature is knowable at first pitch (as-of joins,
   no leakage). Time-based split: train 2015–2023, validate 2024, test 2025–2026.
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

Win-probability models are evaluated on held-out 2025–2026 games (never seen in
training). See `outputs/metrics.json` after running the pipeline — the headline metric
is **log loss / calibration**, not accuracy: baseball games are close to coin flips
(the best teams lose 60+ games a year), so a well-calibrated 58% is the realistic
ceiling, and beating the Elo baseline at all requires the ML features to add signal.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

python run_pipeline.py           # fetch + train + simulate (~5 min)
streamlit run app/streamlit_app.py
```

`python run_pipeline.py --no-fetch` reuses cached data and skips the API calls.

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
└── outputs/                 # odds, metrics, calibration, current Elo
```

## Modeling notes & limitations

- **No data leakage**: every feature is computable before first pitch; rolling stats
  are updated only after each game is recorded.
- Seeding ties are broken randomly rather than by MLB's head-to-head tiebreakers;
  over 10,000 simulations the effect on odds is negligible.
- The simulation uses Elo probabilities (future games have unknown rolling stats);
  the ML models power game-level evaluation and the matchup predictor.
- Starting pitchers, injuries, and trades are not modeled — the largest known gap,
  and the natural next iteration (probable pitchers are available from the same API).
- The shortened 2020 season is kept for Elo continuity but its small sample is
  handled by the cross-season rolling windows.

## Data source

All data from the official MLB Stats API via the
[MLB-StatsAPI](https://github.com/toddrob99/MLB-StatsAPI) Python wrapper.
This project is for educational/portfolio purposes and is not affiliated with MLB.
