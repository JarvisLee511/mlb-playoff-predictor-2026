"""Export everything the static website needs as JSON files in docs/data/."""
import datetime as dt
import json

import numpy as np
import pandas as pd

from src.config import OUTPUTS, ROOT
from src.predictions import load_log, preview_tomorrow, today_et
from src.transactions import fetch_recent_transactions

SITE_DATA = ROOT / "docs" / "data"

MODELS = {
    "elo": "p_home_elo",
    "lr": "p_home_lr",
    "xgb": "p_home_xgb",
    "ens": "p_home_ens",
    "skl": "p_home_skl",
}


def _write(name: str, obj) -> None:
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    (SITE_DATA / name).write_text(json.dumps(obj, indent=1), encoding="utf-8")


GAME_COLS = ["game_time_et", "home_name", "away_name", "home_pitcher",
             "away_pitcher", "p_home_elo", "p_home_lr", "p_home_xgb",
             "p_home_ens", "p_home_skl", "status"]


def _records(df: pd.DataFrame) -> list[dict]:
    cols = [c for c in GAME_COLS if c in df.columns]
    view = df[cols].astype(object).where(pd.notna(df[cols]), None)
    return view.to_dict(orient="records")


def export_today(log: pd.DataFrame) -> None:
    today = today_et().isoformat()
    games = log[log["date"] == today].sort_values("game_time_et")

    tomorrow_rows = preview_tomorrow()
    tomorrow = pd.DataFrame(tomorrow_rows)
    if len(tomorrow):
        tomorrow = tomorrow.sort_values("game_time_et")

    _write(
        "today.json",
        {
            "date": today,
            "games": _records(games),
            "tomorrow_date": tomorrow_rows[0]["date"] if tomorrow_rows else None,
            "tomorrow": _records(tomorrow) if len(tomorrow) else [],
        },
    )


def export_odds() -> None:
    odds = pd.read_csv(OUTPUTS / "playoff_odds_2026.csv")
    cols = ["team_name", "abbrev", "league", "division", "current_wins", "proj_wins",
            "make_playoffs", "win_division", "first_round_bye", "win_pennant",
            "win_world_series"]
    odds["proj_wins"] = odds["proj_wins"].round(1)
    _write("odds.json", odds[cols].to_dict(orient="records"))

    elo = pd.read_csv(OUTPUTS / "elo_current.csv")
    elo["elo"] = elo["elo"].round(1)
    _write("elo.json", elo[["team_name", "abbrev", "league", "elo"]].to_dict(orient="records"))


def _log_loss_series(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# Below this many scored games the realized accuracy is dominated by noise
# (95% CI on a coin-flip at n=37 is ~±16 points), so the site flags it as
# not-yet-meaningful rather than letting a small sample read as signal.
MIN_RELIABLE_N = 200


def _wilson_ci(hits: int, n: int, z: float = 1.96) -> list[float]:
    """95% Wilson score interval for accuracy — honest about small samples."""
    if n == 0:
        return [0.0, 1.0]
    phat = hits / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return [round(float(max(0.0, center - half)), 4),
            round(float(min(1.0, center + half)), 4)]


def _calibration_bins(p: np.ndarray, y: np.ndarray) -> list[dict]:
    """Reliability curve: for each 0.1-wide predicted-prob bin, mean predicted
    vs observed win rate. Empty bins are dropped so the front end just plots
    whatever populated points exist."""
    edges = np.arange(0.0, 1.0001, 0.1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, len(edges) - 2)
    bins = []
    for b in range(len(edges) - 1):
        mask = idx == b
        n = int(mask.sum())
        if not n:
            continue
        bins.append({
            "p_mean": round(float(p[mask].mean()), 4),
            "win_rate": round(float(y[mask].mean()), 4),
            "n": n,
        })
    return bins


def export_accuracy(log: pd.DataFrame) -> None:
    finals = log[log["status"] == "final"].copy()
    out = {
        "n_scored": int(len(finals)),
        "min_reliable_n": MIN_RELIABLE_N,
        "reliable": bool(len(finals) >= MIN_RELIABLE_N),
        "summary": {},
        "calibration": {},
        "daily": {},
        "recent": [],
    }

    if len(finals):
        finals["home_win"] = finals["home_win"].astype(int)

        # per-model: older log rows may predate a model's introduction (NaN)
        daily_dates: list[str] = sorted(finals["date"].unique())
        out["daily"]["dates"] = daily_dates
        for key, col in MODELS.items():
            if col not in finals.columns:
                continue
            sub = finals[finals[col].notna()]
            if not len(sub):
                continue
            y = sub["home_win"].to_numpy()
            p = sub[col].to_numpy(dtype=float)
            ll = _log_loss_series(p, y)
            pred = (p > 0.5).astype(int)
            hits = int((pred == y).sum())
            out["summary"][key] = {
                "log_loss": round(float(ll.mean()), 4),
                "brier": round(float(((p - y) ** 2).mean()), 4),
                "accuracy": round(float(hits / len(sub)), 4),
                "acc_ci": _wilson_ci(hits, len(sub)),
                # what the model itself implies it should hit, given its own
                # probabilities; realized ~= expected means well-calibrated,
                # so a "low" 55% is the honest ceiling, not underperformance.
                "expected_accuracy": round(float(np.maximum(p, 1 - p).mean()), 4),
                "n": int(len(sub)),
            }
            out["calibration"][key] = _calibration_bins(p, y)
            per_day = (
                pd.DataFrame({"date": sub["date"], "ll": ll})
                .groupby("date")["ll"].agg(["sum", "count"])
                .reindex(daily_dates)
            )
            cum = per_day["sum"].cumsum() / per_day["count"].cumsum()
            out["daily"][key] = [None if pd.isna(v) else round(v, 4) for v in cum]

        recent = finals.sort_values(["date", "game_time_et"]).tail(30)
        recent = recent[
            ["date", "home_name", "away_name", "home_score", "away_score",
             "home_win"] + [c for c in MODELS.values() if c in recent.columns]
        ]
        recent = recent.astype(object).where(pd.notna(recent), None)
        out["recent"] = recent.to_dict(orient="records")

    _write("accuracy.json", out)


def export_transactions() -> None:
    trans = fetch_recent_transactions(days=10)
    _write("transactions.json", trans.to_dict(orient="records"))


def export_meta(log: pd.DataFrame) -> None:
    metrics = json.loads((OUTPUTS / "metrics.json").read_text())
    _write(
        "meta.json",
        {
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "n_predictions_logged": int(len(log)),
            "n_scored": int((log["status"] == "final").sum()),
            "backtest_metrics": {k: v for k, v in metrics.items() if isinstance(v, dict)},
            "best_model": metrics.get("best_model"),
        },
    )


def export_all() -> None:
    log = load_log()
    export_today(log)
    export_odds()
    export_accuracy(log)
    export_transactions()
    export_meta(log)
    print(f"Site data exported to {SITE_DATA}")
