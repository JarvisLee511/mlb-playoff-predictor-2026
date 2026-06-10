"""Daily pre-game predictions and scoring of past predictions.

Every day we log a prediction for each scheduled game from all three models
(Elo, logistic regression, XGBoost) BEFORE the games are played, then score
them against actual results the next morning. The accumulated log is a true
out-of-sample track record — the models never see a game before predicting it.
"""
import datetime as dt
from zoneinfo import ZoneInfo

import joblib
import pandas as pd

from src.config import DATA_PROCESSED, DATA_RAW, OUTPUTS, ROOT
from src.data.fetch import SCHEDULE_URL, _get_json
from src.features import FEATURE_COLS
from src.models.elo import elo_win_prob

PRED_LOG = ROOT / "data" / "predictions_log.csv"
ET = ZoneInfo("America/New_York")

LOG_COLS = [
    "date", "game_id", "home_id", "home_name", "away_id", "away_name",
    "game_time_et", "home_pitcher", "away_pitcher",
    "p_home_elo", "p_home_lr", "p_home_xgb",
    "status", "home_score", "away_score", "home_win",
]


def today_et() -> dt.date:
    return dt.datetime.now(ET).date()


def load_log() -> pd.DataFrame:
    if PRED_LOG.exists():
        return pd.read_csv(PRED_LOG)
    return pd.DataFrame(columns=LOG_COLS)


def _fetch_day_schedule(date: dt.date) -> list[dict]:
    data = _get_json(
        SCHEDULE_URL,
        {
            "sportId": 1,
            "date": date.isoformat(),
            "gameType": "R",
            "hydrate": "probablePitcher",
        },
    )
    games = []
    for day in data.get("dates", []):
        games.extend(day["games"])
    return games


def _matchup_features(home_id: int, away_id: int, snap: pd.DataFrame,
                      last_played: dict, date: dt.date) -> pd.DataFrame:
    h, a = snap.loc[home_id], snap.loc[away_id]

    def rest(team_id: int) -> int:
        last = last_played.get(team_id)
        days = (date - last).days if last is not None else 3
        return min(max(days, 0), 10)

    row = {
        "elo_diff": h["elo"] - a["elo"],
        "winpct30_diff": h["winpct_30"] - a["winpct_30"],
        "rundiff30_diff": h["rundiff_30"] - a["rundiff_30"],
        "season_winpct_diff": h["season_winpct"] - a["season_winpct"],
        "rest_diff": rest(home_id) - rest(away_id),
        "home_winpct_30": h["winpct_30"],
        "away_winpct_30": a["winpct_30"],
        "home_rundiff_30": h["rundiff_30"],
        "away_rundiff_30": a["rundiff_30"],
    }
    for stat in ("ops", "era", "fip", "whip", "pyth", "off_bbk"):
        row[f"{stat}_diff"] = h[stat] - a[stat]
    return pd.DataFrame([row])[FEATURE_COLS]


def _predictions_for_date(date: dt.date, skip_ids: set | None = None) -> list[dict]:
    """Compute prediction rows for every scheduled game on `date`."""
    snap = pd.read_csv(DATA_PROCESSED / "current_team_stats.csv").set_index("team_id")
    logreg = joblib.load(OUTPUTS / "model_logreg.joblib")
    xgb = joblib.load(OUTPUTS / "model_xgb.joblib")

    games_hist = pd.read_csv(DATA_RAW / "games.csv", parse_dates=["date"])
    last_played: dict[int, dt.date] = {}
    for side in ("home_id", "away_id"):
        for team, d in games_hist.groupby(side)["date"].max().items():
            prev = last_played.get(team)
            last_played[team] = max(prev, d.date()) if prev else d.date()

    rows = []
    for g in _fetch_day_schedule(date):
        if skip_ids and g["gamePk"] in skip_ids:
            continue
        home, away = g["teams"]["home"], g["teams"]["away"]
        hid, aid = home["team"]["id"], away["team"]["id"]
        if hid not in snap.index or aid not in snap.index:
            continue

        X = _matchup_features(hid, aid, snap, last_played, date)
        time_et = (
            dt.datetime.fromisoformat(g["gameDate"].replace("Z", "+00:00"))
            .astimezone(ET)
            .strftime("%H:%M")
        )
        rows.append(
            {
                "date": date.isoformat(),
                "game_id": g["gamePk"],
                "home_id": hid,
                "home_name": home["team"]["name"],
                "away_id": aid,
                "away_name": away["team"]["name"],
                "game_time_et": time_et,
                "home_pitcher": home.get("probablePitcher", {}).get("fullName", "TBD"),
                "away_pitcher": away.get("probablePitcher", {}).get("fullName", "TBD"),
                "p_home_elo": round(elo_win_prob(snap.loc[hid, "elo"], snap.loc[aid, "elo"]), 4),
                "p_home_lr": round(float(logreg.predict_proba(X)[0, 1]), 4),
                "p_home_xgb": round(float(xgb.predict_proba(X)[0, 1]), 4),
                "status": "pending",
                "home_score": None,
                "away_score": None,
                "home_win": None,
            }
        )
    return rows


def predict_today(date: dt.date | None = None) -> pd.DataFrame:
    """Append predictions for `date` (default: today ET) to the log."""
    date = date or today_et()
    log = load_log()
    rows = _predictions_for_date(date, skip_ids=set(log["game_id"]))
    log = pd.concat([log, pd.DataFrame(rows, columns=LOG_COLS)], ignore_index=True)
    log.to_csv(PRED_LOG, index=False)
    print(f"Logged {len(rows)} new predictions for {date}")
    return log


def preview_tomorrow() -> list[dict]:
    """Tomorrow's predictions for display only — NOT logged. They are recomputed
    (and only then logged) tomorrow morning once tonight's results are in."""
    return _predictions_for_date(today_et() + dt.timedelta(days=1))


def score_pending() -> pd.DataFrame:
    """Fill in results for pending predictions using the freshly fetched games.csv."""
    log = load_log()
    if log.empty:
        return log
    results = pd.read_csv(DATA_RAW / "games.csv").set_index("game_id")

    scored = 0
    for idx in log.index[log["status"] == "pending"]:
        gid = log.at[idx, "game_id"]
        if gid in results.index:
            r = results.loc[gid]
            log.at[idx, "home_score"] = int(r["home_score"])
            log.at[idx, "away_score"] = int(r["away_score"])
            log.at[idx, "home_win"] = int(r["home_win"])
            log.at[idx, "status"] = "final"
            scored += 1

    log.to_csv(PRED_LOG, index=False)
    print(f"Scored {scored} predictions ({(log['status'] == 'final').sum()} total finals)")
    return log
