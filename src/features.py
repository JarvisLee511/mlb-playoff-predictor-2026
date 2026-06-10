"""Build pre-game features for the ML win-probability model.

All features use only information available BEFORE first pitch:
rolling form (last 30 games), season-to-date win%, rest days, Elo, and
season-to-date advanced stats (OPS, ERA, FIP, WHIP, Pythagorean win%)
accumulated from per-game team logs.
"""
from collections import deque

import numpy as np
import pandas as pd

ROLL_WINDOW = 30
MIN_GAMES = 10      # require this many prior games for rolling stats
ADV_MIN_GAMES = 15  # advanced rate stats are noise below this
PYTH_EXP = 1.83

ADV_STATS = ["ops", "era", "fip", "whip", "pyth", "off_bbk"]


def build_advanced_pregame(gamelogs: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date advanced stats BEFORE each game, keyed (team_id, game_id).

    Cumulative counting stats are taken up to but excluding the current game,
    so every value is knowable at first pitch.
    """
    g = gamelogs.sort_values(["team_id", "season", "date", "game_id"]).reset_index(drop=True)
    counting = [c for c in g.columns if c.startswith(("h_", "p_"))]
    g[counting] = g[counting].fillna(0)

    grp = g.groupby(["team_id", "season"], sort=False)
    pre = grp[counting].cumsum() - g[counting]
    games_before = grp.cumcount()

    obp_den = pre["h_atBats"] + pre["h_baseOnBalls"] + pre["h_hitByPitch"] + pre["h_sacFlies"]
    obp = (pre["h_hits"] + pre["h_baseOnBalls"] + pre["h_hitByPitch"]) / obp_den
    total_bases = pre["h_hits"] + pre["h_doubles"] + 2 * pre["h_triples"] + 3 * pre["h_homeRuns"]
    slg = total_bases / pre["h_atBats"]

    rs, ra = pre["h_runs"], pre["p_runs"]
    out = pd.DataFrame(
        {
            "team_id": g["team_id"],
            "game_id": g["game_id"],
            "ops": obp + slg,
            "era": 9 * pre["p_earnedRuns"] / pre["p_ip"],
            "fip": (13 * pre["p_homeRuns"] + 3 * (pre["p_baseOnBalls"] + pre["p_hitByPitch"])
                    - 2 * pre["p_strikeOuts"]) / pre["p_ip"] + 3.10,
            "whip": (pre["p_baseOnBalls"] + pre["p_hits"]) / pre["p_ip"],
            "pyth": rs**PYTH_EXP / (rs**PYTH_EXP + ra**PYTH_EXP),
            "off_bbk": (pre["h_baseOnBalls"] - pre["h_strikeOuts"]) / pre["h_plateAppearances"],
        }
    )
    out.loc[(games_before < ADV_MIN_GAMES).to_numpy(), ADV_STATS] = np.nan
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def current_advanced(gamelogs: pd.DataFrame, season: int) -> pd.DataFrame:
    """Season-to-date advanced stats per team INCLUDING all played games —
    used to predict upcoming games."""
    cur = gamelogs[gamelogs["season"] == season]
    counting = [c for c in cur.columns if c.startswith(("h_", "p_"))]
    tot = cur.groupby("team_id")[counting].sum()

    obp = (tot["h_hits"] + tot["h_baseOnBalls"] + tot["h_hitByPitch"]) / (
        tot["h_atBats"] + tot["h_baseOnBalls"] + tot["h_hitByPitch"] + tot["h_sacFlies"]
    )
    slg = (tot["h_hits"] + tot["h_doubles"] + 2 * tot["h_triples"] + 3 * tot["h_homeRuns"]) / tot["h_atBats"]
    rs, ra = tot["h_runs"], tot["p_runs"]
    return pd.DataFrame(
        {
            "ops": obp + slg,
            "era": 9 * tot["p_earnedRuns"] / tot["p_ip"],
            "fip": (13 * tot["p_homeRuns"] + 3 * (tot["p_baseOnBalls"] + tot["p_hitByPitch"])
                    - 2 * tot["p_strikeOuts"]) / tot["p_ip"] + 3.10,
            "whip": (tot["p_baseOnBalls"] + tot["p_hits"]) / tot["p_ip"],
            "pyth": rs**PYTH_EXP / (rs**PYTH_EXP + ra**PYTH_EXP),
            "off_bbk": (tot["h_baseOnBalls"] - tot["h_strikeOuts"]) / tot["h_plateAppearances"],
        }
    )


def build_features(games: pd.DataFrame, gamelogs: pd.DataFrame | None = None) -> pd.DataFrame:
    """games must already carry home_elo_pre / away_elo_pre / elo_prob_home."""
    games = games.sort_values(["date", "game_id"]).reset_index(drop=True)
    games["date"] = pd.to_datetime(games["date"])

    history: dict[int, deque] = {}      # team_id -> deque of (win, run_diff)
    season_record: dict[tuple, list] = {}  # (team_id, season) -> [wins, games]
    last_played: dict[int, pd.Timestamp] = {}

    feat_rows = []
    for g in games.itertuples():
        row = {}
        for side, team in (("home", g.home_id), ("away", g.away_id)):
            hist = history.setdefault(team, deque(maxlen=ROLL_WINDOW))
            if len(hist) >= MIN_GAMES:
                row[f"{side}_winpct_30"] = sum(w for w, _ in hist) / len(hist)
                row[f"{side}_rundiff_30"] = sum(d for _, d in hist) / len(hist)
            else:
                row[f"{side}_winpct_30"] = 0.5
                row[f"{side}_rundiff_30"] = 0.0

            wins, played = season_record.get((team, g.season), [0, 0])
            row[f"{side}_season_winpct"] = wins / played if played else 0.5

            rest = (g.date - last_played[team]).days if team in last_played else 3
            row[f"{side}_rest_days"] = min(max(rest, 0), 10)
            row[f"{side}_enough_history"] = int(len(hist) >= MIN_GAMES)

        feat_rows.append(row)

        # update state AFTER recording pre-game features
        run_diff = g.home_score - g.away_score
        history[g.home_id].append((g.home_win, run_diff))
        history[g.away_id].append((1 - g.home_win, -run_diff))
        for team, won in ((g.home_id, g.home_win), (g.away_id, 1 - g.home_win)):
            rec = season_record.setdefault((team, g.season), [0, 0])
            rec[0] += won
            rec[1] += 1
            last_played[team] = g.date

    feats = pd.DataFrame(feat_rows)
    out = pd.concat([games.reset_index(drop=True), feats], axis=1)
    out["elo_diff"] = out["home_elo_pre"] - out["away_elo_pre"]
    out["winpct30_diff"] = out["home_winpct_30"] - out["away_winpct_30"]
    out["rundiff30_diff"] = out["home_rundiff_30"] - out["away_rundiff_30"]
    out["season_winpct_diff"] = out["home_season_winpct"] - out["away_season_winpct"]
    out["rest_diff"] = out["home_rest_days"] - out["away_rest_days"]

    if gamelogs is not None:
        adv = build_advanced_pregame(gamelogs)
        for side in ("home", "away"):
            out = out.merge(
                adv.rename(columns={c: f"{side}_{c}" for c in ADV_STATS}),
                how="left",
                left_on=[f"{side}_id", "game_id"],
                right_on=["team_id", "game_id"],
            ).drop(columns="team_id")
        for stat in ADV_STATS:
            out[f"{stat}_diff"] = (out[f"home_{stat}"] - out[f"away_{stat}"]).fillna(0)
    else:
        for stat in ADV_STATS:
            out[f"{stat}_diff"] = 0.0
    return out


FEATURE_COLS = [
    "elo_diff",
    "winpct30_diff",
    "rundiff30_diff",
    "season_winpct_diff",
    "rest_diff",
    "home_winpct_30",
    "away_winpct_30",
    "home_rundiff_30",
    "away_rundiff_30",
    # season-to-date advanced stat differentials (home minus away)
    "ops_diff",
    "era_diff",
    "fip_diff",
    "whip_diff",
    "pyth_diff",
    "off_bbk_diff",
]


def current_team_snapshot(
    features_df: pd.DataFrame,
    ratings: dict[int, float],
    gamelogs: pd.DataFrame | None = None,
    season: int | None = None,
) -> pd.DataFrame:
    """Latest rolling stats + Elo (+ advanced stats) per team, for predicting
    upcoming games."""
    rows = []
    df = features_df.sort_values("date")
    for team_id, elo in ratings.items():
        home_rows = df[df["home_id"] == team_id]
        away_rows = df[df["away_id"] == team_id]
        last_home = home_rows.iloc[-1] if len(home_rows) else None
        last_away = away_rows.iloc[-1] if len(away_rows) else None

        # take whichever appearance is most recent
        use_home = last_away is None or (
            last_home is not None and last_home["date"] >= last_away["date"]
        )
        src, side = (last_home, "home") if use_home else (last_away, "away")
        if src is None:
            continue
        rows.append(
            {
                "team_id": team_id,
                "elo": elo,
                "winpct_30": src[f"{side}_winpct_30"],
                "rundiff_30": src[f"{side}_rundiff_30"],
                "season_winpct": src[f"{side}_season_winpct"],
            }
        )
    snap = pd.DataFrame(rows)
    if gamelogs is not None and season is not None:
        snap = snap.merge(
            current_advanced(gamelogs, season).reset_index(), on="team_id", how="left"
        )
    return snap
