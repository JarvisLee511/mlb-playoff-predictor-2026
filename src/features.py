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


LEAGUE_FIP = 4.20          # league-average FIP used as shrinkage prior
FIP_NUM_PRIOR = (LEAGUE_FIP - 3.10) * 30  # prior numerator at 30 IP
KBB_PRIOR_RATE = 0.14      # league-average (K-BB)/BF
PARK_HFA_PRIOR = 0.54      # league-average home win rate

SP_STATS = ["sp_fip", "sp_kbb", "sp_fip5"]
BP_STATS = ["bp_fip"]


def _shrunk_fip(num, ip, prior_ip=30.0):
    return (num + (LEAGUE_FIP - 3.10) * prior_ip) / (ip + prior_ip) + 3.10


def build_pitcher_snapshots(pitcher_logs: pd.DataFrame) -> pd.DataFrame:
    """Post-appearance cumulative stats per pitcher: season-to-date shrunk FIP,
    (K-BB)/BF, and last-5-starts FIP. As-of joins use date < game date, so all
    values are pre-game knowable."""
    p = pitcher_logs.sort_values(["pitcher_id", "season", "date", "game_id"]).reset_index(drop=True)
    p["fip_num"] = 13 * p["hr"] + 3 * (p["bb"] + p["hbp"]) - 2 * p["so"]

    grp = p.groupby(["pitcher_id", "season"], sort=False)
    cum_num = grp["fip_num"].cumsum()
    cum_ip = grp["ip"].cumsum()
    cum_so, cum_bb, cum_bf = grp["so"].cumsum(), grp["bb"].cumsum(), grp["bf"].cumsum()

    snaps = pd.DataFrame(
        {
            "pitcher_id": p["pitcher_id"],
            "season": p["season"],
            "date": pd.to_datetime(p["date"]),
            "sp_fip": _shrunk_fip(cum_num, cum_ip),
            "sp_kbb": (cum_so - cum_bb + KBB_PRIOR_RATE * 120) / (cum_bf + 120),
        }
    )

    starts = p[p["gs"] >= 1]
    sgrp = starts.groupby(["pitcher_id", "season"], sort=False)
    num5 = sgrp["fip_num"].transform(lambda s: s.rolling(5, min_periods=1).sum())
    ip5 = sgrp["ip"].transform(lambda s: s.rolling(5, min_periods=1).sum())
    snaps.loc[starts.index, "sp_fip5"] = _shrunk_fip(num5, ip5, prior_ip=15.0)
    snaps["sp_fip5"] = snaps.groupby(["pitcher_id", "season"], sort=False)["sp_fip5"].ffill()
    snaps["sp_fip5"] = snaps["sp_fip5"].fillna(snaps["sp_fip"])
    return snaps


def build_bullpen_snapshots(gamelogs: pd.DataFrame, pitcher_logs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Bullpen = team pitching minus the starter's line, per game. Returns
    post-game cumulative bullpen-FIP snapshots and a (team_id, date) -> bullpen
    IP dict for fatigue lookups."""
    starters = (
        pitcher_logs[pitcher_logs["gs"] >= 1]
        .groupby(["team_id", "game_id"])[["ip", "er", "bb", "so", "hr", "hbp"]]
        .sum()
        .add_prefix("sp_")
        .reset_index()
    )
    g = gamelogs.merge(starters, on=["team_id", "game_id"], how="inner")
    for team_col, sp_col in (
        ("p_ip", "sp_ip"), ("p_baseOnBalls", "sp_bb"), ("p_strikeOuts", "sp_so"),
        ("p_homeRuns", "sp_hr"), ("p_hitByPitch", "sp_hbp"),
    ):
        g[f"bp_{team_col}"] = (g[team_col] - g[sp_col]).clip(lower=0)
    g["bp_fip_num"] = (
        13 * g["bp_p_homeRuns"] + 3 * (g["bp_p_baseOnBalls"] + g["bp_p_hitByPitch"])
        - 2 * g["bp_p_strikeOuts"]
    )

    g = g.sort_values(["team_id", "season", "date", "game_id"]).reset_index(drop=True)
    grp = g.groupby(["team_id", "season"], sort=False)
    snaps = pd.DataFrame(
        {
            "team_id": g["team_id"],
            "season": g["season"],
            "date": pd.to_datetime(g["date"]),
            "bp_fip": _shrunk_fip(grp["bp_fip_num"].cumsum(), grp["bp_p_ip"].cumsum()),
        }
    )
    daily_ip = g.groupby(["team_id", "date"])["bp_p_ip"].sum()
    return snaps, daily_ip.to_dict()


def bullpen_fatigue(team_id: int, date, daily_ip: dict, days: int = 3) -> float:
    """Bullpen innings thrown in the `days` days before `date`."""
    total = 0.0
    for k in range(1, days + 1):
        d = (date - pd.Timedelta(days=k)).strftime("%Y-%m-%d")
        total += daily_ip.get((team_id, d), 0.0)
    return total


def _asof_starter(games: pd.DataFrame, snaps: pd.DataFrame, id_col: str, prefix: str) -> pd.DataFrame:
    """As-of join: starter's latest snapshot strictly BEFORE the game date."""
    left = games[["game_id", "date", "season", id_col]].dropna(subset=[id_col]).copy()
    left[id_col] = left[id_col].astype(int)
    left = left.sort_values("date")
    right = snaps.sort_values("date")
    merged = pd.merge_asof(
        left, right, on="date",
        left_by=[id_col, "season"], right_by=["pitcher_id", "season"],
        allow_exact_matches=False,
    )
    out = merged[["game_id"] + SP_STATS].rename(columns={c: f"{prefix}_{c}" for c in SP_STATS})
    return out


def _asof_bullpen(games: pd.DataFrame, snaps: pd.DataFrame, id_col: str, prefix: str) -> pd.DataFrame:
    left = games[["game_id", "date", "season", id_col]].copy().sort_values("date")
    right = snaps.sort_values("date")
    merged = pd.merge_asof(
        left, right, on="date",
        left_by=[id_col, "season"], right_by=["team_id", "season"],
        allow_exact_matches=False,
    )
    return merged[["game_id"] + BP_STATS].rename(columns={c: f"{prefix}_{c}" for c in BP_STATS})


def build_features(games: pd.DataFrame, gamelogs: pd.DataFrame | None = None,
                   probables: pd.DataFrame | None = None,
                   pitcher_logs: pd.DataFrame | None = None) -> pd.DataFrame:
    """games must already carry home_elo_pre / away_elo_pre / elo_prob_home."""
    games = games.sort_values(["date", "game_id"]).reset_index(drop=True)
    games["date"] = pd.to_datetime(games["date"])

    history: dict[int, deque] = {}      # team_id -> deque of (win, run_diff)
    season_record: dict[tuple, list] = {}  # (team_id, season) -> [wins, games]
    last_played: dict[int, pd.Timestamp] = {}
    park_w: dict[int, float] = {}       # home wins at this team's park (all seasons)
    park_n: dict[int, int] = {}

    feat_rows = []
    for g in games.itertuples():
        row = {}
        row["park_hfa"] = (park_w.get(g.home_id, 0) + PARK_HFA_PRIOR * 200) / (
            park_n.get(g.home_id, 0) + 200
        )
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
        park_w[g.home_id] = park_w.get(g.home_id, 0) + g.home_win
        park_n[g.home_id] = park_n.get(g.home_id, 0) + 1

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

    if probables is not None and pitcher_logs is not None:
        out = out.merge(
            probables[["game_id", "home_sp_id", "away_sp_id"]], on="game_id", how="left"
        )
        sp_snaps = build_pitcher_snapshots(pitcher_logs)
        out = out.merge(_asof_starter(out, sp_snaps, "home_sp_id", "home"), on="game_id", how="left")
        out = out.merge(_asof_starter(out, sp_snaps, "away_sp_id", "away"), on="game_id", how="left")
        out["sp_fip_diff"] = (out["home_sp_fip"] - out["away_sp_fip"]).fillna(0)
        out["sp_kbb_diff"] = (out["home_sp_kbb"] - out["away_sp_kbb"]).fillna(0)
        out["sp_fip5_diff"] = (out["home_sp_fip5"] - out["away_sp_fip5"]).fillna(0)

        bp_snaps, bp_daily_ip = build_bullpen_snapshots(gamelogs, pitcher_logs)
        out = out.merge(_asof_bullpen(out, bp_snaps, "home_id", "home"), on="game_id", how="left")
        out = out.merge(_asof_bullpen(out, bp_snaps, "away_id", "away"), on="game_id", how="left")
        out["bp_fip_diff"] = (out["home_bp_fip"] - out["away_bp_fip"]).fillna(0)
        out["bp_fatigue_diff"] = [
            bullpen_fatigue(h, d, bp_daily_ip) - bullpen_fatigue(a, d, bp_daily_ip)
            for h, a, d in zip(out["home_id"], out["away_id"], out["date"])
        ]
    else:
        for col in ("sp_fip_diff", "sp_kbb_diff", "sp_fip5_diff", "bp_fip_diff", "bp_fatigue_diff"):
            out[col] = 0.0
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
    # starting pitchers (shrunk season-to-date + last-5-starts form)
    "sp_fip_diff",
    "sp_kbb_diff",
    "sp_fip5_diff",
    # bullpen quality + 3-day workload fatigue
    "bp_fip_diff",
    "bp_fatigue_diff",
    # per-park home advantage (expanding, shrunk to league 0.54)
    "park_hfa",
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
