"""Monte Carlo simulation of the rest of the 2026 MLB season + postseason.

For each simulation:
  1. simulate every remaining regular-season game from Elo win probabilities
  2. seed the playoffs under the 12-team format (3 division winners + 3 wild
     cards per league; top two division winners get byes)
  3. play out Wild Card (Bo3), Division Series (Bo5), LCS (Bo7), World Series (Bo7)

Ties for seeding are broken randomly (real MLB tiebreakers use head-to-head
records — out of scope, and the random break washes out over 10k sims).

Output: outputs/playoff_odds_2026.csv with per-team probabilities.
"""
import numpy as np
import pandas as pd

from src.config import CURRENT_SEASON, DATA_RAW, N_SIMS, OUTPUTS, RNG_SEED
from src.models.elo import elo_win_prob

# True = higher seed (team_a) is home that game
SERIES_HOME = {
    3: [True, True, True],
    5: [True, True, False, False, True],
    7: [True, True, False, False, False, True, True],
}


def sim_series(a: int, b: int, elos: dict, length: int, rng) -> int:
    """Return the winner of a best-of-`length` series; `a` is the higher seed."""
    need = length // 2 + 1
    wins_a = wins_b = 0
    for a_home in SERIES_HOME[length]:
        p_a = elo_win_prob(elos[a], elos[b]) if a_home else 1 - elo_win_prob(elos[b], elos[a])
        if rng.random() < p_a:
            wins_a += 1
        else:
            wins_b += 1
        if wins_a == need:
            return a
        if wins_b == need:
            return b
    return a if wins_a > wins_b else b


def seed_league(records: dict, teams: pd.DataFrame, league: str, rng) -> list[int]:
    """Return the 6 playoff seeds (team_ids) for one league."""
    lg = teams[teams["league"] == league]
    jitter = {t: rng.random() * 1e-6 for t in lg["team_id"]}

    div_winners = []
    for div in lg["division"].unique():
        ids = lg[lg["division"] == div]["team_id"].tolist()
        div_winners.append(max(ids, key=lambda t: records[t] + jitter[t]))
    div_winners.sort(key=lambda t: records[t] + jitter[t], reverse=True)

    rest = [t for t in lg["team_id"] if t not in div_winners]
    wildcards = sorted(rest, key=lambda t: records[t] + jitter[t], reverse=True)[:3]
    return div_winners + wildcards  # seeds 1..6


def sim_postseason(seeds: dict[str, list[int]], elos: dict, records: dict, rng) -> dict:
    """Play out both leagues' brackets; returns dict of round results."""
    out = {"ds": [], "cs": [], "ws_teams": []}
    pennant = {}
    for league, s in seeds.items():
        wc1 = sim_series(s[2], s[5], elos, 3, rng)  # 3 vs 6
        wc2 = sim_series(s[3], s[4], elos, 3, rng)  # 4 vs 5
        out["ds"] += [s[0], s[1], wc1, wc2]
        d1 = sim_series(s[0], wc2, elos, 5, rng)
        d2 = sim_series(s[1], wc1, elos, 5, rng)
        out["cs"] += [d1, d2]
        pennant[league] = sim_series(d1, d2, elos, 7, rng)

    al, nl = pennant["American League"], pennant["National League"]
    out["ws_teams"] = [al, nl]
    higher, lower = (al, nl) if records[al] >= records[nl] else (nl, al)
    out["champion"] = sim_series(higher, lower, elos, 7, rng)
    return out


def main() -> None:
    games = pd.read_csv(DATA_RAW / "games.csv")
    remaining = pd.read_csv(DATA_RAW / "remaining_2026.csv")
    teams = pd.read_csv(DATA_RAW / "teams.csv")
    elo_now = pd.read_csv(OUTPUTS / "elo_current.csv")
    elos = dict(zip(elo_now["team_id"], elo_now["elo"]))

    cur = games[games["season"] == CURRENT_SEASON]
    base_wins = {t: 0 for t in teams["team_id"]}
    for g in cur.itertuples():
        winner = g.home_id if g.home_win else g.away_id
        base_wins[winner] += 1

    rem = list(zip(remaining["home_id"], remaining["away_id"]))
    p_home = np.array([elo_win_prob(elos[h], elos[a]) for h, a in rem])

    rng = np.random.default_rng(RNG_SEED)
    team_ids = teams["team_id"].tolist()
    counters = {
        k: {t: 0 for t in team_ids}
        for k in ("playoffs", "division", "bye", "pennant", "champion")
    }
    total_wins = {t: 0 for t in team_ids}

    for _ in range(N_SIMS):
        records = dict(base_wins)
        outcomes = rng.random(len(rem)) < p_home
        for (h, a), home_won in zip(rem, outcomes):
            records[h if home_won else a] += 1

        seeds = {
            lg: seed_league(records, teams, lg, rng)
            for lg in ("American League", "National League")
        }
        for lg, s in seeds.items():
            for t in s:
                counters["playoffs"][t] += 1
            for t in s[:3]:
                counters["division"][t] += 1
            for t in s[:2]:
                counters["bye"][t] += 1

        post = sim_postseason(seeds, elos, records, rng)
        for t in post["ws_teams"]:
            counters["pennant"][t] += 1
        counters["champion"][post["champion"]] += 1

        for t in team_ids:
            total_wins[t] += records[t]

    rows = []
    for t in team_ids:
        rows.append(
            {
                "team_id": t,
                "current_wins": base_wins[t],
                "proj_wins": total_wins[t] / N_SIMS,
                "make_playoffs": counters["playoffs"][t] / N_SIMS,
                "win_division": counters["division"][t] / N_SIMS,
                "first_round_bye": counters["bye"][t] / N_SIMS,
                "win_pennant": counters["pennant"][t] / N_SIMS,
                "win_world_series": counters["champion"][t] / N_SIMS,
            }
        )
    odds = pd.DataFrame(rows).merge(teams, on="team_id")
    odds = odds.sort_values("win_world_series", ascending=False)
    odds.to_csv(OUTPUTS / "playoff_odds_2026.csv", index=False)
    print(odds[["team_name", "proj_wins", "make_playoffs", "win_world_series"]].head(10).to_string(index=False))
    print(f"\nSaved playoff odds for {len(odds)} teams ({N_SIMS} sims).")


if __name__ == "__main__":
    main()
