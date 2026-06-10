"""Generate and execute the analysis notebook. Run from project root:
    .venv\\Scripts\\python.exe notebooks\\_build.py
"""
import nbformat as nbf
from nbclient import NotebookClient

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md(
        "# MLB 2026 Playoff Prediction — Analysis\n\n"
        "This notebook walks through the modeling results: data overview, Elo power "
        "ratings, win-probability model comparison (Elo baseline vs logistic regression "
        "vs XGBoost), calibration, and the simulated 2026 playoff odds.\n\n"
        "Run `python run_pipeline.py` first — this notebook reads its outputs."
    ),
    code(
        "import json\n"
        "from pathlib import Path\n\n"
        "import matplotlib.pyplot as plt\n"
        "import pandas as pd\n\n"
        "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "OUT = ROOT / 'outputs'\n\n"
        "games = pd.read_csv(ROOT / 'data/raw/games.csv', parse_dates=['date'])\n"
        "odds = pd.read_csv(OUT / 'playoff_odds_2026.csv')\n"
        "elo = pd.read_csv(OUT / 'elo_current.csv')\n"
        "metrics = json.loads((OUT / 'metrics.json').read_text())\n"
        "calibration = pd.read_csv(OUT / 'calibration.csv')\n"
        "print(f'{len(games):,} completed games, {games.season.min()}-{games.season.max()}')"
    ),
    md(
        "## 1. Data overview\n\n"
        "Completed regular-season games per season, and the league-wide home win rate — "
        "the base rate any model has to beat."
    ),
    code(
        "by_season = games.groupby('season').agg(games_played=('game_id', 'count'),\n"
        "                                        home_win_rate=('home_win', 'mean')).round(3)\n"
        "by_season"
    ),
    md("## 2. Current Elo power ratings (top/bottom 10)"),
    code(
        "ranked = elo.sort_values('elo', ascending=False)\n"
        "show = pd.concat([ranked.head(10), ranked.tail(10)])\n"
        "colors = ['#2a9d8f' if l == 'American League' else '#e76f51' for l in show.league]\n"
        "fig, ax = plt.subplots(figsize=(9, 7))\n"
        "ax.barh(show.team_name[::-1], show.elo[::-1], color=colors[::-1])\n"
        "ax.axvline(1500, color='gray', ls='--', lw=1)\n"
        "ax.set_xlabel('Elo rating')\n"
        "ax.set_title('Elo power ratings — top 10 and bottom 10 (teal=AL, orange=NL)')\n"
        "ax.set_xlim(show.elo.min() - 20, show.elo.max() + 20)\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "## 3. Model comparison\n\n"
        "Held-out test set = 2025 + played 2026 games. Log loss and Brier score are the "
        "headline metrics — for win probabilities, calibration matters more than accuracy."
    ),
    code(
        "rows = [{'model': k, **v} for k, v in metrics.items() if isinstance(v, dict)]\n"
        "pd.DataFrame(rows).set_index('model').round(4)"
    ),
    code(
        "fig, ax = plt.subplots(figsize=(7, 6))\n"
        "for name, grp in calibration.groupby('model'):\n"
        "    ax.plot(grp.mean_predicted, grp.fraction_won, marker='o', label=name)\n"
        "lims = [calibration.mean_predicted.min() - .02, calibration.mean_predicted.max() + .02]\n"
        "ax.plot(lims, lims, 'k--', lw=1, label='perfect calibration')\n"
        "ax.set_xlabel('Predicted home win probability')\n"
        "ax.set_ylabel('Actual home win rate')\n"
        "ax.set_title('Calibration on the test set (quantile bins)')\n"
        "ax.legend(); plt.tight_layout(); plt.show()"
    ),
    md(
        "## 4. Simulated 2026 playoff odds\n\n"
        "10,000 Monte Carlo simulations of the remaining schedule + full postseason bracket."
    ),
    code(
        "cols = ['team_name', 'league', 'division', 'current_wins', 'proj_wins',\n"
        "        'make_playoffs', 'win_division', 'win_pennant', 'win_world_series']\n"
        "styled = odds[cols].sort_values('win_world_series', ascending=False).head(15).copy()\n"
        "styled['proj_wins'] = styled['proj_wins'].round(1)\n"
        "for c in ['make_playoffs', 'win_division', 'win_pennant', 'win_world_series']:\n"
        "    styled[c] = (styled[c] * 100).round(1).astype(str) + '%'\n"
        "styled.reset_index(drop=True)"
    ),
    code(
        "top = odds.sort_values('make_playoffs', ascending=False).head(16)\n"
        "fig, ax = plt.subplots(figsize=(9, 7))\n"
        "colors = ['#2a9d8f' if l == 'American League' else '#e76f51' for l in top.league]\n"
        "ax.barh(top.team_name[::-1], top.make_playoffs[::-1], color=colors[::-1])\n"
        "ax.set_xlabel('P(make playoffs)')\n"
        "ax.set_xlim(0, 1)\n"
        "ax.set_title('2026 playoff probability — top 16 teams (teal=AL, orange=NL)')\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "## 5. Takeaways\n\n"
        "- Baseball single games are near coin flips: a well-calibrated ~0.58 ceiling on "
        "win probability is expected, so **log loss vs the Elo baseline** is the fair test "
        "of whether the ML features add signal.\n"
        "- The Monte Carlo layer converts game-level edges into season-level statements "
        "(playoff %, World Series %) — small per-game differences compound over ~80 "
        "remaining games.\n"
        "- Next iteration: add probable starting pitchers (available from the same API) "
        "and re-run the comparison."
    ),
]

nb.cells = cells
nb.metadata.kernelspec = {"name": "python3", "display_name": "Python 3", "language": "python"}

client = NotebookClient(nb, timeout=300, kernel_name="python3")
client.execute()
out_path = "notebooks/analysis.ipynb"
nbf.write(nb, out_path)
print(f"Wrote and executed {out_path}")
