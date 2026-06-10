"""Streamlit dashboard: 2026 MLB playoff odds, power ratings, model report,
and a head-to-head game predictor.

Run from the project root:
    streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.elo import elo_win_prob  # noqa: E402

OUTPUTS = ROOT / "outputs"
PROCESSED = ROOT / "data" / "processed"

st.set_page_config(page_title="MLB 2026 Playoff Predictor", page_icon="⚾", layout="wide")
st.title("⚾ MLB 2026 Playoff Predictor")
st.caption(
    "Elo + machine learning win-probability models, propagated to season-end "
    "playoff odds with 10,000 Monte Carlo simulations."
)

odds_path = OUTPUTS / "playoff_odds_2026.csv"
if not odds_path.exists():
    st.error("No simulation output found. Run `python run_pipeline.py` first.")
    st.stop()

odds = pd.read_csv(odds_path)
elo_df = pd.read_csv(OUTPUTS / "elo_current.csv")
snapshot = pd.read_csv(PROCESSED / "current_team_stats.csv")
metrics = json.loads((OUTPUTS / "metrics.json").read_text())
calibration = pd.read_csv(OUTPUTS / "calibration.csv")

tab_odds, tab_elo, tab_model, tab_predict = st.tabs(
    ["Playoff odds", "Power ratings", "Model report", "Game predictor"]
)

with tab_odds:
    league = st.radio("League", ["Both", "American League", "National League"], horizontal=True)
    view = odds if league == "Both" else odds[odds["league"] == league]
    pct_cols = ["make_playoffs", "win_division", "first_round_bye", "win_pennant", "win_world_series"]

    fig = px.bar(
        view.sort_values("make_playoffs"),
        x="make_playoffs",
        y="team_name",
        color="division",
        orientation="h",
        labels={"make_playoffs": "P(make playoffs)", "team_name": ""},
        height=700,
    )
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    table = view[["team_name", "league", "division", "current_wins", "proj_wins"] + pct_cols].copy()
    table["proj_wins"] = table["proj_wins"].round(1)
    st.dataframe(
        table.sort_values("win_world_series", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            c: st.column_config.ProgressColumn(c, format="percent", min_value=0, max_value=1)
            for c in pct_cols
        },
    )

with tab_elo:
    fig = px.bar(
        elo_df.sort_values("elo"),
        x="elo",
        y="team_name",
        color="league",
        orientation="h",
        labels={"elo": "Elo rating", "team_name": ""},
        height=700,
    )
    fig.update_xaxes(range=[elo_df["elo"].min() - 20, elo_df["elo"].max() + 20])
    st.plotly_chart(fig, use_container_width=True)

with tab_model:
    st.subheader("Test-set performance (2025 + 2026 to date)")
    rows = [
        {"model": k, **v}
        for k, v in metrics.items()
        if isinstance(v, dict)
    ]
    st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, use_container_width=True)
    st.markdown(f"**Best model by log loss:** `{metrics.get('best_model', 'n/a')}`")

    st.subheader("Calibration (quantile bins)")
    fig = px.line(
        calibration,
        x="mean_predicted",
        y="fraction_won",
        color="model",
        markers=True,
        labels={"mean_predicted": "Predicted home win prob", "fraction_won": "Actual home win rate"},
    )
    fig.add_shape(type="line", x0=0.3, y0=0.3, x1=0.7, y1=0.7, line=dict(dash="dash", color="gray"))
    st.plotly_chart(fig, use_container_width=True)

with tab_predict:
    st.subheader("Head-to-head win probability (Elo, with home advantage)")
    names = snapshot.sort_values("team_name")["team_name"].tolist()
    col1, col2 = st.columns(2)
    home = col1.selectbox("Home team", names, index=0)
    away = col2.selectbox("Away team", names, index=1)
    if home == away:
        st.warning("Pick two different teams.")
    else:
        h = snapshot[snapshot["team_name"] == home].iloc[0]
        a = snapshot[snapshot["team_name"] == away].iloc[0]
        p = elo_win_prob(h["elo"], a["elo"])
        c1, c2 = st.columns(2)
        c1.metric(f"{home} (home)", f"{p:.1%}")
        c2.metric(f"{away} (away)", f"{1 - p:.1%}")
        detail = pd.DataFrame(
            {
                "team": [home, away],
                "Elo": [round(h["elo"], 1), round(a["elo"], 1)],
                "last-30 win%": [f"{h['winpct_30']:.3f}", f"{a['winpct_30']:.3f}"],
                "last-30 run diff/gm": [round(h["rundiff_30"], 2), round(a["rundiff_30"], 2)],
                "season win%": [f"{h['season_winpct']:.3f}", f"{a['season_winpct']:.3f}"],
            }
        )
        st.dataframe(detail, hide_index=True, use_container_width=True)
