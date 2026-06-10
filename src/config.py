"""Central paths and constants for the MLB 2026 playoff prediction project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

for _d in (DATA_RAW, DATA_PROCESSED, OUTPUTS):
    _d.mkdir(parents=True, exist_ok=True)

# Seasons used for training history. 2020 is the shortened COVID season;
# it stays in (Elo handles it) but is excluded from rolling-window features
# crossing the long gap.
FIRST_SEASON = 2015
CURRENT_SEASON = 2026

# Elo parameters — grid-searched on 2023-24 validation log loss (0.68032 vs
# 0.68084 with the FiveThirtyEight-style defaults K=4 / 24 / 2/3). The lower
# home advantage reflects MLB's shrinking home edge in recent seasons.
ELO_START = 1500.0
ELO_K = 3.0
ELO_HOME_ADV = 16.0          # home advantage in Elo points
ELO_SEASON_CARRYOVER = 0.55  # share of rating kept across the offseason

# Simulation
N_SIMS = 10_000
RNG_SEED = 511
