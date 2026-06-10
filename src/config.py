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

# Elo parameters (FiveThirtyEight-style, tuned for MLB's low K)
ELO_START = 1500.0
ELO_K = 4.0
ELO_HOME_ADV = 24.0          # home advantage in Elo points
ELO_SEASON_CARRYOVER = 2 / 3  # regress 1/3 of the way back to 1500 each offseason

# Simulation
N_SIMS = 10_000
RNG_SEED = 511
