"""The yardsticks the tracker publishes alongside its accuracy.

The site's central claim is that a mid-fifties accuracy is near the public-data
ceiling rather than a broken model, and it argues that by placing each model
between two floors (always-pick-home, coin flip) and the closing line. That claim
is only as good as the arithmetic here: if `skill_vs_ceiling` inverted its sign, or
the always-home rate were read off a different sample than the one shown, the page
would state the opposite with equal confidence.
"""
import json

import numpy as np
import pandas as pd
import pytest

from src import site_export
from src.site_export import (MARKET_CEILING_LOG_LOSS, MIN_RELIABLE_N,
                             export_accuracy)

COINFLIP_LL = float(np.log(2))


@pytest.fixture
def exported(tmp_path, monkeypatch):
    """Run the real exporter into a temp dir and hand back the parsed JSON."""
    monkeypatch.setattr(site_export, "SITE_DATA", tmp_path)

    def run(log: pd.DataFrame) -> dict:
        export_accuracy(log)
        return json.loads((tmp_path / "accuracy.json").read_text(encoding="utf-8"))

    return run


def make_log(n=400, home_win_rate=0.54, p_home=None, target_ll=None, seed=0):
    """A scored log with an exactly known home-win rate.

    `p_home` gives every game the same probability, so log loss is hand-computable.
    `target_ll` instead builds a model with real discrimination whose log loss is
    exactly that value — needed because a *constant* probability cannot get below
    the sample's entropy (0.6899 at a 54% base rate), so the market ceiling of
    0.655 is unreachable without a model that actually separates the outcomes.
    """
    rng = np.random.default_rng(seed)
    wins = int(round(n * home_win_rate))
    y = np.array([1] * wins + [0] * (n - wins))
    rng.shuffle(y)
    if target_ll is not None:
        hit = float(np.exp(-target_ll))       # -ln(p) == target_ll on every game
        p = np.where(y == 1, hit, 1 - hit)
    else:
        p = np.full(n, 0.5 if p_home is None else p_home)
    df = pd.DataFrame({
        "date": pd.date_range("2026-04-01", periods=n, freq="h").astype(str),
        "game_time_et": "7:05 PM",
        "status": "final",
        "home_win": y,
        "home_name": "Home", "away_name": "Away",
        "home_score": 4, "away_score": 3,
    })
    for col in ("p_home_elo", "p_home_lr", "p_home_xgb", "p_home_ens", "p_home_skl"):
        df[col] = p
    return df


class TestBaselines:
    def test_always_home_is_the_measured_rate_not_a_constant(self, exported):
        """The home edge drifts season to season (53.9% in June 2026, 51.4% by
        August), so this floor has to be measured, never hardcoded."""
        d = exported(make_log(n=500, home_win_rate=0.512))
        assert d["baselines"]["always_home_accuracy"] == pytest.approx(0.512, abs=5e-4)
        assert d["baselines"]["home_base_rate"] == d["baselines"]["always_home_accuracy"]

    def test_coinflip_log_loss_is_ln2(self, exported):
        d = exported(make_log())
        assert d["baselines"]["coinflip_log_loss"] == pytest.approx(COINFLIP_LL, abs=5e-5)
        assert d["baselines"]["coinflip_accuracy"] == 0.5

    def test_ceiling_sits_above_the_floors(self, exported):
        b = exported(make_log())["baselines"]
        assert b["market_ceiling_accuracy"] > b["always_home_accuracy"]
        assert b["market_ceiling_log_loss"] < b["coinflip_log_loss"]

    def test_absent_when_nothing_is_scored(self, exported):
        """No games means no floors to quote; the front end hides the line rather
        than printing an invented one."""
        d = exported(make_log(n=10).assign(status="scheduled"))
        assert "baselines" not in d
        assert d["n_scored"] == 0


class TestSkillVsCeiling:
    def test_a_coin_flip_model_scores_zero(self, exported):
        """p=0.5 on every game IS the coin flip, so it must land on the floor."""
        d = exported(make_log(p_home=0.5))
        assert d["summary"]["ens"]["log_loss"] == pytest.approx(COINFLIP_LL, abs=5e-5)
        assert d["summary"]["ens"]["skill_vs_ceiling"] == pytest.approx(0.0, abs=2e-3)

    def test_a_market_grade_model_scores_one(self, exported):
        """A model performing exactly at the closing line must read 1.0 — this
        pins the far endpoint of the mapping, the coin flip pins the near one."""
        d = exported(make_log(n=1000, target_ll=MARKET_CEILING_LOG_LOSS))
        assert d["summary"]["ens"]["log_loss"] == pytest.approx(MARKET_CEILING_LOG_LOSS, abs=1e-3)
        assert d["summary"]["ens"]["skill_vs_ceiling"] == pytest.approx(1.0, abs=0.01)

    def test_worse_than_a_coin_flip_goes_negative(self, exported):
        """Skellam ran at -0.024 on the live log; the sign must survive so a bad
        model cannot read as a good one."""
        d = exported(make_log(n=600, home_win_rate=0.54, p_home=0.30))
        assert d["summary"]["ens"]["log_loss"] > COINFLIP_LL
        assert d["summary"]["ens"]["skill_vs_ceiling"] < 0

    def test_better_log_loss_always_means_higher_skill(self, exported):
        """Monotonicity — the score is only a re-expression of log loss, so the two
        rankings can never disagree. (Ordering is taken from the measured log loss,
        not from p: against a 54% base rate, loss falls until p=0.54 and rises
        after, so p itself is not a proxy for quality.)"""
        rows = [exported(make_log(n=600, home_win_rate=0.54, p_home=p))["summary"]["ens"]
                for p in (0.50, 0.53, 0.56, 0.62)]
        by_loss = sorted((r["log_loss"], r["skill_vs_ceiling"]) for r in rows)
        skills = [s for _, s in by_loss]
        assert skills == sorted(skills, reverse=True)
        assert len(set(skills)) == len(skills), "distinct losses must give distinct scores"


def test_reliability_threshold_still_gates_the_framing(exported):
    """The framing line replaces the small-sample warning, so `reliable` has to
    stay honest — the two must never both be showing."""
    assert exported(make_log(n=MIN_RELIABLE_N - 1))["reliable"] is False
    assert exported(make_log(n=MIN_RELIABLE_N))["reliable"] is True
