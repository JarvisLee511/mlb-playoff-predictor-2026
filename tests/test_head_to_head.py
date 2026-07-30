"""The like-for-like model comparison the site publishes.

This block is what stops the tracker from ranking models on different samples,
and its paired-bootstrap interval is the basis for the README's claim that the
ML features are indistinguishable from Elo. If the sign convention or the
pairing ever inverted, the site would confidently state the opposite.
"""
import numpy as np
import pandas as pd
import pytest

from src.site_export import _head_to_head, _wilson_ci


def make_log(n=400, seed=0, ens_missing=0, perfect_lr=False):
    """A synthetic scored log. Elo is a mild edge; the others are copies of it
    unless a test asks for something different."""
    rng = np.random.default_rng(seed)
    p_elo = rng.uniform(0.35, 0.65, n)
    y = (rng.random(n) < p_elo).astype(int)

    df = pd.DataFrame({
        "date": pd.date_range("2026-04-01", periods=n, freq="h").astype(str),
        "status": "final",
        "home_win": y,
        "p_home_elo": p_elo,
        "p_home_lr": np.where(y == 1, 0.99, 0.01) if perfect_lr else p_elo,
        "p_home_xgb": p_elo,
        "p_home_ens": p_elo,
        "p_home_skl": p_elo,
    })
    if ens_missing:
        df.loc[df.index[:ens_missing], "p_home_ens"] = np.nan
    return df


class TestCommonSubset:
    def test_scores_only_games_every_model_predicted(self):
        h = _head_to_head(make_log(n=400, ens_missing=50))
        assert h["n"] == 350

    def test_all_models_report_the_same_sample(self):
        """The whole point: no model may be scored on rows another one missed."""
        h = _head_to_head(make_log(n=400, ens_missing=50))
        # identical predictions on an identical sample -> identical metrics
        losses = {m["log_loss"] for m in h["models"].values()}
        assert len(losses) == 1

    def test_returns_empty_when_the_overlap_is_too_small(self):
        assert _head_to_head(make_log(n=40, ens_missing=20)) == {}

    def test_home_win_rate_is_measured_on_that_subset(self):
        df = make_log(n=400, ens_missing=50)
        h = _head_to_head(df)
        expected = df.dropna(subset=["p_home_ens"])["home_win"].mean()
        assert h["home_win_rate"] == pytest.approx(expected, abs=5e-5)


class TestDeltaVsElo:
    def test_elo_is_the_reference_and_carries_no_delta(self):
        h = _head_to_head(make_log())
        assert "delta_vs_elo" not in h["models"]["elo"]

    def test_a_model_identical_to_elo_has_a_zero_gap(self):
        h = _head_to_head(make_log())
        lr = h["models"]["lr"]
        assert lr["delta_vs_elo"] == pytest.approx(0.0, abs=1e-9)
        assert lr["delta_ci"][0] <= 0 <= lr["delta_ci"][1]

    def test_a_better_model_gets_a_negative_gap(self):
        """Sign convention: negative beats Elo. An almost-perfect model must
        land far below zero with the interval clear of it."""
        h = _head_to_head(make_log(perfect_lr=True))
        lr = h["models"]["lr"]
        assert lr["delta_vs_elo"] < -0.5
        assert lr["delta_ci"][1] < 0
        assert lr["p_beats_elo"] == pytest.approx(1.0)

    def test_interval_is_ordered_and_probability_is_a_probability(self):
        h = _head_to_head(make_log(seed=3))
        for name, m in h["models"].items():
            if name == "elo":
                continue
            lo, hi = m["delta_ci"]
            assert lo <= hi
            assert 0.0 <= m["p_beats_elo"] <= 1.0
            assert lo <= m["delta_vs_elo"] <= hi

    def test_bootstrap_is_reproducible(self):
        df = make_log(seed=7)
        assert _head_to_head(df, seed=42) == _head_to_head(df, seed=42)

    def test_bootstrap_seed_actually_changes_the_resampling(self):
        df = make_log(seed=7, perfect_lr=True)
        a = _head_to_head(df, seed=1)["models"]["lr"]["delta_ci"]
        b = _head_to_head(df, seed=2)["models"]["lr"]["delta_ci"]
        assert a != b

    def test_metrics_are_internally_consistent(self):
        h = _head_to_head(make_log(seed=11))
        for m in h["models"].values():
            assert 0.0 <= m["accuracy"] <= 1.0
            assert 0.0 <= m["brier"] <= 1.0
            assert m["log_loss"] > 0.0


class TestWilsonInterval:
    def test_brackets_the_point_estimate(self):
        lo, hi = _wilson_ci(300, 600)
        assert lo < 0.5 < hi

    def test_narrows_as_the_sample_grows(self):
        small = _wilson_ci(50, 100)
        big = _wilson_ci(5000, 10000)
        assert (big[1] - big[0]) < (small[1] - small[0])

    def test_stays_inside_zero_and_one_at_the_extremes(self):
        assert _wilson_ci(0, 10)[0] >= 0.0
        assert _wilson_ci(10, 10)[1] <= 1.0

    def test_empty_sample_is_maximally_uncertain(self):
        assert _wilson_ci(0, 0) == [0.0, 1.0]
