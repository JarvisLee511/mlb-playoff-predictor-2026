"""The Poisson-Skellam win probability.

`skellam_win_prob` is hand-rolled — it builds both run distributions in log
space and convolves them with cumulative sums — so it gets checked against a
brute-force double sum as well as against its own invariants.
"""
import math

import numpy as np
import pytest

from src.models.train import skellam_win_prob


def brute_force(lam_h: float, lam_a: float, max_runs: int = 40) -> float:
    """P(home wins), ties split in proportion to win/loss, written the slow
    obvious way: two nested loops over a Poisson pmf."""
    def pmf(k, lam):
        return math.exp(-lam) * lam**k / math.factorial(k)

    win = lose = tie = 0.0
    for h in range(max_runs):
        for a in range(max_runs):
            p = pmf(h, lam_h) * pmf(a, lam_a)
            if h > a:
                win += p
            elif a > h:
                lose += p
            else:
                tie += p
    return win + tie * win / (win + lose)


class TestSkellam:
    @pytest.mark.parametrize("lam", [3.0, 4.5, 6.0])
    def test_equal_scoring_rates_are_a_coin_flip(self, lam):
        assert skellam_win_prob(np.array([lam]), np.array([lam]))[0] == pytest.approx(0.5)

    @pytest.mark.parametrize("lam_h,lam_a", [(4.0, 4.0), (5.5, 3.5), (3.0, 6.0), (2.0, 2.5)])
    def test_matches_a_brute_force_double_sum(self, lam_h, lam_a):
        got = skellam_win_prob(np.array([lam_h]), np.array([lam_a]))[0]
        assert got == pytest.approx(brute_force(lam_h, lam_a), abs=1e-9)

    def test_probabilities_stay_in_range(self):
        lam_h = np.array([0.5, 2.0, 4.5, 9.0, 15.0])
        lam_a = np.array([9.0, 4.5, 4.5, 2.0, 0.5])
        p = skellam_win_prob(lam_h, lam_a)
        assert ((p > 0) & (p < 1)).all()

    def test_monotone_in_the_home_scoring_rate(self):
        lam_h = np.linspace(2.0, 8.0, 13)
        p = skellam_win_prob(lam_h, np.full_like(lam_h, 4.5))
        assert (np.diff(p) > 0).all()

    def test_reversing_the_teams_reverses_the_probability(self):
        lam_h, lam_a = np.array([5.5]), np.array([3.5])
        forward = skellam_win_prob(lam_h, lam_a)[0]
        backward = skellam_win_prob(lam_a, lam_h)[0]
        assert forward + backward == pytest.approx(1.0)

    def test_vectorises_elementwise(self):
        lam_h = np.array([4.0, 5.5, 3.0])
        lam_a = np.array([4.0, 3.5, 6.0])
        together = skellam_win_prob(lam_h, lam_a)
        apart = [skellam_win_prob(lam_h[i:i + 1], lam_a[i:i + 1])[0] for i in range(3)]
        assert together == pytest.approx(np.array(apart))
