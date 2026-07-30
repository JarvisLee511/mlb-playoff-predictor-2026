# Tests

```bash
pip install -r requirements-dev.txt
pytest -q          # 77 tests, ~2 s, no network and no data files
```

Every fixture is synthetic and hand-computable, so the suite tests *behaviour*
rather than re-checking numbers against a stored CSV. It runs in `daily.yml`
before the pipeline: there is no point publishing predictions from code that
just failed its own leakage checks.

| File | What it pins down |
|---|---|
| `test_elo.py` | The 400-point scale, zero-sum rating transfer, offseason regression, margin-of-victory, and that `*_elo_pre` is the rating *before* the game |
| `test_no_leakage.py` | Season-to-date stats exclude the current game, both as-of joins reject a same-day snapshot, nothing crosses a season boundary, bullpen fatigue counts only prior days |
| `test_skellam.py` | The hand-rolled convolution matches a brute-force Poisson double sum to 1e-9, plus range/monotonicity/symmetry |
| `test_head_to_head.py` | Models are scored on the same games, the delta-vs-Elo sign convention, paired-bootstrap reproducibility, Wilson intervals |
| `test_simulate.py` | 12 teams in, division winners seeded 1–3 regardless of record, byes, and that the higher seed holds home field in every matchup |

## Are these tests worth anything?

A suite that passes proves nothing on its own, so the invariants were checked by
mutation: plant a plausible bug, confirm the suite goes red.

| Planted bug | Caught by |
|---|---|
| As-of join accepts a same-day starter snapshot | 3 tests |
| As-of join accepts a same-day bullpen snapshot | 2 tests |
| Season-to-date includes the current game | 2 tests |
| Bullpen-fatigue keys inherit the caller's dtype | 1 test |
| Offseason regression removed | 1 test |
| Rating transfer no longer zero-sum | 3 tests |
| Margin of victory ignored | 2 tests |
| Delta-vs-Elo sign flipped | 1 test |
| Models scored on their own rows again | 4 tests |
| Skellam forgets to split ties | 8 tests |
| Wild-card home field given to the lower seed | 1 test |
| Division-series home field given to the lower seed | 1 test |

12 of 12. Two of those tests exist *because* of this exercise: the bullpen as-of
join was unguarded, and the first version of the home-field test summed both
wild-card matchups, which let one reversed series be cancelled out by the other.

The exercise also turned up a real fragility. `bullpen_fatigue` looked its keys
up by formatted date string while `build_bullpen_snapshots` inherited the key
type from whatever the caller passed in. Production happened to be correct
because `gamelogs.load()` does not parse dates — but adding `parse_dates=["date"]`,
an entirely reasonable change, would have made every bullpen look fully rested
with no error anywhere. The keys are now normalised at construction; the
replacement was verified to produce a byte-identical dict on all 52,751 real
team-days before the fix went in.
