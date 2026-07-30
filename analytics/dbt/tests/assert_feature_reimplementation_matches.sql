-- Guards the SQL feature reimplementation: fails if any rebuilt column agrees with the
-- Python pipeline (features.csv) on fewer than 99% of games.
--
-- Observed on a full build: every season-to-date rate stat (ops/era/fip/whip/pyth/
-- off_bbk), both starter snapshots, park_hfa, season win% and rest days match to
-- floating-point precision (<= 1e-15). The only residual is on the 30-game rolling
-- win%/run-diff (~99.9% agreement): a handful of games where the rolling-window edge
-- includes a different single game around doubleheaders / schedule gaps. Per-team runs
-- match the final-score table exactly, so this is a window-composition edge, not a data
-- discrepancy — well within tolerance for a validation harness.
select feature, pct_within_tol
from {{ ref('mart_feature_validation') }}
where pct_within_tol < 0.99
