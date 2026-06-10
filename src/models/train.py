"""Train and compare win-probability models.

Baseline:  raw Elo probability (no fitting).
Model 1:   Logistic regression on pre-game features.
Model 2:   XGBoost on the same features.
Model 3:   Ensemble — mean of Elo + LR, isotonic-calibrated on validation.
Model 4:   Poisson run-scoring model per side -> Skellam win probability
           (also yields expected runs, usable for totals later).

Time-based split — train 2015-2023, validate 2024, test 2025 + played 2026.
Saves models, metrics.json, and calibration data to outputs/.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import DATA_PROCESSED, OUTPUTS
from src.features import FEATURE_COLS

TRAIN_END = 2023
VAL_SEASON = 2024


def skellam_win_prob(lam_h: np.ndarray, lam_a: np.ndarray, max_runs: int = 26) -> np.ndarray:
    """P(home wins) from independent Poisson run distributions; ties (extra
    innings) are split in proportion to the win/loss ratio."""
    lam_h = np.asarray(lam_h, dtype=float)
    lam_a = np.asarray(lam_a, dtype=float)
    k = np.arange(max_runs)
    log_fact = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, max_runs)))])
    ph = np.exp(-lam_h[:, None] + k * np.log(lam_h[:, None]) - log_fact)
    pa = np.exp(-lam_a[:, None] + k * np.log(lam_a[:, None]) - log_fact)
    cum_pa = np.cumsum(pa, axis=1)
    cum_ph = np.cumsum(ph, axis=1)
    win = np.sum(ph[:, 1:] * cum_pa[:, :-1], axis=1)
    lose = np.sum(pa[:, 1:] * cum_ph[:, :-1], axis=1)
    return win + (1 - win - lose) * win / (win + lose)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def stack_features(p_elo: np.ndarray, p_lr: np.ndarray) -> np.ndarray:
    """Logit-space inputs for the ensemble stacker (smooth, 3 parameters —
    unlike isotonic regression it cannot overfit a small validation set)."""
    return np.column_stack([_logit(p_elo), _logit(p_lr)])


def _metrics(y_true, p) -> dict:
    return {
        "log_loss": float(log_loss(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
        "auc": float(roc_auc_score(y_true, p)),
        "accuracy": float(((p > 0.5).astype(int) == y_true).mean()),
        "n_games": int(len(y_true)),
    }


def main() -> None:
    df = pd.read_csv(DATA_PROCESSED / "features.csv", parse_dates=["date"])
    # only games where both teams have real rolling history
    df = df[(df["home_enough_history"] == 1) & (df["away_enough_history"] == 1)]

    train = df[df["season"] <= TRAIN_END]
    val = df[df["season"] == VAL_SEASON]
    test = df[df["season"] > VAL_SEASON]
    X_tr, y_tr = train[FEATURE_COLS], train["home_win"]
    X_va, y_va = val[FEATURE_COLS], val["home_win"]
    X_te, y_te = test[FEATURE_COLS], test["home_win"]
    print(f"train {len(train)} / val {len(val)} / test {len(test)} games")

    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    logreg.fit(X_tr, y_tr)

    # tuned on the validation season with early stopping
    best_xgb, best_ll = None, np.inf
    for depth in (2, 3, 4):
        for lr in (0.01, 0.03, 0.05):
            for mcw in (1, 10, 50):
                m = XGBClassifier(
                    n_estimators=3000,
                    max_depth=depth,
                    learning_rate=lr,
                    min_child_weight=mcw,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    early_stopping_rounds=100,
                    random_state=42,
                )
                m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                ll = log_loss(y_va, m.predict_proba(X_va)[:, 1])
                if ll < best_ll:
                    best_ll, best_xgb = ll, m

    # ensemble: logistic stack of Elo + LR probabilities, fit on the validation season
    stack = LogisticRegression(max_iter=1000)
    stack.fit(stack_features(val["elo_prob_home"], logreg.predict_proba(X_va)[:, 1]), y_va)
    p_ens_te = stack.predict_proba(
        stack_features(test["elo_prob_home"], logreg.predict_proba(X_te)[:, 1])
    )[:, 1]

    # Poisson runs-scored models -> Skellam win probability
    pois_h = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=500))
    pois_a = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=500))
    pois_h.fit(X_tr, train["home_score"])
    pois_a.fit(X_tr, train["away_score"])
    p_skl_te = skellam_win_prob(pois_h.predict(X_te), pois_a.predict(X_te))

    results = {
        "elo_baseline": _metrics(y_te, test["elo_prob_home"]),
        "logistic_regression": _metrics(y_te, logreg.predict_proba(X_te)[:, 1]),
        "xgboost": _metrics(y_te, best_xgb.predict_proba(X_te)[:, 1]),
        "ensemble": _metrics(y_te, p_ens_te),
        "skellam_poisson": _metrics(y_te, p_skl_te),
    }
    print(json.dumps(results, indent=2))

    best_name = min(
        (k for k in results), key=lambda k: results[k]["log_loss"]
    )
    results["best_model"] = best_name
    (OUTPUTS / "metrics.json").write_text(json.dumps(results, indent=2))

    joblib.dump(logreg, OUTPUTS / "model_logreg.joblib")
    joblib.dump(best_xgb, OUTPUTS / "model_xgb.joblib")
    joblib.dump(stack, OUTPUTS / "model_stack.joblib")
    joblib.dump(pois_h, OUTPUTS / "model_poisson_home.joblib")
    joblib.dump(pois_a, OUTPUTS / "model_poisson_away.joblib")

    # calibration data on the test set
    cal_rows = []
    for name, p in (
        ("elo_baseline", test["elo_prob_home"].to_numpy()),
        ("logistic_regression", logreg.predict_proba(X_te)[:, 1]),
        ("xgboost", best_xgb.predict_proba(X_te)[:, 1]),
        ("ensemble", p_ens_te),
        ("skellam_poisson", p_skl_te),
    ):
        frac_pos, mean_pred = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
        for fp, mp in zip(frac_pos, mean_pred):
            cal_rows.append({"model": name, "mean_predicted": mp, "fraction_won": fp})
    pd.DataFrame(cal_rows).to_csv(OUTPUTS / "calibration.csv", index=False)
    print(f"Best model on test log loss: {best_name}")


if __name__ == "__main__":
    main()
