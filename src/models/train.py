"""Train and compare win-probability models.

Baseline: raw Elo probability (no fitting).
Model 1:  Logistic regression on pre-game features.
Model 2:  XGBoost on the same features.

Time-based split — train 2015-2023, validate 2024, test 2025 + played 2026.
Saves the best model, metrics.json, and calibration data to outputs/.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import DATA_PROCESSED, OUTPUTS
from src.features import FEATURE_COLS

TRAIN_END = 2023
VAL_SEASON = 2024


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

    # light tuning on the validation season
    best_xgb, best_ll = None, np.inf
    for depth in (2, 3, 4):
        for lr in (0.02, 0.05):
            m = XGBClassifier(
                n_estimators=400,
                max_depth=depth,
                learning_rate=lr,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
            )
            m.fit(X_tr, y_tr)
            ll = log_loss(y_va, m.predict_proba(X_va)[:, 1])
            if ll < best_ll:
                best_ll, best_xgb = ll, m

    results = {
        "elo_baseline": _metrics(y_te, test["elo_prob_home"]),
        "logistic_regression": _metrics(y_te, logreg.predict_proba(X_te)[:, 1]),
        "xgboost": _metrics(y_te, best_xgb.predict_proba(X_te)[:, 1]),
    }
    print(json.dumps(results, indent=2))

    best_name = min(results, key=lambda k: results[k]["log_loss"])
    results["best_model"] = best_name
    (OUTPUTS / "metrics.json").write_text(json.dumps(results, indent=2))

    joblib.dump(logreg, OUTPUTS / "model_logreg.joblib")
    joblib.dump(best_xgb, OUTPUTS / "model_xgb.joblib")

    # calibration data on the test set for all three models
    cal_rows = []
    for name, p in (
        ("elo_baseline", test["elo_prob_home"].to_numpy()),
        ("logistic_regression", logreg.predict_proba(X_te)[:, 1]),
        ("xgboost", best_xgb.predict_proba(X_te)[:, 1]),
    ):
        frac_pos, mean_pred = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
        for fp, mp in zip(frac_pos, mean_pred):
            cal_rows.append({"model": name, "mean_predicted": mp, "fraction_won": fp})
    pd.DataFrame(cal_rows).to_csv(OUTPUTS / "calibration.csv", index=False)
    print(f"Best model on test log loss: {best_name}")


if __name__ == "__main__":
    main()
