"""Throwaway: does lineup_woba_diff actually improve the model?

Same data, same split, toggle the one feature. Reports test log loss / AUC /
Brier with and without it for LR, XGB, and the Elo+LR ensemble stack.
Run AFTER features.csv has been rebuilt with lineups.
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import DATA_PROCESSED
from src.features import FEATURE_COLS
from src.models.train import TRAIN_END, VAL_SEASON, stack_features

LINEUP = "lineup_woba_diff"


def fit_eval(cols, tr, va, te):
    Xtr, ytr = tr[cols], tr["home_win"]
    Xva, yva = va[cols], va["home_win"]
    Xte, yte = te[cols], te["home_win"]

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
    p_lr = lr.predict_proba(Xte)[:, 1]

    xgb = XGBClassifier(n_estimators=2000, max_depth=3, learning_rate=0.03,
                        min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                        eval_metric="logloss", early_stopping_rounds=100, random_state=42)
    xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    p_xgb = xgb.predict_proba(Xte)[:, 1]

    stack = LogisticRegression(max_iter=1000)
    stack.fit(stack_features(va["elo_prob_home"], lr.predict_proba(Xva)[:, 1]), yva)
    p_ens = stack.predict_proba(stack_features(te["elo_prob_home"], p_lr))[:, 1]

    def m(p):
        return (log_loss(yte, p), roc_auc_score(yte, p), brier_score_loss(yte, p))
    coef = dict(zip(cols, lr.named_steps["logisticregression"].coef_[0]))
    return {"lr": m(p_lr), "xgb": m(p_xgb), "ens": m(p_ens),
            "lr_lineup_coef": coef.get(LINEUP)}


def main():
    df = pd.read_csv(DATA_PROCESSED / "features.csv", parse_dates=["date"])
    df = df[(df["home_enough_history"] == 1) & (df["away_enough_history"] == 1)]
    tr = df[df["season"] <= TRAIN_END]
    va = df[df["season"] == VAL_SEASON]
    te = df[df["season"] > VAL_SEASON]
    print(f"train {len(tr)} / val {len(va)} / test {len(te)}")
    cov = (te[LINEUP] != 0).mean()
    print(f"test-set games with non-zero {LINEUP}: {cov:.1%}\n")

    without = fit_eval([c for c in FEATURE_COLS if c != LINEUP], tr, va, te)
    with_ = fit_eval(FEATURE_COLS, tr, va, te)

    print(f"{'model':>6} | {'logloss WITHOUT':>16} {'logloss WITH':>13} {'Δ':>9} | {'AUC w/o→w':>16}")
    for k in ("lr", "xgb", "ens"):
        ll0, auc0, _ = without[k]
        ll1, auc1, _ = with_[k]
        print(f"{k:>6} | {ll0:16.5f} {ll1:13.5f} {ll1-ll0:+9.5f} | {auc0:.4f}->{auc1:.4f}")
    print(f"\nLR coefficient on {LINEUP}: {with_['lr_lineup_coef']:+.4f} "
          f"(standardized; sign should be positive — better lineup -> home win)")


if __name__ == "__main__":
    main()
