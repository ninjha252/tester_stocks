
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier

@dataclass
class WalkForwardConfig:
    test_window_days: int = 2
    min_train_days: int = 10
    embargo_minutes: int = 30
    features: list = None
    target_col: str = "y"

def _time_splits(idx: pd.DatetimeIndex, min_train_days: int, test_window_days: int, embargo_minutes: int):
    start = idx.min().normalize()
    end = idx.max().normalize()
    current_train_end = start + pd.Timedelta(days=min_train_days)
    folds = []
    while current_train_end + pd.Timedelta(days=test_window_days) <= end + pd.Timedelta(days=1):
        train_end = current_train_end
        test_start = train_end + pd.Timedelta(minutes=embargo_minutes)
        test_end = test_start + pd.Timedelta(days=test_window_days)
        train_idx = (idx < train_end)
        test_idx = (idx >= test_start) & (idx < test_end)
        if test_idx.sum() < 50:
            current_train_end += pd.Timedelta(days=test_window_days)
            continue
        folds.append((train_idx, test_idx))
        current_train_end += pd.Timedelta(days=test_window_days)
    return folds

def run_walkforward(df: pd.DataFrame, cfg: WalkForwardConfig):
    feats = cfg.features
    X_all = df[feats].replace([np.inf, -np.inf], np.nan).dropna()
    y_all = df.loc[X_all.index, cfg.target_col]

    idx = X_all.index
    folds = _time_splits(idx, cfg.min_train_days, cfg.test_window_days, cfg.embargo_minutes)

    oof_pred = pd.Series(index=idx, dtype=float)
    metrics = []
    models = []

    for i, (tr_mask, te_mask) in enumerate(folds, 1):
        X_tr, y_tr = X_all[tr_mask], y_all[tr_mask]
        X_te, y_te = X_all[te_mask], y_all[te_mask]
        if len(X_tr) == 0 or len(X_te) == 0:
            continue

        model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.02,
            max_depth=-1,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:,1]
        oof_pred.loc[X_te.index] = p

        y_hat = (p > 0.5).astype(int)
        metrics.append({
            "fold": i,
            "AUC": roc_auc_score(y_te, p),
            "Acc": accuracy_score(y_te, y_hat),
            "F1": f1_score(y_te, y_hat),
            "Precision": precision_score(y_te, y_hat),
            "Recall": recall_score(y_te, y_hat),
            "n_train": len(X_tr),
            "n_test": len(X_te),
        })
        models.append(model)

    return {"oof_pred": oof_pred, "metrics": pd.DataFrame(metrics), "models": models, "features": feats}
