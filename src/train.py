"""Train, compare and tune churn models.

Steps
-----
1. Stratified 80/20 train/test split (test set is never touched until evaluation).
2. 5-fold stratified CV on the training set for Logistic Regression,
   Random Forest and LightGBM.
3. Tune the best model with Optuna (maximising CV ROC-AUC).
4. Platt-calibrate the probabilities on out-of-fold predictions.
5. Save the calibrated model, out-of-fold predictions
   for threshold selection, and the held-out test set.

Class imbalance (26.5% churners)
--------------------------------
We use **class weights** (``class_weight="balanced"`` / ``scale_pos_weight``)
instead of SMOTE because:
* it needs no synthetic rows, so there is no risk of leaking oversampled
  points across CV folds;
* all three models support it natively, keeping the comparison fair;
* the decision threshold is tuned afterwards with a cost model anyway, so
  choosing *where* to cut matters more than resampling.

Class weights distort probabilities (they over-predict churn), so the final
model is Platt-calibrated on out-of-fold predictions before it is saved.
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import joblib
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import MODEL_FILE, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, SEED, TARGET  # noqa: E402
from src.data_prep import clean_data, load_raw  # noqa: E402
from src.features import FEATURES, add_features, build_preprocessor  # noqa: E402
from src.model import CalibratedChurnModel  # noqa: E402

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
SCORING = ["roc_auc", "average_precision", "f1", "recall", "precision"]


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load raw data, clean it, add features and return ``X, y``."""
    df = add_features(clean_data(load_raw()))
    return df[FEATURES], df[TARGET]


def candidate_models(pos_weight: float) -> dict[str, Pipeline]:
    """Return the three baseline pipelines, each with class-imbalance handling."""
    return {
        "Logistic Regression": Pipeline([
            ("prep", build_preprocessor(scale=True)),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
        ]),
        "Random Forest": Pipeline([
            ("prep", build_preprocessor(scale=False)),
            ("model", RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                             class_weight="balanced", n_jobs=-1, random_state=SEED)),
        ]),
        "LightGBM": Pipeline([
            ("prep", build_preprocessor(scale=False)),
            ("model", LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                     scale_pos_weight=pos_weight, random_state=SEED, verbose=-1)),
        ]),
    }


def compare_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Cross-validate every candidate and return a sorted comparison table."""
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    rows = []
    for name, pipe in candidate_models(pos_weight).items():
        cv = cross_validate(pipe, X, y, cv=CV, scoring=SCORING, n_jobs=1)
        row = {"model": name}
        for m in SCORING:
            row[m] = cv[f"test_{m}"].mean()
            row[f"{m}_std"] = cv[f"test_{m}"].std()
        rows.append(row)
        logger.info("%-20s ROC-AUC %.4f ± %.4f | recall %.3f | F1 %.3f", name,
                    row["roc_auc"], row["roc_auc_std"], row["recall"], row["f1"])
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def _suggest(trial: optuna.Trial, name: str, pos_weight: float) -> Pipeline:
    """Build a pipeline with Optuna-suggested hyper-parameters for ``name``."""
    if name == "Logistic Regression":
        model = LogisticRegression(
            C=trial.suggest_float("C", 1e-3, 10, log=True),
            l1_ratio=trial.suggest_categorical("l1_ratio", [0.0, 1.0]),
            solver="saga", max_iter=5000, class_weight="balanced", random_state=SEED)
        return Pipeline([("prep", build_preprocessor(scale=True)), ("model", model)])
    if name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 200, 600, step=100),
            max_depth=trial.suggest_int("max_depth", 4, 16),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
            max_features=trial.suggest_float("max_features", 0.2, 0.8),
            class_weight="balanced", n_jobs=-1, random_state=SEED)
        return Pipeline([("prep", build_preprocessor(scale=False)), ("model", model)])
    model = LGBMClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 4, 48),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 120),
        subsample=trial.suggest_float("subsample", 0.6, 1.0), subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        scale_pos_weight=pos_weight, random_state=SEED, verbose=-1)
    return Pipeline([("prep", build_preprocessor(scale=False)), ("model", model)])


def tune(name: str, X: pd.DataFrame, y: pd.Series, n_trials: int = 40) -> tuple[Pipeline, dict, float]:
    """Tune model ``name`` with Optuna; return the best pipeline, params and CV AUC."""
    pos_weight = float((y == 0).sum() / (y == 1).sum())

    def objective(trial: optuna.Trial) -> float:
        pipe = _suggest(trial, name, pos_weight)
        return cross_validate(pipe, X, y, cv=CV, scoring="roc_auc")["test_score"].mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    best = _suggest(optuna.trial.FixedTrial(study.best_params), name, pos_weight)
    logger.info("Optuna best CV ROC-AUC for %s: %.4f", name, study.best_value)
    return best, study.best_params, float(study.best_value)


def main(n_trials: int = 40) -> None:
    """Run the full training pipeline and save artefacts."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    X, y = load_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    logger.info("Train %d | Test %d | churn rate %.1f%%", len(X_tr), len(X_te), 100 * y.mean())

    comparison = compare_models(X_tr, y_tr)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    best_name = comparison.loc[0, "model"]
    logger.info("Best baseline: %s -> tuning with Optuna (%d trials)", best_name, n_trials)

    pipe, params, tuned_auc = tune(best_name, X_tr, y_tr, n_trials)
    oof = cross_val_predict(pipe, X_tr, y_tr, cv=CV, method="predict_proba")[:, 1]
    pipe.fit(X_tr, y_tr)

    # Platt-calibrate on out-of-fold scores so probabilities are honest (see src/model.py).
    model = CalibratedChurnModel.from_oof(pipe, y_tr.to_numpy(), oof)
    oof_cal = model.calibrate(oof)
    logger.info("Calibration: mean OOF prob %.3f -> %.3f (actual churn rate %.3f)",
                oof.mean(), oof_cal.mean(), y_tr.mean())

    joblib.dump(model, MODEL_FILE)
    pd.DataFrame({"y": y_tr.values, "prob": oof_cal, "prob_raw": oof}).to_csv(PROCESSED_DIR / "oof_predictions.csv", index=False)
    X_te.assign(**{TARGET: y_te.values}).to_csv(PROCESSED_DIR / "test_set.csv", index=True)
    (MODELS_DIR / "model_info.json").write_text(json.dumps({
        "model": best_name, "best_params": params, "cv_roc_auc_tuned": round(tuned_auc, 4),
        "baseline_comparison": comparison[["model", "roc_auc", "roc_auc_std", "recall", "f1"]]
        .round(4).to_dict(orient="records"), "calibration": {"method": "platt", "a": round(model.a, 4), "b": round(model.b, 4)},
        "seed": SEED,
    }, indent=2))
    logger.info("Saved model -> %s", MODEL_FILE.name)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
