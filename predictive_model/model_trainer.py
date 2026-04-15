"""
predictive_model/model_trainer.py - Multi-Model Training
===========================================================

Trains RF, GB, XGBoost, LR, and SVM classifiers with temporal
train/test split and SMOTE resampling.
"""

import os
import sys
import time
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.svm import SVC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config
from predictive_model.imbalance_handler import ImbalanceHandler


class SECOMModelTrainer:
    """Train and cross-validate multiple classifiers.

    Args:
        cfg: Optional configuration override.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or config

    # =================================================================
    # Data preparation – temporal split + SMOTE
    # =================================================================
    def prepare_data(
        self, X_pca: np.ndarray, y: np.ndarray | pd.Series
    ) -> Dict:
        """Create temporal train/test split and apply SMOTE.

        Args:
            X_pca: PCA score matrix (n × p).
            y: Binary labels.

        Returns:
            Dictionary with train/test arrays and SMOTE-resampled
            training data.
        """
        print("\n══════════════════════════════════════════════════")
        print("        ML DATA PREPARATION")
        print("══════════════════════════════════════════════════")

        y = np.asarray(y)
        n = len(y)
        split = int(n * (1 - self.cfg.test_size))

        X_train, X_test = X_pca[:split], X_pca[split:]
        y_train, y_test = y[:split], y[split:]

        print(f"  Total  : {n}")
        print(f"  Train  : {len(y_train)} (class 0={int((y_train==0).sum())}, "
              f"class 1={int((y_train==1).sum())})")
        print(f"  Test   : {len(y_test)} (class 0={int((y_test==0).sum())}, "
              f"class 1={int((y_test==1).sum())})")

        handler = ImbalanceHandler(self.cfg)
        X_train_smote, y_train_smote = handler.fit_resample(X_train, y_train)

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "X_train_smote": X_train_smote,
            "y_train_smote": y_train_smote,
        }

    # =================================================================
    # Training
    # =================================================================
    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Dict:
        """Train all configured classifiers.

        Args:
            X_train: Training features (post-SMOTE).
            y_train: Training labels (post-SMOTE).

        Returns:
            Dictionary of ``{model_name: fitted_model}``.
        """
        print("\n══════════════════════════════════════════════════")
        print("        MODEL TRAINING")
        print("══════════════════════════════════════════════════")

        models: Dict = {}
        model_defs = {
            "RandomForest": RandomForestClassifier(
                n_estimators=self.cfg.rf_n_estimators,
                max_depth=self.cfg.rf_max_depth,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=self.cfg.random_seed,
                n_jobs=-1,
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=self.cfg.gb_n_estimators,
                learning_rate=self.cfg.gb_learning_rate,
                max_depth=5,
                subsample=0.8,
                random_state=self.cfg.random_seed,
            ),
            "LogisticRegression": LogisticRegression(
                C=0.1,
                class_weight="balanced",
                max_iter=1000,
                random_state=self.cfg.random_seed,
            ),
            "SVM": SVC(
                kernel="rbf",
                C=1.0,
                class_weight="balanced",
                probability=True,
                random_state=self.cfg.random_seed,
            ),
        }

        # Try XGBoost
        try:
            from xgboost import XGBClassifier

            n_fail = int((y_train == 1).sum())
            n_pass = int((y_train == 0).sum())
            spw = n_pass / max(n_fail, 1)
            model_defs["XGBoost"] = XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                scale_pos_weight=spw,
                random_state=self.cfg.random_seed,
                eval_metric="logloss",
                use_label_encoder=False,
            )
        except ImportError:
            print("  ⚠ XGBoost not installed – skipping.")

        for name, model in model_defs.items():
            print(f"  Training {name} …", end=" ")
            t0 = time.time()
            model.fit(X_train, y_train)
            elapsed = time.time() - t0
            print(f"Done! Time: {elapsed:.1f}s")
            models[name] = model

            # Save
            os.makedirs(self.cfg.models_dir, exist_ok=True)
            path = os.path.join(self.cfg.models_dir, f"{name}.pkl")
            joblib.dump(model, path)

        return models

    # =================================================================
    # Cross-validation
    # =================================================================
    def cross_validate_all(
        self,
        models: Dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> pd.DataFrame:
        """Run stratified k-fold CV for every model.

        Args:
            models: Dictionary of fitted (or unfitted) models.
            X_train: Training features.
            y_train: Training labels.

        Returns:
            DataFrame of CV scores.
        """
        print("\n── Cross-Validation ──")
        cv = StratifiedKFold(
            n_splits=self.cfg.cv_folds,
            shuffle=True,
            random_state=self.cfg.random_seed,
        )
        scoring = ["roc_auc", "f1", "recall", "precision"]

        records = []
        for name, model in models.items():
            try:
                res = cross_validate(
                    model, X_train, y_train, cv=cv,
                    scoring=scoring, n_jobs=-1,
                )
                rec = {"model": name}
                for s in scoring:
                    key = f"test_{s}"
                    rec[f"cv_{s}_mean"] = round(float(res[key].mean()), 4)
                    rec[f"cv_{s}_std"] = round(float(res[key].std()), 4)
                records.append(rec)
                print(f"  {name:<22} AUC={rec['cv_roc_auc_mean']:.4f} ± {rec['cv_roc_auc_std']:.4f}")
            except Exception as exc:
                print(f"  {name}: CV failed – {exc}")

        df = pd.DataFrame(records)
        os.makedirs(self.cfg.tables_dir, exist_ok=True)
        path = os.path.join(self.cfg.tables_dir, "cv_results.csv")
        df.to_csv(path, index=False)
        print(f"  ✓ Saved → {path}")
        return df

    # =================================================================
    # Hyperparameter tuning
    # =================================================================
    def tune_best_model(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ):
        """Tune the best model with randomised search.

        Args:
            model: A scikit-learn–compatible estimator.
            X_train: Training features.
            y_train: Training labels.

        Returns:
            Best estimator from the search.
        """
        print("\n── Hyperparameter Tuning ──")
        model_type = type(model).__name__

        if "RandomForest" in model_type:
            param_dist = {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [5, 8, 10, 15, None],
                "min_samples_leaf": [3, 5, 10],
                "max_features": ["sqrt", "log2"],
            }
        elif "GradientBoosting" in model_type:
            param_dist = {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7],
                "subsample": [0.7, 0.8, 0.9],
            }
        else:
            param_dist = {"C": [0.01, 0.1, 1, 10]}

        cv = StratifiedKFold(n_splits=self.cfg.cv_folds, shuffle=True,
                             random_state=self.cfg.random_seed)
        search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=min(50, len(param_dist) * 10),
            scoring=self.cfg.scoring_metric,
            cv=cv,
            random_state=self.cfg.random_seed,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)
        print(f"  Best params : {search.best_params_}")
        print(f"  Best score  : {search.best_score_:.4f}")
        return search.best_estimator_
