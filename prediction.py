"""
prediction.py
-------------
Module 4: Train and evaluate models to predict next-day stock direction.

What we're predicting:
    next_day_direction: +1 (up >0.1%), -1 (down >0.1%), 0 (flat)

Model pipeline:
    1. Logistic Regression  — fast baseline, interpretable
    2. Random Forest        — captures non-linear feature interactions
    3. Walk-forward validation — honest evaluation that respects time order

Why walk-forward matters:
    Standard train/test split randomly shuffles rows. For time-series data
    this causes lookahead bias — the model trains on "future" data relative
    to some test rows. Walk-forward always trains on the past and tests on
    the future, which is the only honest evaluation for financial prediction.

    Example with 242 rows and 3 folds:
        Fold 1: train rows 1-161,  test rows 162-202
        Fold 2: train rows 1-202,  test rows 203-222  (training grows)
        Fold 3: train rows 1-222,  test rows 223-242

Usage:
    from src.prediction import PredictionModel

    model = PredictionModel()
    results = model.train_evaluate(X, y)
    model.save("models/rf_model.pkl")
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


class PredictionModel:
    """
    Trains and evaluates direction prediction models.

    Args:
        n_folds     : Number of walk-forward folds (default: 5)
        test_size   : Fraction of data per fold used as test (default: 0.15)
        random_state: Reproducibility seed (default: 42)

    Example:
        model   = PredictionModel()
        results = model.train_evaluate(X, y)
    """

    def __init__(
        self,
        n_folds: int = 5,
        test_size: float = 0.15,
        random_state: int = 42,
    ):
        self.n_folds      = n_folds
        self.test_size    = test_size
        self.random_state = random_state
        self.best_model   = None
        self.best_name    = None
        self.feature_cols = None
        self.scaler       = None

    # ── Main entry point ───────────────────────────────────────────────────────

    def train_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict:
        """
        Train all models with walk-forward validation and return results.

        Args:
            X : Feature matrix (output of FeatureEngineer.split_features_target)
            y : Target series (next_day_direction)

        Returns:
            dict with keys: 'summary', 'fold_details', 'best_model_name'
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        self.feature_cols = list(X.columns)

        # Drop flat class (0) — too few samples and not actionable for trading.
        # In practice, "flat" days are noise around the ±0.1% threshold.
        mask = y != 0
        X    = X[mask]
        y    = y[mask]
        logger.info(
            f"After dropping flat class: {len(X)} rows "
            f"({(y == 1).sum()} up, {(y == -1).sum()} down)"
        )

        # ── Define models ──────────────────────────────────────────────────────
        # StandardScaler is fitted inside each fold to prevent leakage.
        models = {
            "Logistic Regression": LogisticRegression(
                max_iter     = 1000,
                random_state = self.random_state,
                class_weight = "balanced",  # handles any remaining imbalance
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators = 200,
                max_depth    = 4,       # shallow trees → less overfitting
                min_samples_leaf = 10,  # require meaningful leaf size
                random_state = self.random_state,
                class_weight = "balanced",
                n_jobs       = -1,
            ),
        }

        # ── Walk-forward validation ────────────────────────────────────────────
        folds   = self._make_folds(len(X))
        results = {name: [] for name in models}

        logger.info(f"Running walk-forward validation ({self.n_folds} folds)...")

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            X_train = X.iloc[train_idx]
            X_test  = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test  = y.iloc[test_idx]

            # Fit scaler on training data only — never on test data
            scaler  = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s  = scaler.transform(X_test)

            for name, model in models.items():
                model.fit(X_train_s, y_train)
                preds   = model.predict(X_test_s)
                metrics = self._compute_metrics(y_test, preds)
                metrics["fold"]       = fold_idx + 1
                metrics["train_rows"] = len(train_idx)
                metrics["test_rows"]  = len(test_idx)
                results[name].append(metrics)

                logger.info(
                    f"  Fold {fold_idx+1} | {name:22s} | "
                    f"acc={metrics['accuracy']:.3f} | "
                    f"f1={metrics['f1']:.3f} | "
                    f"precision={metrics['precision']:.3f}"
                )

        # ── Summarise results ──────────────────────────────────────────────────
        summary = {}
        for name, fold_results in results.items():
            fold_df = pd.DataFrame(fold_results)
            summary[name] = {
                "accuracy_mean":  fold_df["accuracy"].mean(),
                "accuracy_std":   fold_df["accuracy"].std(),
                "f1_mean":        fold_df["f1"].mean(),
                "f1_std":         fold_df["f1"].std(),
                "precision_mean": fold_df["precision"].mean(),
                "precision_std":  fold_df["precision"].std(),
            }

        self._log_summary(summary)

        # ── Pick best model and retrain on all data ────────────────────────────
        best_name = max(summary, key=lambda k: summary[k]["f1_mean"])
        logger.info(f"\nBest model: {best_name} — retraining on full dataset...")

        self.scaler = StandardScaler()
        X_all_s     = self.scaler.fit_transform(X)

        self.best_model = models[best_name]
        self.best_model.fit(X_all_s, y)
        self.best_name = best_name

        # Feature importance (Random Forest only)
        if best_name == "Random Forest":
            self._log_feature_importance(self.best_model, self.feature_cols)

        return {
            "summary":         summary,
            "fold_details":    results,
            "best_model_name": best_name,
        }

    # ── Walk-forward fold construction ─────────────────────────────────────────

    def _make_folds(self, n: int) -> list[tuple]:
        """
        Build walk-forward (expanding window) folds.

        Each fold:
            - Training set: all rows from start to the fold boundary
            - Test set:     the next test_size fraction of rows
            - Training set GROWS with each fold (expanding window)

        This mirrors how a live system works: you always train on
        all available history and test on the most recent data.
        """
        test_n   = max(10, int(n * self.test_size))
        # First training set ends at: n - n_folds * test_n
        min_train = n - self.n_folds * test_n

        if min_train < 20:
            # Not enough data for requested folds — reduce
            self.n_folds = max(2, (n - 20) // test_n)
            min_train    = n - self.n_folds * test_n
            logger.warning(
                f"Reduced to {self.n_folds} folds due to limited data."
            )

        folds = []
        for i in range(self.n_folds):
            train_end = min_train + i * test_n
            test_start = train_end
            test_end   = min(test_start + test_n, n)
            folds.append((
                list(range(0, train_end)),
                list(range(test_start, test_end)),
            ))

        return folds

    # ── Metrics ────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_metrics(y_true, y_pred) -> dict:
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score,
        )
        return {
            "accuracy":  accuracy_score(y_true, y_pred),
            "f1":        f1_score(y_true, y_pred, average="macro", zero_division=0),
            "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        }

    # ── Logging helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _log_summary(summary: dict):
        logger.info("\n── Model comparison ─────────────────────────────")
        for name, s in summary.items():
            logger.info(
                f"  {name:25s} | "
                f"accuracy={s['accuracy_mean']:.3f}±{s['accuracy_std']:.3f} | "
                f"f1={s['f1_mean']:.3f}±{s['f1_std']:.3f} | "
                f"precision={s['precision_mean']:.3f}±{s['precision_std']:.3f}"
            )

    @staticmethod
    def _log_feature_importance(model, feature_cols: list):
        importances = model.feature_importances_
        pairs = sorted(
            zip(feature_cols, importances),
            key=lambda x: x[1],
            reverse=True,
        )
        logger.info("\n── Random Forest feature importance ─────────────")
        for feat, imp in pairs[:10]:
            bar = "█" * int(imp * 200)
            logger.info(f"  {feat:25s}: {imp:.4f}  {bar}")

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the best trained model.

        Args:
            X : Feature DataFrame with same columns as training data

        Returns:
            Array of predictions: +1 (up), -1 (down)
        """
        if self.best_model is None:
            raise RuntimeError("Model not trained. Call train_evaluate() first.")

        X_s = self.scaler.transform(X[self.feature_cols])
        return self.best_model.predict(X_s)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return class probabilities for confidence-weighted signals.
        Higher probability = stronger signal for backtesting.
        """
        if self.best_model is None:
            raise RuntimeError("Model not trained. Call train_evaluate() first.")

        X_s = self.scaler.transform(X[self.feature_cols])
        return self.best_model.predict_proba(X_s)

    # ── Save / load ────────────────────────────────────────────────────────────

    def save(self, path: str | None = None) -> Path:
        """Save the trained model + scaler to disk."""
        if self.best_model is None:
            raise RuntimeError("No trained model to save.")

        path = Path(path) if path else MODELS_DIR / f"{self.best_name.lower().replace(' ', '_')}.pkl"
        payload = {
            "model":        self.best_model,
            "scaler":       self.scaler,
            "feature_cols": self.feature_cols,
            "best_name":    self.best_name,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info(f"Model saved → {path}")
        return path

    @classmethod
    def load(cls, path: str) -> "PredictionModel":
        """Load a previously saved model."""
        with open(path, "rb") as f:
            payload = pickle.load(f)

        instance              = cls()
        instance.best_model   = payload["model"]
        instance.scaler       = payload["scaler"]
        instance.feature_cols = payload["feature_cols"]
        instance.best_name    = payload["best_name"]
        logger.info(f"Loaded {instance.best_name} from {path}")
        return instance


# ══════════════════════════════════════════════════════════════════════════════
# QUICK TEST  (run: python -m src.prediction)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import logging
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    from src.rss_ingester import fetch_rss_and_stock
    from src.sentiment_model import SentimentPipeline
    from src.feature_engineering import FeatureEngineer

    print("── Loading data via RSS ──")
    news_df, stock_df = fetch_rss_and_stock("AAPL")

    print("── Scoring sentiment ──")
    pipe    = SentimentPipeline()
    scored  = pipe.score(news_df)
    daily   = pipe.aggregate_daily(scored)

    print("── Engineering features ──")
    fe       = FeatureEngineer()
    features = fe.build(daily, stock_df)
    X, y     = fe.split_features_target(features, target="next_day_direction")

    print(f"\n── Training models (X={X.shape}, y={y.shape}) ──")
    model   = PredictionModel()
    results = model.train_evaluate(X, y)

    print("\n── Summary ──")
    for name, s in results["summary"].items():
        print(
            f"  {name:25s} | "
            f"accuracy={s['accuracy_mean']:.3f} | "
            f"f1={s['f1_mean']:.3f}"
        )

    path = model.save()
    print(f"\nModel saved → {path}")
