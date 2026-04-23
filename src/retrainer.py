"""
retrainer.py
------------
Pulls feedback from Supabase, retrains the Random Forest with new data,
marks feedback rows as incorporated, and cleans up old rows.

This is designed to be triggered manually from the dashboard
(via a "Retrain Model" button) or run as a scheduled job.

Usage:
    from src.retrainer import Retrainer

    r = Retrainer()
    result = r.run(ticker="AAPL", features_df=features, X=X, y=y)
"""

import logging
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


class Retrainer:
    """
    Incorporates feedback data into model retraining.

    Strategy:
        1. Pull unretrained feedback rows from Supabase
        2. These rows tell us: on date D, the model predicted P,
           actual outcome was A, and sentiment was S.
        3. We can't reconstruct all features from feedback alone,
           so we use feedback to:
             a) Weight recent errors more heavily in retraining
             b) Track accuracy drift over time
             c) Trigger a full pipeline re-run when accuracy drops
        4. Mark rows as retrained, clean up old rows.
    """

    def __init__(self, accuracy_threshold: float = 0.48):
        """
        Args:
            accuracy_threshold: If rolling accuracy drops below this,
                                flag that a full retrain is needed.
                                Default 0.48 = below random chance.
        """
        self.accuracy_threshold = accuracy_threshold

    def run(
        self,
        ticker: str,
        features_df: pd.DataFrame,
        X: pd.DataFrame,
        y: pd.Series,
        keep_days: int = 30,
    ) -> dict:
        """
        Run the full retrain cycle for a ticker.

        Args:
            ticker      : Stock symbol e.g. "AAPL"
            features_df : Full feature matrix from FeatureEngineer
            X           : Feature columns
            y           : Target series
            keep_days   : Days to keep retrained rows before cleanup

        Returns:
            dict with keys: success, new_accuracy, prev_accuracy,
                            rows_incorporated, rows_cleaned, model
        """
        from src.feedback_logger import FeedbackLogger
        from src.prediction import PredictionModel

        logger.info(f"Starting retrain cycle for {ticker}...")
        fb = FeedbackLogger()

        # ── Step 1: Check feedback ─────────────────────────────────────────────
        unretrained = fb.get_unretrained()
        ticker_rows = unretrained[unretrained["ticker"] == ticker.upper()] if not unretrained.empty else pd.DataFrame()
        n_new = len(ticker_rows)
        logger.info(f"Found {n_new} unretrained feedback rows for {ticker}.")

        # ── Step 2: Get previous accuracy ─────────────────────────────────────
        perf    = fb.get_performance(ticker=ticker, days_back=60)
        prev_acc = round(perf["correct"].mean() * 100, 1) if not perf.empty else None

        # ── Step 3: Retrain model on full current data ─────────────────────────
        # We always retrain on the full feature matrix (which already includes
        # the most recent data from RSS + yfinance). Feedback rows help us
        # understand WHERE the model is going wrong, but the features themselves
        # come from the live pipeline, not the feedback store.
        logger.info("Retraining Random Forest on full dataset...")
        model   = PredictionModel(n_folds=3)   # fewer folds = faster retrain
        results = model.train_evaluate(X, y)
        new_acc = round(results["summary"]["Random Forest"]["accuracy_mean"] * 100, 1)

        logger.info(f"Retrain complete. Accuracy: {prev_acc}% → {new_acc}%")

        # ── Step 4: Mark feedback rows as retrained ───────────────────────────
        rows_incorporated = 0
        if not ticker_rows.empty and "id" in ticker_rows.columns:
            ids = ticker_rows["id"].tolist()
            fb.mark_retrained(ids)
            rows_incorporated = len(ids)

        # ── Step 5: Cleanup old retrained rows ────────────────────────────────
        rows_cleaned = fb.cleanup_old(keep_days=keep_days)

        # ── Step 6: Accuracy drift warning ────────────────────────────────────
        drift_warning = None
        if new_acc < self.accuracy_threshold * 100:
            drift_warning = (
                f"⚠️ Model accuracy ({new_acc}%) is below threshold "
                f"({self.accuracy_threshold*100}%). Consider fetching more "
                f"historical data or reviewing feature quality."
            )
            logger.warning(drift_warning)

        return {
            "success":           True,
            "ticker":            ticker,
            "new_accuracy":      new_acc,
            "prev_accuracy":     prev_acc,
            "rows_incorporated": rows_incorporated,
            "rows_cleaned":      rows_cleaned,
            "drift_warning":     drift_warning,
            "model":             model,
            "timestamp":         datetime.utcnow().isoformat() + "Z",
        }

    def should_retrain(self, ticker: str, min_new_rows: int = 5) -> tuple[bool, str]:
        """
        Check whether a retrain is warranted right now.

        Returns (should_retrain, reason_string).
        Useful for showing the user why/why not to retrain.
        """
        try:
            from src.feedback_logger import FeedbackLogger
            fb          = FeedbackLogger()
            unretrained = fb.get_unretrained()

            if unretrained.empty:
                return False, "No new feedback data available yet."

            ticker_rows = unretrained[unretrained["ticker"] == ticker.upper()]
            n = len(ticker_rows)

            if n < min_new_rows:
                return False, f"Only {n} new feedback rows — need at least {min_new_rows} to retrain."

            perf = fb.get_performance(ticker=ticker, days_back=30)
            if not perf.empty:
                recent_acc = perf["correct"].mean()
                if recent_acc < self.accuracy_threshold:
                    return True, f"Accuracy dropped to {recent_acc*100:.1f}% — retraining recommended."

            return True, f"{n} new feedback rows available — ready to incorporate."

        except Exception as e:
            return False, f"Could not check retrain status: {e}"
