"""
feedback_logger.py
------------------
Logs model predictions and actual outcomes to Supabase (free Postgres).

Robustness features:
    - Auto cleanup on every log_prediction call
    - Row count guard: force-deletes oldest retrained rows if table exceeds MAX_ROWS
    - Connection test method for dashboard diagnostics
    - All methods fail gracefully — app works without Supabase

Cleanup schedule:
    - Automatic: runs after every insert (deletes retrained rows older than KEEP_DAYS)
    - Manual: Retrain button triggers additional cleanup
    - Guard: if total rows exceed MAX_ROWS, oldest retrained rows are force-deleted

SQL to run in Supabase SQL Editor (one-time):
    CREATE TABLE predictions (
        id          uuid DEFAULT gen_random_uuid() PRIMARY KEY,
        created_at  timestamptz DEFAULT now(),
        ticker      text NOT NULL,
        trade_date  date NOT NULL,
        predicted   int,
        actual      int,
        correct     boolean,
        confidence  float,
        sentiment   float,
        retrained   boolean DEFAULT false
    );
    CREATE INDEX idx_predictions_ticker    ON predictions(ticker);
    CREATE INDEX idx_predictions_retrained ON predictions(retrained);
    CREATE INDEX idx_predictions_date      ON predictions(trade_date);
    ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
    CREATE POLICY "app_select" ON predictions FOR SELECT TO anon USING (true);
    CREATE POLICY "app_insert" ON predictions FOR INSERT TO anon WITH CHECK (true);
    CREATE POLICY "app_update" ON predictions FOR UPDATE TO anon USING (true);
    CREATE POLICY "app_delete" ON predictions FOR DELETE TO anon USING (retrained = true);

Streamlit secrets:
    [supabase]
    url = "https://xxxx.supabase.co"
    key = "sb_publishable_xxxx"
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

MAX_ROWS   = 10_000
KEEP_DAYS  = 30
AUTO_CLEAN = True


def _get_client():
    url = key = ""
    try:
        import streamlit as st
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
    except Exception:
        pass
    if not url or not key:
        import os
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_KEY", "")
    if not url or not key:
        raise EnvironmentError(
            "Supabase credentials not found.\n"
            "Add to Streamlit Cloud App Settings → Secrets:\n"
            "  [supabase]\n"
            "  url = 'https://xxxx.supabase.co'\n"
            "  key = 'sb_publishable_xxxx'"
        )
    from supabase import create_client
    return create_client(url, key)


class FeedbackLogger:
    TABLE = "predictions"

    # ── Connection test ────────────────────────────────────────────────────────

    def test_connection(self) -> tuple:
        """Test Supabase connection. Returns (ok: bool, message: str)."""
        try:
            client = _get_client()
            result = client.table(self.TABLE).select("id", count="exact").limit(1).execute()
            count  = result.count if hasattr(result, "count") else "?"
            return True, f"Connected — {count} rows in predictions table"
        except EnvironmentError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Connection failed: {e}"

    # ── Write ──────────────────────────────────────────────────────────────────

    def log_prediction(self, ticker, trade_date, predicted, confidence, sentiment) -> bool:
        """Log a new prediction. Also triggers auto cleanup."""
        try:
            client = _get_client()

            # Duplicate guard
            existing = (
                client.table(self.TABLE)
                .select("id")
                .eq("ticker", ticker.upper())
                .eq("trade_date", trade_date)
                .execute()
            )
            if existing.data:
                logger.info(f"Already logged: {ticker} {trade_date}")
                return True

            # Row count guard
            try:
                count_result = client.table(self.TABLE).select("id", count="exact").limit(1).execute()
                total = count_result.count if hasattr(count_result, "count") else 0
                if total and total >= MAX_ROWS:
                    logger.warning(f"Row limit reached ({total}) — force cleaning 500 rows.")
                    self._force_cleanup(client, n=500)
            except Exception:
                pass

            # Insert
            client.table(self.TABLE).insert({
                "ticker":     ticker.upper(),
                "trade_date": trade_date,
                "predicted":  int(predicted),
                "confidence": round(float(confidence), 4),
                "sentiment":  round(float(sentiment), 4),
                "retrained":  False,
            }).execute()

            logger.info(f"Logged: {ticker} {trade_date} → {'UP' if predicted==1 else 'DOWN'} (conf={confidence:.1%})")

            if AUTO_CLEAN:
                self._auto_cleanup(client)

            return True

        except EnvironmentError:
            return False
        except Exception as e:
            logger.warning(f"Could not log prediction for {ticker}: {e}")
            return False

    def update_actual(self, ticker, trade_date, actual) -> bool:
        """Update a prediction with the actual next-day outcome."""
        if actual == 0:
            return True
        try:
            client = _get_client()
            rows = (
                client.table(self.TABLE)
                .select("id,predicted")
                .eq("ticker", ticker.upper())
                .eq("trade_date", trade_date)
                .execute()
            )
            if not rows.data:
                logger.debug(f"No row for {ticker} {trade_date}")
                return False
            row     = rows.data[0]
            correct = int(row["predicted"]) == int(actual)
            client.table(self.TABLE).update({
                "actual":  int(actual),
                "correct": correct,
            }).eq("id", row["id"]).execute()
            logger.info(f"Updated actual {ticker} {trade_date}: {'correct' if correct else 'wrong'}")
            return True
        except EnvironmentError:
            return False
        except Exception as e:
            logger.warning(f"update_actual failed for {ticker}: {e}")
            return False

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_performance(self, ticker=None, days_back=90) -> pd.DataFrame:
        """Fetch completed predictions (those with known actuals)."""
        try:
            client = _get_client()
            cutoff = (datetime.utcnow() - timedelta(days=days_back)).date().isoformat()
            query  = (
                client.table(self.TABLE)
                .select("trade_date,ticker,predicted,actual,correct,confidence,sentiment")
                .gte("trade_date", cutoff)
                .not_.is_("actual", "null")
                .order("trade_date", desc=True)
            )
            if ticker:
                query = query.eq("ticker", ticker.upper())
            result = query.execute()
            if not result.data:
                return pd.DataFrame()
            df = pd.DataFrame(result.data)
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df["correct"]    = df["correct"].astype(bool)
            return df
        except EnvironmentError:
            return pd.DataFrame()
        except Exception as e:
            logger.warning(f"get_performance failed: {e}")
            return pd.DataFrame()

    def get_unretrained(self) -> pd.DataFrame:
        """Rows with outcomes not yet used in retraining."""
        try:
            client = _get_client()
            result = (
                client.table(self.TABLE)
                .select("*")
                .eq("retrained", False)
                .not_.is_("actual", "null")
                .execute()
            )
            return pd.DataFrame(result.data) if result.data else pd.DataFrame()
        except Exception as e:
            logger.warning(f"get_unretrained failed: {e}")
            return pd.DataFrame()

    def table_stats(self) -> dict:
        """Return row counts for dashboard health display."""
        try:
            client    = _get_client()
            total     = client.table(self.TABLE).select("id", count="exact").limit(1).execute()
            retrained = client.table(self.TABLE).select("id", count="exact").eq("retrained", True).limit(1).execute()
            pending   = client.table(self.TABLE).select("id", count="exact").eq("retrained", False).is_("actual", "null").limit(1).execute()
            return {
                "total":     getattr(total,     "count", 0) or 0,
                "retrained": getattr(retrained, "count", 0) or 0,
                "pending":   getattr(pending,   "count", 0) or 0,
                "max_rows":  MAX_ROWS,
                "keep_days": KEEP_DAYS,
            }
        except Exception as e:
            logger.warning(f"table_stats failed: {e}")
            return {}

    # ── Cleanup ────────────────────────────────────────────────────────────────

    def mark_retrained(self, ids) -> bool:
        try:
            client = _get_client()
            client.table(self.TABLE).update({"retrained": True}).in_("id", ids).execute()
            logger.info(f"Marked {len(ids)} rows as retrained.")
            return True
        except Exception as e:
            logger.warning(f"mark_retrained failed: {e}")
            return False

    def cleanup_old(self, keep_days=KEEP_DAYS) -> int:
        """Manual cleanup — called after retraining."""
        try:
            client  = _get_client()
            deleted = self._auto_cleanup(client, keep_days=keep_days)
            logger.info(f"Manual cleanup: deleted {deleted} rows.")
            return deleted
        except Exception as e:
            logger.warning(f"cleanup_old failed: {e}")
            return 0

    def summary_stats(self) -> dict:
        df = self.get_performance(days_back=90)
        if df.empty:
            return {}
        by_ticker = (
            df.groupby("ticker")
            .agg(predictions=("correct","count"), accuracy=("correct", lambda x: round(x.mean()*100,1)), avg_confidence=("confidence","mean"))
            .reset_index().sort_values("predictions", ascending=False)
        )
        return {
            "total_predictions": len(df),
            "accuracy":          round(df["correct"].mean()*100, 1),
            "avg_confidence":    round(df["confidence"].mean(), 3),
            "by_ticker":         by_ticker.to_dict("records"),
            "daily_accuracy":    df.groupby("trade_date")["correct"].mean().reset_index(),
        }

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _auto_cleanup(client, keep_days=KEEP_DAYS) -> int:
        """Delete retrained rows older than keep_days. Runs after every insert."""
        try:
            cutoff = (datetime.utcnow() - timedelta(days=keep_days)).isoformat()
            result = (
                client.table(FeedbackLogger.TABLE)
                .delete()
                .eq("retrained", True)
                .lt("created_at", cutoff)
                .execute()
            )
            deleted = len(result.data) if result.data else 0
            if deleted:
                logger.info(f"Auto-cleanup: deleted {deleted} rows older than {keep_days} days.")
            return deleted
        except Exception as e:
            logger.debug(f"Auto-cleanup skipped: {e}")
            return 0

    @staticmethod
    def _force_cleanup(client, n=500) -> int:
        """Emergency cleanup when row count exceeds MAX_ROWS."""
        try:
            oldest = (
                client.table(FeedbackLogger.TABLE)
                .select("id")
                .eq("retrained", True)
                .order("created_at", desc=False)
                .limit(n)
                .execute()
            )
            if not oldest.data:
                logger.warning("Force cleanup: no retrained rows to delete.")
                return 0
            ids    = [r["id"] for r in oldest.data]
            result = client.table(FeedbackLogger.TABLE).delete().in_("id", ids).execute()
            deleted = len(result.data) if result.data else len(ids)
            logger.info(f"Force cleanup: deleted {deleted} rows.")
            return deleted
        except Exception as e:
            logger.warning(f"Force cleanup failed: {e}")
            return 0
