"""
feedback_logger.py
------------------
Logs model predictions to Supabase with smart update logic.

Design: one row per ticker per trading day.
    - First prediction of the day → INSERT
    - Subsequent predictions → UPDATE if new confidence is higher
      or sentiment changed significantly (>0.05 difference)
    - update_count tracks how many times signal changed that day

Schema (run in Supabase SQL Editor):
--------------------------------------
    CREATE TABLE predictions (
        id           uuid DEFAULT gen_random_uuid() PRIMARY KEY,
        created_at   timestamptz DEFAULT now(),
        updated_at   timestamptz DEFAULT now(),
        update_count int DEFAULT 1,
        ticker       text NOT NULL,
        trade_date   date NOT NULL,
        predicted    int,
        actual       int,
        correct      boolean,
        confidence   float,
        sentiment    float,
        retrained    boolean DEFAULT false
    );
    CREATE UNIQUE INDEX idx_predictions_unique ON predictions(ticker, trade_date);
    CREATE INDEX idx_predictions_retrained ON predictions(retrained);
    CREATE INDEX idx_predictions_date      ON predictions(trade_date);

    ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
    CREATE POLICY "app_select" ON predictions FOR SELECT TO anon USING (true);
    CREATE POLICY "app_insert" ON predictions FOR INSERT TO anon WITH CHECK (true);
    CREATE POLICY "app_update" ON predictions FOR UPDATE TO anon USING (true);
    CREATE POLICY "app_delete" ON predictions FOR DELETE TO anon USING (retrained = true);

    -- Daily cleanup via pg_cron (run this once to enable):
    CREATE EXTENSION IF NOT EXISTS pg_cron;
    SELECT cron.schedule(
        'daily-cleanup',
        '0 2 * * *',  -- 2am UTC daily (before most markets open)
        $$DELETE FROM predictions WHERE retrained = true AND created_at < NOW() - INTERVAL '30 days'$$
    );

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

MAX_ROWS              = 10_000
KEEP_DAYS             = 7      # Daily retraining means rows are incorporated the next day;
                               # 7 days is enough for the dashboard accuracy trend chart.
                               # At 28 tickers × 5 trading days = ~140 rows steady state.
AUTO_CLEAN            = True
MIN_CONFIDENCE_DELTA  = 0.02   # update if new confidence is this much higher
MIN_SENTIMENT_DELTA   = 0.05   # update if sentiment shifted by this much


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
        """Returns (ok: bool, message: str)."""
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
        """
        Log or update a prediction for ticker+date.

        Logic:
            - No existing row → INSERT
            - Existing row + new confidence higher by MIN_CONFIDENCE_DELTA → UPDATE
            - Existing row + sentiment shifted by MIN_SENTIMENT_DELTA → UPDATE
            - Otherwise → skip (existing prediction is better or same)
        """
        try:
            client = _get_client()

            # ── Row count guard ────────────────────────────────────────────────
            try:
                count_r = client.table(self.TABLE).select("id", count="exact").limit(1).execute()
                total   = count_r.count if hasattr(count_r, "count") else 0
                if total and total >= MAX_ROWS:
                    logger.warning(f"Row limit ({total}/{MAX_ROWS}) — force cleaning.")
                    self._force_cleanup(client, n=500)
            except Exception:
                pass

            # ── Check for existing row ─────────────────────────────────────────
            existing = (
                client.table(self.TABLE)
                .select("id,predicted,confidence,sentiment,update_count")
                .eq("ticker", ticker.upper())
                .eq("trade_date", trade_date)
                .execute()
            )

            if existing.data:
                row         = existing.data[0]
                old_conf    = float(row.get("confidence") or 0)
                old_sent    = float(row.get("sentiment")  or 0)
                old_count   = int(row.get("update_count") or 1)

                conf_delta  = float(confidence) - old_conf
                sent_delta  = abs(float(sentiment) - old_sent)
                should_update = (
                    conf_delta  >= MIN_CONFIDENCE_DELTA or
                    sent_delta  >= MIN_SENTIMENT_DELTA
                )

                if should_update:
                    client.table(self.TABLE).update({
                        "predicted":    int(predicted),
                        "confidence":   round(float(confidence), 4),
                        "sentiment":    round(float(sentiment),  4),
                        "updated_at":   datetime.utcnow().isoformat(),
                        "update_count": old_count + 1,
                    }).eq("id", row["id"]).execute()
                    logger.info(
                        f"Updated: {ticker} {trade_date} "
                        f"conf {old_conf:.2f}→{confidence:.2f} "
                        f"sent {old_sent:+.3f}→{sentiment:+.3f} "
                        f"(update #{old_count + 1})"
                    )
                else:
                    logger.info(
                        f"Skipped update: {ticker} {trade_date} — "
                        f"existing prediction is equally or more confident "
                        f"(conf delta={conf_delta:+.3f}, sent delta={sent_delta:.3f})"
                    )
                return True

            # ── Insert new row ─────────────────────────────────────────────────
            client.table(self.TABLE).insert({
                "ticker":       ticker.upper(),
                "trade_date":   trade_date,
                "predicted":    int(predicted),
                "confidence":   round(float(confidence), 4),
                "sentiment":    round(float(sentiment),  4),
                "update_count": 1,
                "retrained":    False,
            }).execute()
            logger.info(
                f"Inserted: {ticker} {trade_date} → "
                f"{'UP' if predicted==1 else 'DOWN'} "
                f"(conf={confidence:.1%}, sent={sentiment:+.3f})"
            )

            if AUTO_CLEAN:
                self._auto_cleanup(client)

            return True

        except EnvironmentError:
            return False
        except Exception as e:
            logger.warning(f"log_prediction failed for {ticker}: {e}")
            return False

    def update_actual(self, ticker, trade_date, actual) -> bool:
        """Fill in the actual outcome once next-day price data arrives."""
        if actual == 0:
            return True
        try:
            client = _get_client()
            rows   = (
                client.table(self.TABLE)
                .select("id,predicted")
                .eq("ticker", ticker.upper())
                .eq("trade_date", trade_date)
                .execute()
            )
            if not rows.data:
                return False
            row     = rows.data[0]
            correct = int(row["predicted"]) == int(actual)
            client.table(self.TABLE).update({
                "actual":  int(actual),
                "correct": correct,
            }).eq("id", row["id"]).execute()
            logger.info(f"Actual updated: {ticker} {trade_date} → {'correct' if correct else 'wrong'}")
            return True
        except EnvironmentError:
            return False
        except Exception as e:
            logger.warning(f"update_actual failed: {e}")
            return False

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_performance(self, ticker=None, days_back=90) -> pd.DataFrame:
        """Completed predictions (with known actuals)."""
        try:
            client = _get_client()
            cutoff = (datetime.utcnow() - timedelta(days=days_back)).date().isoformat()
            query  = (
                client.table(self.TABLE)
                .select("trade_date,ticker,predicted,actual,correct,confidence,sentiment,update_count,updated_at")
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
        """Row counts for dashboard health display."""
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
            .agg(
                predictions=("correct", "count"),
                accuracy=("correct", lambda x: round(x.mean()*100, 1)),
                avg_confidence=("confidence", "mean"),
                avg_updates=("update_count", "mean"),
            )
            .reset_index()
            .sort_values("predictions", ascending=False)
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
                logger.info(f"Auto-cleanup: deleted {deleted} rows.")
            return deleted
        except Exception as e:
            logger.debug(f"Auto-cleanup skipped: {e}")
            return 0

    @staticmethod
    def _force_cleanup(client, n=500) -> int:
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
                logger.warning("Force cleanup: no retrained rows available.")
                return 0
            ids    = [r["id"] for r in oldest.data]
            result = client.table(FeedbackLogger.TABLE).delete().in_("id", ids).execute()
            deleted = len(result.data) if result.data else len(ids)
            logger.info(f"Force cleanup: deleted {deleted} rows.")
            return deleted
        except Exception as e:
            logger.warning(f"Force cleanup failed: {e}")
            return 0
