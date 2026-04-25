"""
scripts/daily_retrain.py
------------------------
Runs the full retrain + cleanup cycle for all tracked tickers.
Called daily by GitHub Actions before market open.

Can be run manually:
    SUPABASE_URL=... SUPABASE_KEY=... python scripts/daily_retrain.py

Or for specific tickers:
    TICKERS_OVERRIDE="AAPL,MSFT" python scripts/daily_retrain.py
"""

import os
import sys
import logging
from datetime import datetime, timedelta, date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("daily_retrain")

DEFAULT_TICKERS = [
    # USA
    "AAPL", "MSFT", "GOOGL", "TSLA", "NVDA",
    # India
    "RELIANCE.NS", "TCS.NS", "INFY.NS",
    # Europe
    "ASML.AS", "SAP.DE", "BP.L",
    # HK/China
    "0700.HK", "BABA",
    # Japan
    "7203.T",
]


def get_tickers() -> list[str]:
    override = os.getenv("TICKERS_OVERRIDE", "").strip()
    if override:
        tickers = [t.strip().upper() for t in override.split(",") if t.strip()]
        logger.info(f"Using override tickers: {tickers}")
        return tickers
    logger.info(f"Using default tickers: {DEFAULT_TICKERS}")
    return DEFAULT_TICKERS


def fill_missing_actuals(fb) -> int:
    """
    Key fix: scan ALL pending rows in Supabase (actual IS NULL) and
    fill in actuals using fresh yfinance data, regardless of whether
    the user has visited the dashboard.

    This runs BEFORE any retraining, ensuring every row that can be
    resolved is resolved before the retrain query filters on actual IS NOT NULL.

    Returns: number of rows updated.
    """
    import yfinance as yf
    import pandas as pd

    logger.info("Filling missing actuals from yfinance...")

    try:
        # Get all rows with null actuals
        from supabase import create_client
        client = _get_supabase()
        pending = (
            client.table("predictions")
            .select("id,ticker,trade_date,predicted")
            .is_("actual", "null")
            .execute()
        )

        if not pending.data:
            logger.info("No pending actuals to fill.")
            return 0

        # Group by ticker to minimise yfinance calls
        from collections import defaultdict
        by_ticker = defaultdict(list)
        for row in pending.data:
            by_ticker[row["ticker"]].append(row)

        updated = 0
        for ticker, rows in by_ticker.items():
            try:
                # Fetch last 10 days of price data for this ticker
                df = yf.download(
                    ticker,
                    start=(datetime.utcnow() - timedelta(days=15)).strftime("%Y-%m-%d"),
                    end=datetime.utcnow().strftime("%Y-%m-%d"),
                    progress=False,
                    auto_adjust=True,
                )
                if df.empty:
                    continue

                # Flatten MultiIndex if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df["date"]         = pd.to_datetime(df.index).date
                df["daily_return"] = df["Close"].pct_change() * 100
                df["direction"]    = df["daily_return"].apply(
                    lambda r: 1 if r > 0.1 else (-1 if r < -0.1 else 0)
                )

                # Build a date → direction lookup
                price_map = {str(row["date"]): int(row["direction"])
                             for _, row in df.iterrows()
                             if not pd.isna(row["direction"])}

                for pred_row in rows:
                    td = pred_row["trade_date"]

                    # We need the return on trade_date+1
                    # (prediction is made on T, outcome is T+1's return)
                    trade_dt    = datetime.strptime(td, "%Y-%m-%d").date()
                    next_day    = trade_dt + timedelta(days=1)

                    # Walk forward to find the next trading day with data
                    actual = None
                    for offset in range(5):   # look up to 5 days ahead (weekends/holidays)
                        check = str(next_day + timedelta(days=offset))
                        if check in price_map:
                            actual = price_map[check]
                            break

                    if actual is not None and actual != 0:
                        correct = int(pred_row["predicted"]) == actual
                        client.table("predictions").update({
                            "actual":  actual,
                            "correct": correct,
                        }).eq("id", pred_row["id"]).execute()
                        updated += 1
                        logger.info(
                            f"  Filled actual: {ticker} {td} → "
                            f"{'correct' if correct else 'wrong'} "
                            f"(predicted={pred_row['predicted']}, actual={actual})"
                        )

            except Exception as e:
                logger.warning(f"  Could not fill actuals for {ticker}: {e}")
                continue

        logger.info(f"Filled {updated} missing actuals.")
        return updated

    except Exception as e:
        logger.error(f"fill_missing_actuals failed: {e}")
        return 0


def _get_supabase():
    """Get Supabase client directly (without Streamlit secrets)."""
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    if not url or not key:
        raise EnvironmentError("SUPABASE_URL and SUPABASE_KEY must be set.")
    from supabase import create_client
    return create_client(url, key)


def retrain_ticker(ticker: str, fb) -> dict:
    """Run the full pipeline and retrain for a single ticker."""
    from src.rss_ingester import fetch_rss_and_stock
    from src.sentiment_model import SentimentPipeline
    from src.feature_engineering import FeatureEngineer
    from src.prediction import PredictionModel

    logger.info(f"── {ticker} ──────────────────────────────────")

    try:
        # Fetch data
        news_df, stock_df = fetch_rss_and_stock(ticker, stock_days_back=365)
        if stock_df.empty:
            return {"ticker": ticker, "status": "error", "reason": "No stock data"}

        # Score sentiment
        import pandas as pd
        pipe   = SentimentPipeline()
        scored = pipe.score(news_df) if not news_df.empty else pd.DataFrame()
        daily  = pipe.aggregate_daily(scored) if not scored.empty else pd.DataFrame()

        # Build features
        fe       = FeatureEngineer()
        features = fe.build(daily, stock_df)
        X, y     = fe.split_features_target(features, target="next_day_direction")

        if len(X) < 30:
            return {"ticker": ticker, "status": "skipped", "reason": f"Only {len(X)} rows"}

        # Retrain
        model   = PredictionModel(n_folds=3)
        results = model.train_evaluate(X, y)
        acc     = results["summary"]["Random Forest"]["accuracy_mean"]

        # Log today's prediction
        X_latest   = X.iloc[[-1]]
        signal     = int(model.predict(X_latest)[0])
        confidence = float(max(model.predict_proba(X_latest)[0]))
        sentiment  = float(daily["mean_score"].iloc[-1]) if not daily.empty else 0.0
        trade_date = str(stock_df.iloc[-1]["date"])

        fb.log_prediction(
            ticker=ticker, trade_date=trade_date,
            predicted=signal, confidence=confidence, sentiment=sentiment,
        )

        # Mark unretrained feedback as incorporated
        unretrained = fb.get_unretrained()
        if not unretrained.empty:
            ticker_rows = unretrained[unretrained["ticker"] == ticker.upper()]
            if not ticker_rows.empty and "id" in ticker_rows.columns:
                fb.mark_retrained(ticker_rows["id"].tolist())
                logger.info(f"  Marked {len(ticker_rows)} rows as retrained.")

        signal_str = "BUY" if signal == 1 else ("SELL" if signal == -1 else "HOLD")
        logger.info(
            f"  ✓ {ticker}: accuracy={acc:.1%} signal={signal_str} "
            f"conf={confidence:.1%} sent={sentiment:+.3f}"
        )

        return {
            "ticker":     ticker,
            "status":     "ok",
            "accuracy":   round(acc * 100, 1),
            "signal":     signal_str,
            "confidence": round(confidence, 3),
        }

    except Exception as e:
        logger.error(f"  ✗ {ticker} failed: {e}")
        return {"ticker": ticker, "status": "error", "reason": str(e)}


def main():
    logger.info(f"Daily retrain started at {datetime.utcnow().isoformat()}Z")

    from src.feedback_logger import FeedbackLogger
    fb = FeedbackLogger()

    # Test connection
    ok, msg = fb.test_connection()
    if not ok:
        logger.error(f"Supabase connection failed: {msg}")
        sys.exit(1)
    logger.info(f"Supabase: {msg}")

    # ── STEP 1: Fill all missing actuals first ─────────────────────────────────
    # This is the critical step that was missing before.
    # Resolves all NULL actuals using yfinance before retraining begins,
    # so get_unretrained() finds rows to work with.
    filled = fill_missing_actuals(fb)
    logger.info(f"Step 1 complete: filled {filled} actuals.")

    # ── STEP 2: Retrain all tickers ────────────────────────────────────────────
    tickers = get_tickers()
    results = []
    for ticker in tickers:
        result = retrain_ticker(ticker, fb)
        results.append(result)

    # ── STEP 3: Cleanup ────────────────────────────────────────────────────────
    logger.info("Step 3: Running cleanup...")
    deleted = fb.cleanup_old(keep_days=30)
    logger.info(f"Cleanup complete: {deleted} rows deleted.")

    # ── Summary ────────────────────────────────────────────────────────────────
    ok_count  = sum(1 for r in results if r["status"] == "ok")
    err_count = sum(1 for r in results if r["status"] == "error")
    skipped   = sum(1 for r in results if r["status"] == "skipped")

    logger.info("")
    logger.info("── Daily retrain summary ─────────────────────────────────")
    logger.info(f"  Actuals filled : {filled}")
    logger.info(f"  Retrained      : {ok_count}/{len(tickers)}")
    logger.info(f"  Errors         : {err_count}")
    logger.info(f"  Skipped        : {skipped}")
    logger.info(f"  Rows cleaned   : {deleted}")
    logger.info("")

    for r in results:
        if r["status"] == "ok":
            logger.info(f"  ✓ {r['ticker']:15s} acc={r['accuracy']}% signal={r['signal']} conf={r['confidence']:.1%}")
        elif r["status"] == "error":
            logger.info(f"  ✗ {r['ticker']:15s} ERROR: {r.get('reason','')}")
        else:
            logger.info(f"  ⚠ {r['ticker']:15s} SKIPPED: {r.get('reason','')}")

    if err_count > len(tickers) // 2:
        logger.error("More than half of tickers failed.")
        sys.exit(1)

    logger.info("Daily retrain complete.")


if __name__ == "__main__":
    main()
