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
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("daily_retrain")

# ── Tickers to retrain ─────────────────────────────────────────────────────────
# Default: representative set across all markets.
# Override via TICKERS_OVERRIDE env var (comma-separated).

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


def update_actuals(fb, ticker: str, stock_df) -> None:
    """
    Fill in actual outcomes for yesterday's prediction.
    Called first so today's retraining has the freshest labels.
    """
    if len(stock_df) < 2:
        return
    yesterday      = stock_df.iloc[-2]
    yesterday_date = str(yesterday["date"])
    actual         = int(yesterday.get("direction", 0))
    if actual != 0:
        fb.update_actual(ticker=ticker, trade_date=yesterday_date, actual=actual)


def retrain_ticker(ticker: str, fb) -> dict:
    """Run the full pipeline and retrain for a single ticker."""
    from src.rss_ingester import fetch_rss_and_stock
    from src.sentiment_model import SentimentPipeline
    from src.feature_engineering import FeatureEngineer
    from src.prediction import PredictionModel
    from src.backtesting import Backtester

    logger.info(f"── {ticker} ──────────────────────────────────")

    try:
        # Fetch data
        news_df, stock_df = fetch_rss_and_stock(ticker, stock_days_back=365)
        if stock_df.empty:
            return {"ticker": ticker, "status": "error", "reason": "No stock data"}

        # Update yesterday's actual before retraining
        update_actuals(fb, ticker, stock_df)

        # Score sentiment
        pipe   = SentimentPipeline()
        scored = pipe.score(news_df) if not news_df.empty else __import__("pandas").DataFrame()
        daily  = pipe.aggregate_daily(scored) if not scored.empty else __import__("pandas").DataFrame()

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
        proba      = model.predict_proba(X_latest)[0]
        confidence = float(max(proba))
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

    # Test Supabase connection first
    from src.feedback_logger import FeedbackLogger
    fb = FeedbackLogger()
    ok, msg = fb.test_connection()
    if not ok:
        logger.error(f"Supabase connection failed: {msg}")
        logger.error("Set SUPABASE_URL and SUPABASE_KEY environment variables.")
        sys.exit(1)
    logger.info(f"Supabase: {msg}")

    tickers = get_tickers()
    results = []

    for ticker in tickers:
        result = retrain_ticker(ticker, fb)
        results.append(result)

    # ── Cleanup ────────────────────────────────────────────────────────────────
    logger.info("Running cleanup...")
    deleted = fb.cleanup_old(keep_days=30)
    logger.info(f"Cleanup complete: {deleted} rows deleted.")

    # ── Summary ────────────────────────────────────────────────────────────────
    ok_count  = sum(1 for r in results if r["status"] == "ok")
    err_count = sum(1 for r in results if r["status"] == "error")
    skipped   = sum(1 for r in results if r["status"] == "skipped")

    logger.info("")
    logger.info("── Daily retrain summary ─────────────────────────────────")
    logger.info(f"  Completed : {ok_count}/{len(tickers)}")
    logger.info(f"  Errors    : {err_count}")
    logger.info(f"  Skipped   : {skipped}")
    logger.info(f"  Rows cleaned: {deleted}")
    logger.info("")

    for r in results:
        if r["status"] == "ok":
            logger.info(f"  ✓ {r['ticker']:15s} acc={r['accuracy']}% signal={r['signal']} conf={r['confidence']:.1%}")
        elif r["status"] == "error":
            logger.info(f"  ✗ {r['ticker']:15s} ERROR: {r.get('reason','')}")
        else:
            logger.info(f"  ⚠ {r['ticker']:15s} SKIPPED: {r.get('reason','')}")

    # Exit with error code if too many failures (useful for GitHub Actions alerts)
    if err_count > len(tickers) // 2:
        logger.error("More than half of tickers failed — exiting with error.")
        sys.exit(1)

    logger.info("Daily retrain complete.")


if __name__ == "__main__":
    main()
