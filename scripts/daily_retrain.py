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

# Resolve project root — works whether called from repo root, fintech-sentiment/,
# or scripts/ directory directly
_script_dir = Path(__file__).resolve().parent      # scripts/
ROOT        = _script_dir.parent                   # fintech-sentiment/ or repo root

# If src/ isn't directly in ROOT, check if we're one level too deep
if not (ROOT / "src").exists():
    ROOT = ROOT.parent

sys.path.insert(0, str(ROOT))
logger_setup = logging.getLogger("path_setup")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("daily_retrain")

# ── Expanded ticker universe ──────────────────────────────────────────────────
# These are retrained every day by GitHub Actions regardless of user visits.
# The model improves continuously for all these stocks even with zero traffic.
# Costs: ~14 tickers × ~90s = ~21 minutes, well within 2000 min/month free quota.
# To add more tickers, simply append here and push to GitHub.

# ── Full ticker universe — top 10 per market ──────────────────────────────────
# Defined here for reference and weekly/manual runs via TICKERS_OVERRIDE.
# Full 100-ticker run = ~150 min, exceeds GitHub free tier (2,000 min/month).
# Daily run uses DAILY_TICKERS (25 tickers, ~37 min) to stay within quota.

ALL_TICKERS = [
    # USA (NYSE / NASDAQ)
    "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN",
    "META", "TSLA", "BRK-B", "AVGO", "LLY",
    # UK (LSE — .L)
    "SHEL.L", "HSBA.L", "ULVR.L", "BP.L", "AZN.L",
    "GSK.L", "DGE.L", "RIO.L", "BATS.L", "BARC.L",
    # Germany (XETRA — .DE)
    "SAP.DE", "SIE.DE", "VOW3.DE", "MBG.DE", "ALV.DE",
    "BAS.DE", "BMW.DE", "DTE.DE", "IFX.DE", "ADS.DE",
    # Netherlands (Euronext Amsterdam — .AS)
    # NXP Semiconductors trades NASDAQ (NXPI); Airbus primary listing Paris (AIR.PA)
    "ASML.AS", "PRX.AS", "INGA.AS", "HEIA.AS", "AGN.AS",
    "ASRNL.AS", "ARGX", "RAND.AS", "NXPI", "AIR.PA",
    # India (NSE — .NS)
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "BHARTIARTL.NS", "LT.NS", "SBIN.NS", "ADANIENT.NS",
    # China / Hong Kong (HKEX — .HK)
    # Kweichow Moutai (600519.SS) is Shanghai A-share only — replaced with BIDU
    "0700.HK", "9988.HK", "1398.HK", "0939.HK", "1211.HK",
    "JD", "2318.HK", "0857.HK", "3690.HK", "BIDU",
    # Japan (TSE — .T)
    "7203.T", "6758.T", "8306.T", "9984.T", "6861.T",
    "7974.T", "8035.T", "6501.T", "7267.T", "8316.T",
    # France (Euronext Paris — .PA)
    "MC.PA", "TTE.PA", "OR.PA", "SU.PA", "AIR.PA",
    "SAN.PA", "BNP.PA", "CS.PA", "SAF.PA", "RMS.PA",
    # Switzerland (SIX — .SW)
    "NESN.SW", "ROG.SW", "NOVN.SW", "UBSG.SW", "ZURN.SW",
    "ABBN.SW", "CFR.SW", "SREN.SW", "LONN.SW", "HOLN.SW",
    # Italy (Borsa Italiana — .MI)
    "ENEL.MI", "ENI.MI", "RACE.MI", "ISP.MI", "UCG.MI",
    "STLAM.MI", "PIRC.MI", "TIT.MI", "LDO.MI", "MONC.MI",
]

# ── DEFAULT_TICKERS — full list from your specification ───────────────────────
# Split into BATCH_A (Mon/Wed/Fri) and BATCH_B (Tue/Thu) to stay within
# GitHub Actions free tier (2,000 min/month).
# Each batch ~50 tickers × 90s = ~75 min/run.
# BATCH_A: 3 runs/week × 75 min = 225 min/week × 4.3 = ~968 min/month
# BATCH_B: 2 runs/week × 75 min = 150 min/week × 4.3 = ~645 min/month
# Total: ~1,613 min/month — within the 2,000 min free quota.

BATCH_A = [
    # USA (NYSE / NASDAQ) — all 10
    "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN",
    "META", "TSLA", "BRK-B", "AVGO", "LLY",
    # India (NSE) — all 10
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "BHARTIARTL.NS", "LT.NS", "SBIN.NS", "ADANIENT.NS",
    # UK (LSE) — all 10
    "SHEL.L", "HSBA.L", "ULVR.L", "BP.L", "AZN.L",
    "GSK.L", "DGE.L", "RIO.L", "BATS.L", "BARC.L",
    # Netherlands (AEX) — all 8 + 2 exceptions
    "ASML.AS", "PRX.AS", "INGA.AS", "HEIA.AS", "AGN.AS",
    "ASRNL.AS", "ARGX", "RAND.AS", "NXPI", "AIR.PA",
    # France (Euronext Paris) — first 5
    "MC.PA", "TTE.PA", "OR.PA", "SU.PA", "SAN.PA",
]

BATCH_B = [
    # Germany (XETRA) — all 10
    "SAP.DE", "SIE.DE", "VOW3.DE", "MBG.DE", "ALV.DE",
    "BAS.DE", "BMW.DE", "DTE.DE", "IFX.DE", "ADS.DE",
    # France (Euronext Paris) — remaining 5
    "BNP.PA", "CS.PA", "SAF.PA", "RMS.PA", "AIR.PA",
    # Switzerland (SIX) — all 10
    "NESN.SW", "ROG.SW", "NOVN.SW", "UBSG.SW", "ZURN.SW",
    "ABBN.SW", "CFR.SW", "SREN.SW", "LONN.SW", "HOLN.SW",
    # Italy (Borsa Italiana) — all 10
    "ENEL.MI", "ENI.MI", "RACE.MI", "ISP.MI", "UCG.MI",
    "STLAM.MI", "PIRC.MI", "TIT.MI", "LDO.MI", "MONC.MI",
    # China / HK (HKEX) — all 10
    "0700.HK", "9988.HK", "1398.HK", "0939.HK", "1211.HK",
    "JD", "2318.HK", "0857.HK", "3690.HK", "BIDU",
    # Japan (TSE) — all 10
    "7203.T", "6758.T", "8306.T", "9984.T", "6861.T",
    "7974.T", "8035.T", "6501.T", "7267.T", "8316.T",
]

# DEFAULT_TICKERS is selected by the workflow based on the day of week.
# daily_retrain.py reads the DAY_BATCH env var set by the GitHub Actions workflow.
import datetime as _dt
_dow = _dt.datetime.utcnow().weekday()   # 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun
_batch_override = os.getenv("DAY_BATCH", "")
if _batch_override == "A":
    DEFAULT_TICKERS = BATCH_A
elif _batch_override == "B":
    DEFAULT_TICKERS = BATCH_B
elif _dow in (0, 2, 4, 5):   # Mon, Wed, Fri, Sat → Batch A
    DEFAULT_TICKERS = BATCH_A
else:                          # Tue, Thu, Sun → Batch B
    DEFAULT_TICKERS = BATCH_B


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
                # end = today + 2 ensures the most recent closed trading day
                # is always included (yfinance end is exclusive).
                # For weekend runs: Saturday/Sunday have no trading data —
                # the price_map will simply not contain those dates, and the
                # walk-forward loop will correctly skip to the next trading day.
                # Friday's close is the last available and is used for
                # predictions made on Friday, Saturday, or Sunday.
                df = yf.download(
                    ticker,
                    start=(datetime.utcnow() - timedelta(days=20)).strftime("%Y-%m-%d"),
                    end=(datetime.utcnow() + timedelta(days=2)).strftime("%Y-%m-%d"),
                    progress=False,
                    auto_adjust=True,
                )
                if df.empty:
                    continue

                # Flatten MultiIndex if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # Keep index as DatetimeIndex — use it directly for date lookup
                df["daily_return"] = df["Close"].pct_change() * 100
                df["direction"]    = df["daily_return"].apply(
                    lambda r: 1 if r > 0.1 else (-1 if r < -0.1 else 0)
                )

                # Build a date → direction lookup
                # Use the DatetimeIndex directly — most reliable approach
                # df.index is a DatetimeIndex; convert to YYYY-MM-DD strings
                price_map = {}
                for idx, row in df.iterrows():
                    if pd.isna(row["direction"]):
                        continue
                    # idx is a Timestamp — use strftime directly on it
                    key = pd.Timestamp(idx).strftime("%Y-%m-%d")
                    price_map[key] = int(row["direction"])

                logger.info(f"  {ticker}: {len(price_map)} days, latest={max(price_map.keys()) if price_map else 'none'}")
                logger.info(f"  {ticker}: all dates = {sorted(price_map.keys())}")

                for pred_row in rows:
                    td = pred_row["trade_date"][:10]   # ensure YYYY-MM-DD

                    # We need the return on trade_date+1
                    # (prediction made on T, outcome is T+1 return)
                    trade_dt = datetime.strptime(td, "%Y-%m-%d").date()
                    next_day = trade_dt + timedelta(days=1)

                    # Walk forward up to 5 days to find next trading day
                    actual = None
                    for offset in range(5):
                        check = (next_day + timedelta(days=offset)).strftime("%Y-%m-%d")
                        if check in price_map:
                            actual = price_map[check]
                            logger.info(f"  Found actual for {ticker} {td}: {check} → direction={actual}")
                            break

                    if actual is not None:
                        # Write actual regardless of value (including 0 for flat days)
                        # so the row is marked as resolved and not re-processed
                        correct = int(pred_row["predicted"]) == actual if actual != 0 else None
                        update_payload = {"actual": actual}
                        if correct is not None:
                            update_payload["correct"] = correct
                        client.table("predictions").update(update_payload).eq("id", pred_row["id"]).execute()
                        updated += 1
                        logger.info(
                            f"  Filled: {ticker} {td} actual={actual} "
                            f"predicted={pred_row['predicted']} "
                            f"correct={correct}"
                        )
                    else:
                        logger.info(f"  No next-day data yet for {ticker} {td} (looking for {next_day} onwards) — will retry tomorrow")

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
        # On weekends, stock_df.iloc[-1] is still Friday's close — correct.
        # We log trade_date as today so the prediction is linked to
        # the day it was made (weekend news context included).
        import datetime as _dt_mod
        _today    = _dt_mod.date.today()
        _weekday  = _today.weekday()   # 5=Sat, 6=Sun
        if _weekday >= 5:
            # Weekend — use today's date as trade_date but Friday's price
            trade_date = _today.isoformat()
        else:
            trade_date = str(stock_df.iloc[-1]["date"])

        X_latest   = X.iloc[[-1]]
        signal     = int(model.predict(X_latest)[0])
        confidence = float(max(model.predict_proba(X_latest)[0]))
        sentiment  = float(daily["mean_score"].iloc[-1]) if not daily.empty else 0.0

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
    deleted = fb.cleanup_old(keep_days=7)
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
