"""
api/app.py
----------
Module 6: FastAPI REST API exposing the sentiment pipeline.

Endpoints:
    GET  /                          health check + API info
    GET  /stock/{ticker}            latest stock data + sentiment score
    POST /predict                   predict next-day direction for a ticker
    GET  /backtest/{ticker}         run and return backtest metrics
    GET  /docs                      auto-generated Swagger UI (free from FastAPI)

Run locally:
    uvicorn api.app:app --reload --port 8000

Then visit:
    http://localhost:8000/docs      → interactive Swagger UI
    http://localhost:8000/predict   → try the API
"""

import logging
import time
from datetime import datetime
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Financial Sentiment Trading Signal API",
    description = (
        "Scores financial news sentiment with FinBERT and generates "
        "next-day trading signals for stocks."
    ),
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# Allow requests from any origin (needed for Streamlit dashboard in Module 7)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Pydantic models (request / response schemas) ───────────────────────────────

class PredictRequest(BaseModel):
    ticker : str = Field(..., example="AAPL", description="Stock ticker symbol")
    days_back: int = Field(365, ge=30, le=730, description="Days of stock history")

class SentimentScore(BaseModel):
    mean_score    : float
    net_sentiment : float
    n_articles    : int
    label         : str   # "bullish" | "bearish" | "neutral"

class PredictResponse(BaseModel):
    ticker          : str
    signal          : int           # +1 buy, -1 sell, 0 hold
    signal_label    : str           # "BUY" | "SELL" | "HOLD"
    confidence      : float         # model probability of predicted class
    sentiment       : SentimentScore
    latest_close    : float
    generated_at    : str           # ISO timestamp

class StockResponse(BaseModel):
    ticker       : str
    latest_close : float
    daily_return : float
    direction    : int
    sentiment    : Optional[SentimentScore]
    fetched_at   : str

class BacktestResponse(BaseModel):
    ticker              : str
    total_return        : float
    annual_return       : float
    sharpe_ratio        : float
    max_drawdown        : float
    win_rate            : float
    n_trades            : int
    buy_and_hold_return : float
    vs_buy_and_hold     : float
    period_days         : int
    computed_at         : str


# ── Pipeline cache ─────────────────────────────────────────────────────────────
# The pipeline is expensive to initialise (FinBERT ~440MB).
# We cache it so it's only loaded once per server process.

_pipeline_cache: dict = {}

def get_pipeline():
    """Load and cache the full pipeline (lazy, thread-safe enough for dev)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    if "pipe" not in _pipeline_cache:
        from src.sentiment_model import SentimentPipeline
        logger.info("Loading FinBERT pipeline (one-time)...")
        _pipeline_cache["pipe"] = SentimentPipeline()
        logger.info("Pipeline ready.")

    return _pipeline_cache["pipe"]


def run_full_pipeline(ticker: str, days_back: int = 365):
    """Run the full data → sentiment → features pipeline for a ticker."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from src.rss_ingester import fetch_rss_and_stock
    from src.feature_engineering import FeatureEngineer

    pipe = get_pipeline()

    news_df, stock_df = fetch_rss_and_stock(
        ticker          = ticker,
        stock_days_back = days_back,
    )

    scored   = pipe.score(news_df)
    daily    = pipe.aggregate_daily(scored)

    fe       = FeatureEngineer()
    features = fe.build(daily, stock_df)

    return features, stock_df, daily


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Health check — confirms the API is running."""
    return {
        "status":    "ok",
        "service":   "Financial Sentiment Trading Signal API",
        "version":   "1.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "endpoints": [
            "GET  /stock/{ticker}    — latest stock + sentiment",
            "POST /predict           — next-day trading signal",
            "GET  /backtest/{ticker} — backtest metrics",
            "GET  /docs             — Swagger UI",
        ],
    }


@app.get("/stock/{ticker}", response_model=StockResponse, tags=["Market Data"])
def get_stock(
    ticker: str,
    days_back: int = Query(30, ge=7, le=365),
):
    """
    Fetch the latest stock data and most recent sentiment score for a ticker.

    Returns the latest closing price, daily return, and a sentiment snapshot
    from the most recent news available via RSS.
    """
    ticker = ticker.upper()
    logger.info(f"GET /stock/{ticker}")

    try:
        features, stock_df, daily = run_full_pipeline(ticker, days_back)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latest_stock = stock_df.iloc[-1]
    latest_sent  = daily.iloc[-1] if not daily.empty else None

    sentiment = None
    if latest_sent is not None:
        score = float(latest_sent.get("mean_score", 0))
        sentiment = SentimentScore(
            mean_score    = round(score, 4),
            net_sentiment = round(float(latest_sent.get("net_sentiment", 0)), 4),
            n_articles    = int(latest_sent.get("n_articles", 0)),
            label         = _sentiment_label(score),
        )

    return StockResponse(
        ticker       = ticker,
        latest_close = round(float(latest_stock["close"]), 2),
        daily_return = round(float(latest_stock["daily_return"]), 4),
        direction    = int(latest_stock["direction"]),
        sentiment    = sentiment,
        fetched_at   = datetime.utcnow().isoformat() + "Z",
    )


@app.post("/predict", response_model=PredictResponse, tags=["Signals"])
def predict(req: PredictRequest):
    """
    Generate a next-day trading signal for a ticker.

    Uses the trained Random Forest model to predict whether the stock
    will go UP (+1), DOWN (-1), or FLAT (0) tomorrow, based on
    current sentiment features and recent price action.

    Returns the signal, confidence score, and supporting sentiment data.
    """
    ticker = req.ticker.upper()
    logger.info(f"POST /predict — ticker={ticker}")

    try:
        features, stock_df, daily = run_full_pipeline(ticker, req.days_back)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Train model on available data
    try:
        from src.feature_engineering import FeatureEngineer
        from src.prediction import PredictionModel

        fe = FeatureEngineer()
        X, y = fe.split_features_target(features, target="next_day_direction")

        if len(X) < 30:
            raise HTTPException(
                status_code=422,
                detail=f"Not enough data for {ticker}. Need at least 30 rows, got {len(X)}."
            )

        model = PredictionModel()
        model.train_evaluate(X, y)

        # Predict on the most recent row
        X_latest = X.iloc[[-1]]
        signal   = int(model.predict(X_latest)[0])
        proba    = model.predict_proba(X_latest)[0]
        confidence = round(float(max(proba)), 3)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")

    # Sentiment snapshot
    latest_sent = daily.iloc[-1] if not daily.empty else None
    score       = float(latest_sent.get("mean_score", 0)) if latest_sent is not None else 0.0

    sentiment = SentimentScore(
        mean_score    = round(score, 4),
        net_sentiment = round(float(latest_sent.get("net_sentiment", 0)), 4) if latest_sent is not None else 0.0,
        n_articles    = int(latest_sent.get("n_articles", 0)) if latest_sent is not None else 0,
        label         = _sentiment_label(score),
    )

    signal_labels = {1: "BUY", -1: "SELL", 0: "HOLD"}

    return PredictResponse(
        ticker       = ticker,
        signal       = signal,
        signal_label = signal_labels.get(signal, "HOLD"),
        confidence   = confidence,
        sentiment    = sentiment,
        latest_close = round(float(stock_df["close"].iloc[-1]), 2),
        generated_at = datetime.utcnow().isoformat() + "Z",
    )


@app.get("/backtest/{ticker}", response_model=BacktestResponse, tags=["Backtest"])
def backtest(
    ticker: str,
    days_back: int = Query(365, ge=90, le=730),
):
    """
    Run the full backtest for a ticker and return performance metrics.

    Trains the model on the first 70% of data and backtests on the
    remaining 30%, with transaction costs included.
    """
    ticker = ticker.upper()
    logger.info(f"GET /backtest/{ticker}")

    try:
        features, stock_df, daily = run_full_pipeline(ticker, days_back)

        from src.feature_engineering import FeatureEngineer
        from src.prediction import PredictionModel
        from src.backtesting import Backtester

        fe       = FeatureEngineer()
        X, y     = fe.split_features_target(features, target="next_day_direction")
        model    = PredictionModel()
        model.train_evaluate(X, y)

        bt     = Backtester()
        report = bt.run(features, model)
        m      = report["metrics"]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return BacktestResponse(
        ticker              = ticker,
        total_return        = m["total_return"],
        annual_return       = m["annual_return"],
        sharpe_ratio        = m["sharpe_ratio"],
        max_drawdown        = m["max_drawdown"],
        win_rate            = m["win_rate"],
        n_trades            = m["n_trades"],
        buy_and_hold_return = m["buy_and_hold_return"],
        vs_buy_and_hold     = m["vs_buy_and_hold"],
        period_days         = m["n_days"],
        computed_at         = datetime.utcnow().isoformat() + "Z",
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sentiment_label(score: float) -> str:
    if score > 0.05:
        return "bullish"
    if score < -0.05:
        return "bearish"
    return "neutral"
