"""
load_data.py
------------
Smart data loader — single entry point for all downstream modules.

Priority order:
    1. RSS feeds (default) — no key, 60-180 days of news history
    2. NewsAPI (optional)  — set NEWS_API_KEY env var for live API data
    3. Sample data         — pre-fetched CSVs committed to the repo

Usage:
    from src.load_data import load

    news_df, stock_df = load("AAPL")
"""

import os
import logging
from pathlib import Path

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger    = logging.getLogger(__name__)
ROOT      = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT / "data" / "sample"

SAMPLE_TICKERS = {"AAPL", "MSFT", "GOOGL"}

COMPANY_NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google",
    "AMZN": "Amazon", "META": "Meta", "TSLA": "Tesla",
    "NVDA": "NVIDIA", "NFLX": "Netflix",
}


def load(ticker: str, days_back: int = 365, prefer_rss: bool = True):
    """
    Load news and stock data for a ticker.

    Args:
        ticker     : Stock symbol, e.g. "AAPL"
        days_back  : Days of stock history to fetch (default: 365)
        prefer_rss : Use RSS feeds first (default: True)
                     Set False to force NewsAPI if key is available.

    Returns:
        (news_df, stock_df) — ready for Module 2
    """
    ticker  = ticker.upper()
    api_key = os.getenv("NEWS_API_KEY", "").strip()

    if prefer_rss:
        return _load_rss(ticker, days_back)
    elif api_key and api_key != "your_newsapi_key_here":
        return _load_newsapi(ticker, api_key, days_back)
    else:
        return _load_sample(ticker)


# ── RSS mode (default) ─────────────────────────────────────────────────────────

def _load_rss(ticker: str, days_back: int):
    from src.rss_ingester import fetch_rss_and_stock
    return fetch_rss_and_stock(
        ticker          = ticker,
        company_name    = COMPANY_NAMES.get(ticker, ticker),
        stock_days_back = days_back,
    )


# ── NewsAPI mode (optional, requires key) ─────────────────────────────────────

def _load_newsapi(ticker: str, api_key: str, days_back: int):
    from src.data_ingestion import NewsIngester, StockIngester, align_news_to_market

    logger.info(f"[newsapi] Fetching data for {ticker} ...")
    company  = COMPANY_NAMES.get(ticker, ticker)
    stock_df = StockIngester().fetch(ticker=ticker, days_back=days_back)
    news_df  = NewsIngester(api_key=api_key).fetch(
        ticker=ticker, company_name=company, days_back=min(days_back, 14)
    )
    news_df = align_news_to_market(news_df, stock_df)
    return news_df, stock_df


# ── Sample data mode (fallback) ───────────────────────────────────────────────

def _load_sample(ticker: str):
    news_path  = SAMPLE_DIR / f"news_{ticker.lower()}.csv"
    stock_path = SAMPLE_DIR / f"stock_{ticker.lower()}.csv"

    if not news_path.exists() or not stock_path.exists():
        raise FileNotFoundError(
            f"No sample data for {ticker}.\n"
            f"Available: {sorted(SAMPLE_TICKERS)}\n"
            f"Run: python fetch_sample_data.py"
        )

    logger.info(f"[sample] Loading pre-fetched data for {ticker}.")
    news_df  = pd.read_csv(news_path,  parse_dates=["published_at"])
    stock_df = pd.read_csv(stock_path, parse_dates=["date"])
    news_df["date"]  = pd.to_datetime(news_df["date"]).dt.date
    stock_df["date"] = pd.to_datetime(stock_df["date"]).dt.date
    return news_df, stock_df
