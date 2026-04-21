"""
data_ingestion.py
-----------------
Module 1: Fetching financial news headlines and stock price data.

Sources:
    - NewsAPI (free tier) → headlines + timestamps
    - yfinance (no key needed) → OHLCV stock data

Usage:
    from src.data_ingestion import NewsIngester, StockIngester
"""

import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

NEWSAPI_BASE_URL = "https://newsapi.org/v2/everything"

# Free-tier NewsAPI only allows ~30 days back.
MAX_LOOKBACK_DAYS = 29


# ══════════════════════════════════════════════════════════════════════════════
# NEWS INGESTION
# ══════════════════════════════════════════════════════════════════════════════

class NewsIngester:
    """
    Fetches financial news headlines from NewsAPI.

    Args:
        api_key (str): Your NewsAPI key. Sign up free at https://newsapi.org
        page_size (int): Articles per request (max 100 on free tier).

    Example:
        ingester = NewsIngester(api_key="YOUR_KEY")
        df = ingester.fetch(ticker="AAPL", company_name="Apple", days_back=14)
    """

    def __init__(self, api_key: str, page_size: int = 100):
        if not api_key:
            raise ValueError("NewsAPI key is required. Get one free at https://newsapi.org")
        self.api_key = api_key
        self.page_size = page_size

    def fetch(
        self,
        ticker: str,
        company_name: str,
        days_back: int = 14,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Pull headlines mentioning a company over the past N days.

        Args:
            ticker:       Stock symbol, e.g. "AAPL"
            company_name: Human-readable name for the search query, e.g. "Apple"
            days_back:    How many days of history to fetch (max 29 on free tier)
            save:         Whether to save raw results to data/

        Returns:
            DataFrame with columns: [published_at, title, description, source, url]
        """
        days_back = min(days_back, MAX_LOOKBACK_DAYS)
        from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        to_date = datetime.utcnow().strftime("%Y-%m-%d")

        # We combine the ticker symbol + company name to improve recall.
        # e.g. query = "AAPL OR Apple" catches both kinds of mentions.
        query = f"{ticker} OR {company_name}"

        logger.info(f"Fetching news for '{query}' from {from_date} to {to_date} ...")

        params = {
            "q": query,
            "from": from_date,
            "to": to_date,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": self.page_size,
            "apiKey": self.api_key,
        }

        try:
            response = requests.get(NEWSAPI_BASE_URL, params=params, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"NewsAPI request failed: {e}")
            raise

        payload = response.json()

        if payload.get("status") != "ok":
            raise RuntimeError(f"NewsAPI error: {payload.get('message', 'Unknown error')}")

        articles = payload.get("articles", [])
        logger.info(f"Retrieved {len(articles)} articles for {ticker}.")

        if not articles:
            logger.warning("No articles returned. Try broadening your query or date range.")
            return pd.DataFrame()

        df = pd.DataFrame([
            {
                "published_at": art.get("publishedAt"),
                "title": art.get("title", ""),
                "description": art.get("description", ""),
                "source": art.get("source", {}).get("name", ""),
                "url": art.get("url", ""),
                "ticker": ticker.upper(),
            }
            for art in articles
            if art.get("title")  # drop articles with no title
        ])

        # ── Parse & normalise timestamps ───────────────────────────────────────
        # NewsAPI returns ISO-8601 UTC strings. We convert to timezone-aware
        # datetime and then extract just the date for later alignment.
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
        df["date"] = df["published_at"].dt.date  # plain date for market alignment
        df.sort_values("published_at", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)

        if save:
            out_path = DATA_DIR / f"news_{ticker.lower()}_{from_date}_{to_date}.csv"
            df.to_csv(out_path, index=False)
            logger.info(f"Saved news data → {out_path}")

        return df


# ══════════════════════════════════════════════════════════════════════════════
# STOCK PRICE INGESTION
# ══════════════════════════════════════════════════════════════════════════════

class StockIngester:
    """
    Fetches OHLCV stock price data using yfinance (no API key required).

    Example:
        ingester = StockIngester()
        df = ingester.fetch(ticker="AAPL", days_back=30)
    """

    def fetch(
        self,
        ticker: str,
        days_back: int = 30,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Download daily OHLCV data for a single ticker.

        Args:
            ticker:   Stock symbol, e.g. "AAPL"
            days_back: Number of calendar days of history to fetch
            save:     Whether to save raw results to data/

        Returns:
            DataFrame with columns: [date, open, high, low, close, volume,
                                      daily_return, direction]
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)

        logger.info(f"Fetching stock data for {ticker} ({start_date.date()} → {end_date.date()}) ...")

        try:
            raw = yf.download(
                ticker,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,  # adjusts for splits/dividends automatically
            )
        except Exception as e:
            logger.error(f"yfinance download failed: {e}")
            raise

        if raw.empty:
            logger.warning(f"No stock data returned for {ticker}. Check the ticker symbol.")
            return pd.DataFrame()

        # Flatten MultiIndex columns that yfinance sometimes returns
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = raw.copy()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df.index.name = "date"
        df.reset_index(inplace=True)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["ticker"] = ticker.upper()

        # ── Derived columns ────────────────────────────────────────────────────
        # daily_return: % change close-to-close. This is our prediction target.
        # direction: +1 (up), -1 (down), 0 (flat within ±0.1% noise band)
        df["daily_return"] = df["close"].pct_change() * 100  # as percentage
        df["direction"] = df["daily_return"].apply(self._classify_direction)

        # ── IMPORTANT: next-day target ─────────────────────────────────────────
        # When we later join with news sentiment, the news on day T should
        # predict the return on day T+1 (next trading day), NOT day T.
        # We pre-compute this here to make alignment explicit and avoid
        # accidental lookahead bias in downstream modules.
        df["next_day_return"] = df["daily_return"].shift(-1)
        df["next_day_direction"] = df["direction"].shift(-1)

        df.dropna(subset=["daily_return"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        logger.info(f"Fetched {len(df)} trading days for {ticker}.")

        if save:
            out_path = DATA_DIR / f"stocks_{ticker.lower()}_{start_date.date()}_{end_date.date()}.csv"
            df.to_csv(out_path, index=False)
            logger.info(f"Saved stock data → {out_path}")

        return df

    @staticmethod
    def _classify_direction(ret: float) -> int:
        """
        Convert a daily return % into a directional label.

        Returns:
             1  → stock went up (> +0.1%)
            -1  → stock went down (< -0.1%)
             0  → effectively flat (within ±0.1% noise band)
        """
        if pd.isna(ret):
            return 0
        if ret > 0.1:
            return 1
        if ret < -0.1:
            return -1
        return 0


# ══════════════════════════════════════════════════════════════════════════════
# ALIGNMENT UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def align_news_to_market(
    news_df: pd.DataFrame,
    stock_df: pd.DataFrame,
    close_hour: int = 16,
    timezone: str = "America/New_York",
) -> pd.DataFrame:
    """
    Align news dates to the correct trading day.

    The rule:
        - News published BEFORE market close on a trading day → assigned to THAT day.
        - News published AFTER market close, or on a weekend/holiday
          → assigned to the NEXT trading day.

    Supports international markets via close_hour and timezone parameters.

    Args:
        news_df   : Output of NewsIngester.fetch() or RSSIngester.fetch()
        stock_df  : Output of StockIngester.fetch()
        close_hour: Local market close hour (16 = 4pm, 15 = 3pm for India)
        timezone  : Market timezone string e.g. "America/New_York", "Asia/Kolkata"

    Returns:
        news_df with an added 'market_date' column aligned to trading days.
    """
    if news_df.empty or stock_df.empty:
        logger.warning("One or both DataFrames are empty. Skipping alignment.")
        return news_df

    trading_days = sorted(stock_df["date"].unique())

    def _assign_market_date(published_at):
        """Map a UTC publish timestamp to the appropriate trading day."""
        import pytz

        tz      = pytz.timezone(timezone)
        pub_local = published_at.astimezone(tz)
        pub_date  = pub_local.date()

        # After market close → next trading day
        if pub_local.hour >= close_hour:
            pub_date = pub_date + timedelta(days=1)

        # Walk forward to next actual trading day (handles weekends + holidays)
        for td in trading_days:
            if td >= pub_date:
                return td

        return None  # news is beyond our stock data window

    news_df = news_df.copy()
    news_df["market_date"] = news_df["published_at"].apply(_assign_market_date)
    news_df.dropna(subset=["market_date"], inplace=True)

    logger.info(
        f"Aligned {len(news_df)} articles to {news_df['market_date'].nunique()} trading days."
    )
    return news_df


# ══════════════════════════════════════════════════════════════════════════════
# QUICK TEST  (run: python -m src.data_ingestion)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os

    # ── Stock data (no key needed) ─────────────────────────────────────────────
    stock_ingester = StockIngester()
    aapl_stocks = stock_ingester.fetch(ticker="AAPL", days_back=30)
    print("\n── Stock sample ──")
    print(aapl_stocks[["date", "close", "daily_return", "direction"]].tail(5))

    # ── News data (requires key) ───────────────────────────────────────────────
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
    if not NEWS_API_KEY:
        print("\n⚠ Set NEWS_API_KEY env var to test news ingestion.")
        print("  export NEWS_API_KEY=your_key_here")
    else:
        news_ingester = NewsIngester(api_key=NEWS_API_KEY)
        aapl_news = news_ingester.fetch(ticker="AAPL", company_name="Apple", days_back=14)

        aligned = align_news_to_market(aapl_news, aapl_stocks)
        print("\n── News sample (aligned) ──")
        print(aligned[["market_date", "title", "source"]].head(5))
