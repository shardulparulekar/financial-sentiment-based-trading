"""
rss_ingester.py
---------------
Free news ingestion via RSS feeds — no API key, no rate limits.
Supports US, Indian, European, Asian, and other international markets.

Sources by region:
    Global  : Google News, Seeking Alpha
    India   : Economic Times Markets, Moneycontrol, Google News India
    Europe  : Google News Europe
    Asia/HK : Google News Asia

Usage:
    from src.rss_ingester import RSSIngester, fetch_rss_and_stock

    news_df, stock_df = fetch_rss_and_stock("RELIANCE.NS")
    news_df, stock_df = fetch_rss_and_stock("AAPL")
    news_df, stock_df = fetch_rss_and_stock("ASML.AS")
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ── RSS feed registry ─────────────────────────────────────────────────────────
# {query}  = company name (URL encoded)
# {ticker} = raw ticker symbol

RSS_FEEDS = {
    # ── Global ────────────────────────────────────────────────────────────────
    "seeking_alpha": (
        "https://seekingalpha.com/api/sa/combined/{base_ticker}.xml"
    ),
    "google_news": (
        "https://news.google.com/rss/search"
        "?q={query}+stock&hl=en-US&gl=US&ceid=US:en"
    ),

    # ── India ─────────────────────────────────────────────────────────────────
    "economic_times": (
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"
    ),
    "moneycontrol": (
        "https://www.moneycontrol.com/rss/marketreports.xml"
    ),
    "google_news_india": (
        "https://news.google.com/rss/search"
        "?q={query}+share+price&hl=en-IN&gl=IN&ceid=IN:en"
    ),

    # ── Europe ────────────────────────────────────────────────────────────────
    "google_news_europe": (
        "https://news.google.com/rss/search"
        "?q={query}+stock&hl=en-GB&gl=GB&ceid=GB:en"
    ),

    # ── Asia / HK / China ─────────────────────────────────────────────────────
    "google_news_asia": (
        "https://news.google.com/rss/search"
        "?q={query}+stock&hl=en-HK&gl=HK&ceid=HK:en"
    ),
}

# Which sources to use per region
REGION_SOURCES = {
    "IN": ["economic_times", "moneycontrol", "google_news_india"],
    "HK": ["google_news_asia", "seeking_alpha"],
    "CN": ["google_news_asia", "seeking_alpha"],
    "JP": ["google_news_asia", "seeking_alpha"],
    "GB": ["google_news_europe", "seeking_alpha"],
    "DE": ["google_news_europe", "seeking_alpha"],
    "FR": ["google_news_europe", "seeking_alpha"],
    "NL": ["google_news_europe", "seeking_alpha"],
    "CH": ["google_news_europe", "seeking_alpha"],
    "US": ["seeking_alpha", "google_news"],
}


class RSSIngester:
    """
    Fetches financial news from RSS feeds for any market.

    Args:
        sources : Override source list (default: auto-detected from market)
        timeout : Request timeout in seconds (default: 10)

    Example:
        ingester = RSSIngester()
        df = ingester.fetch(ticker="RELIANCE.NS", company_name="Reliance Industries")
        df = ingester.fetch(ticker="AAPL", company_name="Apple")
    """

    def __init__(
        self,
        sources: list[str] | None = None,
        timeout: int = 10,
    ):
        self.sources_override = sources
        self.timeout = timeout

    def fetch(
        self,
        ticker: str,
        company_name: str,
        market_info: dict | None = None,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch headlines from appropriate RSS sources for the ticker's market.

        Args:
            ticker       : Stock symbol e.g. "RELIANCE.NS", "AAPL", "ASML.AS"
            company_name : Human name for search e.g. "Reliance Industries"
            market_info  : Output of get_market_info() — auto-detected if None
            save         : Whether to save results to data/

        Returns:
            DataFrame with columns:
            [published_at, title, description, source, url, ticker, date]
        """
        try:
            import feedparser
        except ImportError:
            raise ImportError(
                "feedparser is required.\n"
                "Install: pip install feedparser"
            )

        from src.ticker_utils import get_market_info as _get_market_info
        market_info = market_info or _get_market_info(ticker)
        region      = market_info.get("news_region", "US")

        # Pick sources based on region, unless overridden
        sources = self.sources_override or REGION_SOURCES.get(region, REGION_SOURCES["US"])

        # Base ticker without exchange suffix (for Seeking Alpha)
        base_ticker = ticker.split(".")[0]
        query       = company_name.replace(" ", "+")

        logger.info(
            f"Fetching RSS for {ticker} ({market_info['market']}) "
            f"from {len(sources)} sources..."
        )

        all_articles = []

        for source_key in sources:
            if source_key not in RSS_FEEDS:
                logger.warning(f"Unknown source '{source_key}' — skipping.")
                continue

            url = RSS_FEEDS[source_key].format(
                ticker      = ticker,
                base_ticker = base_ticker,
                query       = query,
            )

            try:
                articles = self._fetch_feed(source_key, url, ticker, feedparser)
                logger.info(f"  {source_key}: {len(articles)} articles")
                all_articles.extend(articles)
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"  {source_key} failed: {e} — skipping.")
                continue

        if not all_articles:
            logger.warning("No articles retrieved from any RSS source.")
            return pd.DataFrame()

        df = pd.DataFrame(all_articles)

        # ── Normalise timestamps ───────────────────────────────────────────────
        df["published_at"] = pd.to_datetime(
            df["published_at"], utc=True, errors="coerce"
        )
        df.dropna(subset=["published_at"], inplace=True)
        df["date"] = df["published_at"].dt.date

        # ── Filter to company-relevant articles for broad feeds ────────────────
        # Economic Times and Moneycontrol return all market news,
        # not just articles about our specific company.
        # Filter to keep only articles mentioning the company name.
        broad_sources = {"economic_times", "moneycontrol"}
        broad_mask    = df["source"].isin(broad_sources)
        if broad_mask.any():
            name_parts  = company_name.lower().split()[:2]  # first two words
            title_lower = df["title"].str.lower()
            relevant    = title_lower.apply(
                lambda t: any(part in t for part in name_parts)
            )
            # Keep non-broad source rows always; filter broad source rows
            df = df[~broad_mask | relevant]
            logger.info(
                f"After relevance filter: {len(df)} articles "
                f"(was {len(broad_mask)} before filter)"
            )

        # ── Deduplicate ────────────────────────────────────────────────────────
        before = len(df)
        df.drop_duplicates(subset=["title"], keep="first", inplace=True)
        if before - len(df) > 0:
            logger.info(f"Removed {before - len(df)} duplicate headlines.")

        df.sort_values("published_at", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)

        if not df.empty:
            logger.info(
                f"Total: {len(df)} articles for {ticker} "
                f"({df['date'].min()} → {df['date'].max()})"
            )

        if save and not df.empty:
            today    = datetime.now().strftime("%Y-%m-%d")
            out_path = DATA_DIR / f"rss_{ticker.lower().replace('.', '_')}_{today}.csv"
            df.to_csv(out_path, index=False)
            logger.info(f"Saved → {out_path}")

        return df

    def _fetch_feed(self, source_key, url, ticker, feedparser):
        response = requests.get(url, headers=HEADERS, timeout=self.timeout)
        response.raise_for_status()
        feed     = feedparser.parse(response.content)

        articles = []
        for entry in feed.entries:
            title = entry.get("title", "").strip()
            if not title:
                continue

            pub = entry.get("published_parsed") or entry.get("updated_parsed")
            published_at = (
                datetime(*pub[:6], tzinfo=timezone.utc).isoformat()
                if pub else
                datetime.now(timezone.utc).isoformat()
            )

            articles.append({
                "published_at": published_at,
                "title":        title,
                "description":  entry.get("summary", "")[:500],
                "source":       source_key,
                "url":          entry.get("link", ""),
                "ticker":       ticker.upper(),
            })

        return articles


# ── Main entry point ───────────────────────────────────────────────────────────

def fetch_rss_and_stock(
    ticker: str,
    company_name: str | None = None,
    stock_days_back: int = 365,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch RSS news + stock history for any ticker, any market.

    Handles exchange suffixes automatically:
        "AAPL"         → US market, ET close cutoff
        "RELIANCE.NS"  → India NSE, IST close cutoff
        "ASML.AS"      → Netherlands Euronext
        "0700.HK"      → Hong Kong HKEX

    Args:
        ticker         : yfinance-compatible ticker e.g. "RELIANCE.NS"
        company_name   : Override company name for news search
        stock_days_back: Days of stock history (default: 365)

    Returns:
        (news_df, stock_df)
    """
    from src.data_ingestion import StockIngester, align_news_to_market
    from src.ticker_utils import get_market_info, get_company_name

    ticker      = ticker.upper()
    market_info = get_market_info(ticker)
    company     = company_name or get_company_name(ticker)

    logger.info(
        f"Market: {market_info['market']} | "
        f"Company: {company} | "
        f"Currency: {market_info['currency']}"
    )

    logger.info(f"Fetching stock data ({stock_days_back} days)...")
    stock_df = StockIngester().fetch(ticker=ticker, days_back=stock_days_back)

    if stock_df.empty:
        logger.error(
            f"No stock data for {ticker}. "
            f"Check the ticker format — Indian NSE stocks need '.NS' suffix "
            f"(e.g. RELIANCE.NS), BSE stocks need '.BO'."
        )
        return pd.DataFrame(), stock_df

    logger.info(f"Fetching RSS news for {ticker} / {company}...")
    ingester = RSSIngester()
    news_df  = ingester.fetch(
        ticker      = ticker,
        company_name= company,
        market_info = market_info,
    )

    if news_df.empty:
        logger.warning("No news found. Returning stock data only.")
        return news_df, stock_df

    # Use market-aware close hour for alignment
    news_df = align_news_to_market(
        news_df, stock_df,
        close_hour = market_info.get("close_hour", 16),
        timezone   = market_info.get("timezone", "America/New_York"),
    )

    logger.info(
        f"Ready: {len(news_df)} articles aligned to "
        f"{news_df['market_date'].nunique()} trading days | "
        f"{len(stock_df)} stock rows."
    )

    return news_df, stock_df
