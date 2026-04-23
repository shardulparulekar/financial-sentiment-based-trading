"""
market_sentiment.py
-------------------
Computes an aggregate sentiment score for each market (USA, India, Europe, etc.)
by sampling a small basket of representative stocks and averaging their FinBERT scores.

This gives a "market mood" reading — shown on the dashboard home page
below the index price charts.

The basket is intentionally small (3-4 stocks per market) to keep
RSS fetching fast. We care about direction, not precision.

Sentiment scale:
    > +0.05  → Bullish  🟢
    < -0.05  → Bearish  🔴
    otherwise → Neutral ⚪

Usage:
    from src.market_sentiment import MarketSentimentScorer

    scorer  = MarketSentimentScorer()
    results = scorer.score_all_markets()
    # returns dict: {"🇺🇸 USA": {"score": 0.12, "label": "Bullish", ...}, ...}
"""

import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

# Representative basket per market — small enough to fetch quickly,
# diverse enough to capture market mood.
MARKET_BASKETS = {
    "🇺🇸 USA": [
        {"ticker": "AAPL",  "company": "Apple"},
        {"ticker": "MSFT",  "company": "Microsoft"},
        {"ticker": "JPM",   "company": "JPMorgan"},
    ],
    "🇮🇳 India": [
        {"ticker": "RELIANCE.NS", "company": "Reliance Industries"},
        {"ticker": "TCS.NS",      "company": "Tata Consultancy Services"},
        {"ticker": "HDFCBANK.NS", "company": "HDFC Bank"},
    ],
    "🇪🇺 Europe": [
        {"ticker": "ASML.AS", "company": "ASML"},
        {"ticker": "SAP.DE",  "company": "SAP"},
        {"ticker": "BP.L",    "company": "BP"},
    ],
    "🇨🇳 China / HK": [
        {"ticker": "BABA",   "company": "Alibaba"},
        {"ticker": "0700.HK","company": "Tencent"},
    ],
    "🇯🇵 Japan": [
        {"ticker": "7203.T", "company": "Toyota"},
        {"ticker": "6758.T", "company": "Sony"},
    ],
}


class MarketSentimentScorer:
    """
    Scores overall market sentiment by averaging FinBERT scores
    across a representative basket of stocks per market.
    """

    def __init__(self, max_articles_per_stock: int = 20):
        self.max_articles = max_articles_per_stock
        self._pipe        = None

    def _get_pipe(self):
        """Load FinBERT pipeline lazily (reuses cached instance if available)."""
        if self._pipe is None:
            from src.sentiment_model import SentimentPipeline
            self._pipe = SentimentPipeline()
        return self._pipe

    def score_market(self, market_name: str) -> dict:
        """
        Score sentiment for a single market.

        Args:
            market_name: Key from MARKET_BASKETS e.g. "🇺🇸 USA"

        Returns:
            dict with score, label, n_articles, stocks, error (if any)
        """
        basket = MARKET_BASKETS.get(market_name, [])
        if not basket:
            return {"score": 0.0, "label": "Neutral", "n_articles": 0, "stocks": []}

        try:
            import feedparser
            import requests
            from src.sentiment_model import SentimentPipeline

            pipe     = self._get_pipe()
            all_headlines = []
            stock_scores  = []

            for item in basket:
                try:
                    headlines = self._fetch_headlines(
                        item["ticker"], item["company"], feedparser, requests
                    )
                    if not headlines:
                        continue

                    import pandas as pd
                    df = pd.DataFrame({"title": headlines[:self.max_articles]})
                    scored = pipe.score(df)

                    if scored.empty:
                        continue

                    stock_score = float(scored["sentiment_score"].mean())
                    stock_scores.append({
                        "ticker":     item["ticker"],
                        "company":    item["company"],
                        "score":      round(stock_score, 4),
                        "label":      self._to_label(stock_score),
                        "n_articles": len(scored),
                    })
                    all_headlines.extend(headlines[:self.max_articles])
                    time.sleep(0.3)   # be polite to RSS servers

                except Exception as e:
                    logger.warning(f"  {item['ticker']} failed: {e}")
                    continue

            if not stock_scores:
                return {
                    "score": 0.0, "label": "Neutral",
                    "n_articles": 0, "stocks": [],
                    "error": "No articles fetched",
                }

            # Market score = average of stock scores (equal weighted)
            market_score = sum(s["score"] for s in stock_scores) / len(stock_scores)

            return {
                "score":      round(market_score, 4),
                "label":      self._to_label(market_score),
                "n_articles": sum(s["n_articles"] for s in stock_scores),
                "stocks":     stock_scores,
                "error":      None,
            }

        except Exception as e:
            logger.error(f"Market sentiment failed for {market_name}: {e}")
            return {"score": 0.0, "label": "Neutral", "n_articles": 0, "stocks": [], "error": str(e)}

    def score_all_markets(self) -> dict:
        """
        Score all markets in MARKET_BASKETS.

        Returns:
            dict keyed by market name, each value is a score dict.
        """
        logger.info("Scoring sentiment for all markets...")
        results = {}
        for market_name in MARKET_BASKETS:
            logger.info(f"  Scoring {market_name}...")
            results[market_name] = self.score_market(market_name)
        logger.info("Market sentiment scoring complete.")
        return results

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _fetch_headlines(
        ticker: str,
        company: str,
        feedparser,
        requests,
    ) -> list[str]:
        """Fetch recent headlines for a stock via Google News RSS."""
        headers = {"User-Agent": "Mozilla/5.0"}
        query   = company.replace(" ", "+")
        url     = (
            f"https://news.google.com/rss/search"
            f"?q={query}+stock&hl=en-US&gl=US&ceid=US:en"
        )
        try:
            r    = requests.get(url, headers=headers, timeout=8)
            feed = feedparser.parse(r.content)
            return [
                e.get("title", "").strip()
                for e in feed.entries
                if e.get("title", "").strip()
            ]
        except Exception:
            return []

    @staticmethod
    def _to_label(score: float) -> str:
        if score > 0.05:
            return "Bullish"
        if score < -0.05:
            return "Bearish"
        return "Neutral"
