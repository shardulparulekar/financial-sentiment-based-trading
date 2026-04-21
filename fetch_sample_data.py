"""
fetch_sample_data.py
--------------------
Run this ONCE locally to generate the sample data that ships with the repo.
Uses RSS feeds — no API key required.

The output goes to data/sample/ which IS committed to git.
This means anyone who clones the project can run the full pipeline
immediately with no setup at all.

Usage:
    python fetch_sample_data.py
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

ROOT       = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "data" / "sample"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

TICKERS = [
    {"ticker": "AAPL",  "company": "Apple"},
    {"ticker": "MSFT",  "company": "Microsoft"},
    {"ticker": "GOOGL", "company": "Google"},
]


def main():
    from src.rss_ingester import RSSIngester
    from src.data_ingestion import StockIngester, align_news_to_market

    stock_ingester = StockIngester()
    rss_ingester   = RSSIngester()

    meta = {
        "fetched_at": datetime.utcnow().isoformat() + "Z",
        "source":     "RSS feeds (no API key required)",
        "tickers":    [t["ticker"] for t in TICKERS],
        "note":       "Pre-fetched sample data — safe to commit.",
    }

    for item in TICKERS:
        ticker  = item["ticker"]
        company = item["company"]
        print(f"\n── {ticker} ──────────────────────────────────")

        print(f"  Fetching stock prices (365 days)...")
        stock_df   = stock_ingester.fetch(ticker=ticker, days_back=365, save=False)
        stock_path = SAMPLE_DIR / f"stock_{ticker.lower()}.csv"
        stock_df.to_csv(stock_path, index=False)
        print(f"  Saved {len(stock_df)} rows → {stock_path.relative_to(ROOT)}")

        print(f"  Fetching RSS news...")
        news_df = rss_ingester.fetch(ticker=ticker, company_name=company, save=False)

        if news_df.empty:
            print(f"  WARNING: No news returned for {ticker}. Skipping.")
            continue

        aligned_df = align_news_to_market(news_df, stock_df)
        news_path  = SAMPLE_DIR / f"news_{ticker.lower()}.csv"
        aligned_df.to_csv(news_path, index=False)
        print(f"  Saved {len(aligned_df)} articles → {news_path.relative_to(ROOT)}")

    meta_path = SAMPLE_DIR / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Sample data written to data/sample/")
    print(f"  Run: git add data/sample/ && git commit -m 'add sample data'")


if __name__ == "__main__":
    main()
