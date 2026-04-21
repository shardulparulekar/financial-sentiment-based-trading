# Financial News Sentiment → Trading Signal System

An end-to-end fintech pipeline that scores financial news sentiment with FinBERT,
engineers predictive features, and generates backtested trading signals.

## Architecture

```
RSS feeds + yfinance → FinBERT sentiment → Feature engineering →
Prediction model → Backtesting engine → FastAPI → Streamlit dashboard
```

## Quickstart — no API key, no setup

```bash
git clone https://github.com/you/fintech-sentiment
cd fintech-sentiment
pip install -r requirements.txt
python fetch_sample_data.py   # fetch sample data via RSS (free, no key)
```

Then open Jupyter and run:
```python
import sys; sys.path.insert(0, '.')
from src.load_data import load
news_df, stock_df = load("AAPL")
```

## Modules

| # | Module | File | Status |
|---|--------|------|--------|
| 1 | Data ingestion | `src/data_ingestion.py` | ✅ Complete |
| 1b | RSS ingester | `src/rss_ingester.py` | ✅ Complete |
| 1c | Smart data loader | `src/load_data.py` | ✅ Complete |
| 2 | FinBERT sentiment | `src/sentiment_model.py` | ✅ Complete |
| 3 | Feature engineering | `src/feature_engineering.py` | ✅ Complete |
| 4 | Prediction model | `src/prediction.py` | 🔜 Next |
| 5 | Backtesting engine | `src/backtesting.py` | 🔜 |
| 6 | API | `api/app.py` | 🔜 |
| 7 | Dashboard | `dashboard/streamlit_app.py` | 🔜 |

## Optional: NewsAPI (more sources, same 14-day limit)

If you want to supplement RSS with NewsAPI:
```bash
# Get a free key at https://newsapi.org
export NEWS_API_KEY=your_key_here
```
The pipeline auto-detects the key and uses it alongside RSS.

## Troubleshooting

**`ValueError: Keras 3 not supported...`**
```bash
pip install tf-keras
# restart Jupyter kernel
```

**`ModuleNotFoundError: feedparser`**
```bash
pip install feedparser
```

## Key design decisions

- **RSS over NewsAPI** — no key needed, 60-180 days of history vs 14 days
- **No lookahead bias** — news after 4pm ET assigned to next trading day
- **FinBERT over VADER** — domain-trained, understands financial language
- **Walk-forward backtesting** — train/test split respects time order
- **Transaction costs included** — realistic backtest results

## Project structure

```
fintech-sentiment/
├── data/sample/              ← pre-fetched data, committed to git
├── src/
│   ├── rss_ingester.py       ← primary news source (no key needed)
│   ├── data_ingestion.py     ← optional NewsAPI + yfinance
│   ├── load_data.py          ← single entry point for all modules
│   ├── sentiment_model.py    ← FinBERT scoring + daily aggregation
│   ├── feature_engineering.py
│   ├── prediction.py
│   └── backtesting.py
│   └── ticker_utils.py
├── api/app.py
├── dashboard/streamlit_app.py
├── fetch_sample_data.py      ← run once to populate data/sample/
└── requirements.txt
```
