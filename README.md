# 📡 Sentiment Signal — Financial News → Trading Signal System

A production-grade, fully automated fintech ML pipeline that reads financial news via RSS, scores it with FinBERT, and generates next-day BUY/SELL/HOLD trading signals across 10 global markets. The model retrains itself every day including weekends — weekend news directly affects Monday's opening price.

**🔗 Live app:** https://financial-sentiment-based-trading-aydxhwawjtqzf495yssd55.streamlit.app

---

## How it works

```
RSS news (no API key)
    ↓
FinBERT sentiment scoring (auto-fallback to lighter model on low RAM)
    ↓
22 features: sentiment mean, disagreement (std), volume spikes, price momentum
    ↓
Random Forest — walk-forward validated, no data leakage
    ↓
BUY / SELL / HOLD + confidence score + sentiment disagreement gauge
    ↓
Backtested with 10bps transaction costs
    ↓
Prediction logged to Supabase → actual filled next trading day → retrain
```

---

## Architecture — 9 stages

| Stage | File | Description |
|-------|------|-------------|
| 1 | `src/rss_ingester.py` | RSS ingestion — Google News, Reuters, Seeking Alpha. No API key. |
| 2 | `src/sentiment_model.py` | FinBERT scoring. Auto-fallback to lighter model if RAM exceeded. |
| 3 | `src/feature_engineering.py` | 22 features including sentiment disagreement (std_score) |
| 4 | `src/prediction.py` | Random Forest, walk-forward CV, 5 folds |
| 5 | `src/backtesting.py` | Strategy vs buy-and-hold, Sharpe, drawdown, win rate |
| 6 | `api/app.py` | FastAPI REST endpoints |
| 7 | `dashboard/streamlit_app.py` | Streamlit dashboard — dark theme, multi-market, 5 analysis tabs |
| 8 | `src/feedback_logger.py` | Supabase feedback loop — smart dedup, orphan cleanup |
| 9 | `scripts/daily_retrain.py` | GitHub Actions — daily retrain across 100 tickers / 10 markets |

---

## Markets covered

| Market | Exchange | Example tickers |
|--------|----------|----------------|
| 🇺🇸 USA | NYSE / NASDAQ | NVDA, AAPL, MSFT, GOOGL, AMZN, META, TSLA, BRK-B, AVGO, LLY |
| 🇬🇧 UK | LSE (.L) | SHEL, HSBA, ULVR, BP, AZN, GSK, DGE, RIO, BATS, BARC |
| 🇩🇪 Germany | XETRA (.DE) | SAP, SIE, VOW3, MBG, ALV, BAS, BMW, DTE, IFX, ADS |
| 🇳🇱 Netherlands | AEX (.AS) | ASML, PRX, INGA, HEIA, AGN, ASRNL, ARGX, RAND + NXPI, AIR.PA |
| 🇫🇷 France | Euronext Paris (.PA) | MC, TTE, OR, SU, AIR, SAN, BNP, CS, SAF, RMS |
| 🇨🇭 Switzerland | SIX (.SW) | NESN, ROG, NOVN, UBSG, ZURN, ABBN, CFR, SREN, LONN, HOLN |
| 🇮🇹 Italy | Borsa Italiana (.MI) | ENEL, ENI, RACE, ISP, UCG, STLAM, PIRC, TIT, LDO, MONC |
| 🇮🇳 India | NSE (.NS) | RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK, HINDUNILVR, BHARTIARTL, LT, SBIN, ADANIENT |
| 🇨🇳 China/HK | HKEX (.HK) | 0700, 9988, 1398, 0939, 1211, 2318, 0857, 3690 + JD, BIDU |
| 🇯🇵 Japan | TSE (.T) | 7203, 6758, 8306, 9984, 6861, 7974, 8035, 6501, 7267, 8316 |

---

## Automated daily pipeline

Runs every day at **13:00 UTC including weekends** via GitHub Actions. Weekend news (earnings, analyst reports, geopolitical events) affects Monday's opening price — skipping weekends would mean Monday's prediction uses stale Friday sentiment.

**On weekdays:** fresh price data + fresh news → full retrain
**On weekends:** Friday EOD prices + weekend headlines → sentiment-updated signal

**Batch rotation** (stays within 2,000 min/month free tier):
- Mon / Wed / Fri / Sat → Batch A: USA, India, UK, Netherlands, France (~45 tickers, ~67 min)
- Tue / Thu / Sun → Batch B: Germany, France, Switzerland, Italy, China/HK, Japan (~55 tickers, ~82 min)

Total: ~1,600 min/month. Every ticker retrains 3–4× per week.

---

## Feedback loop

Every prediction is logged to Supabase. The next day's actual price direction fills in automatically. Three-layer cleanup keeps the table small:

| Layer | What | When |
|-------|------|------|
| Normal | Delete `retrained=true` rows older than 7 days | After every retrain run + pg_cron 02:00 UTC |
| Orphan | Delete `retrained=false` rows older than 21 days | Same — catches unresolved/dashboard-only rows |
| Guard | Force-delete oldest 500 rows when table hits 10,000 rows | On every insert |

**Smart dedup:** one row per ticker per trading day. Updated only if new confidence is ≥0.02 higher or sentiment shifts ≥0.05.

---

## Dashboard

**Home page**
- 🌍 Six global index cards with 30-day sparklines and live open/closed/pre-market status
- 🧠 Market-wide FinBERT sentiment per region
- 🔥 Trending financial news (live RSS)
- 🔍 Global news search
- 🕐 User's local time and timezone (JavaScript, updates every second)
- 📅 Weekend mode banner with explanation

**Per-ticker tabs**
- 📈 Price vs Sentiment — bars colour-coded by direction AND disagreement (amber = divided headlines)
- 💼 Backtest — equity curve vs buy-and-hold
- 🧠 Model — feature importance, walk-forward results, active model indicator
- 📰 Headlines — searchable, grouped by Bullish / Neutral / Bearish
- 📊 Model Performance — live accuracy chart from Supabase

**Sentiment disagreement panel**
Shows headline split %, disagreement gauge (Low/Moderate/High based on std_score), and plain-English interpretation. High disagreement = market hasn't priced in uncertainty = larger move possible.

---

## Backtesting results (AAPL sample)

| Metric | Strategy | Buy & Hold |
|--------|----------|-----------|
| Total return | +54.3% | -3.0% |
| Annual return | +356.2% | — |
| Sharpe ratio | 6.59 | — |
| Max drawdown | -4.6% | — |
| Win rate | 64% | — |

*73-day test window · 10bps transaction costs · walk-forward validated*

---

## Model

- **Algorithm:** Random Forest, walk-forward cross-validation (5 folds)
- **Accuracy:** ~54% directional (above 50% random baseline)
- **Top features:** `volatility_3d`, `daily_return`, `close`, `sentiment_std` (disagreement)
- **Key insight:** Sentiment disagreement between headlines predicts moves better than mean sentiment alone

---

## Tech stack

| Component | Technology |
|-----------|-----------|
| Sentiment (primary) | ProsusAI/finbert (~440MB, ~1.5GB RAM) |
| Sentiment (fallback) | yiyanghkust/finbert-tone (~110MB, auto-switches on OOM) |
| ML | scikit-learn Random Forest |
| News | RSS via feedparser — no API key |
| Prices | yfinance — 10 markets |
| Database | Supabase Postgres (free tier) |
| API | FastAPI |
| Dashboard | Streamlit Community Cloud |
| Automation | GitHub Actions (daily cron, free tier) |
| DB cleanup | Supabase pg_cron (02:00 UTC daily) |

---

## Free tier budget

| Provider | Limit | Usage | Headroom |
|----------|-------|-------|---------|
| GitHub Actions | 2,000 min/month | ~1,600 min | 400 min |
| Supabase storage | 500MB | ~0.7MB | 499.3MB |
| Streamlit Cloud | Free forever | Live | — |

---

## Local setup

```bash
git clone https://github.com/shardulparulekar/financial-sentiment-based-trading
cd financial-sentiment-based-trading
pip install -r requirements.txt
streamlit run dashboard/streamlit_app.py
```

No API keys needed. Uses RSS feeds and yfinance — both free and keyless.

**Enable Supabase feedback loop** — add to Streamlit Cloud → App Settings → Secrets:
```toml
[supabase]
url = "https://xxxx.supabase.co"
key = "sb_publishable_xxxx"
```

**Enable GitHub Actions retraining** — add to GitHub → Settings → Secrets → Actions:
```
SUPABASE_URL = https://xxxx.supabase.co
SUPABASE_KEY = sb_publishable_xxxx
```

**Supabase one-time setup** — run `scripts/supabase_setup.sql` in Supabase SQL Editor.

---

## Project structure

```
financial-sentiment-based-trading/
├── .github/workflows/daily_retrain.yml  ← daily cron (every day, 13:00 UTC)
├── .streamlit/config.toml               ← dark theme
├── .python-version                      ← pins Python 3.11
├── .gitignore
├── requirements.txt
│
├── src/
│   ├── rss_ingester.py       stage 1: news ingestion
│   ├── sentiment_model.py    stage 2: FinBERT + auto-fallback
│   ├── feature_engineering.py stage 3: 22 features incl. disagreement
│   ├── prediction.py         stage 4: Random Forest, walk-forward CV
│   ├── backtesting.py        stage 5: strategy simulation
│   ├── feedback_logger.py    stage 8: Supabase feedback + 2-pass cleanup
│   ├── market_sentiment.py   market-wide sentiment aggregation
│   ├── load_data.py          smart data loader
│   └── ticker_utils.py       ticker/exchange utilities
│
├── api/app.py                stage 6: FastAPI REST
├── dashboard/streamlit_app.py stage 7: Streamlit UI
│
├── scripts/
│   ├── daily_retrain.py      stage 9: automated retraining (BATCH_A/BATCH_B)
│   └── supabase_setup.sql    one-time DB + RLS + pg_cron setup
│
└── data/sample/              pre-fetched sample data
```

---

## Key design decisions

| Decision | Reason |
|----------|--------|
| RSS over NewsAPI | No key, 60-180 days history vs 14 days |
| Weekend retraining | Weekend news moves Monday prices — skipping it loses signal |
| Walk-forward validation | Respects time order, no data leakage |
| Sentiment disagreement | std_score more predictive than mean score alone |
| Smart dedup (1 row/day) | Intraday rescoring updates only on meaningful change |
| FinBERT auto-fallback | Graceful degradation on memory-constrained servers |
| Two-pass cleanup | Normal pass removes incorporated rows; orphan pass removes unresolved rows that would otherwise accumulate indefinitely |
| 10bps transaction costs | Realistic, not cherry-picked backtest results |
