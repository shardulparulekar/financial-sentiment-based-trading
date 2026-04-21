"""
dashboard/streamlit_app.py
--------------------------
Module 7: Sentiment Signal Dashboard

Run:
    streamlit run dashboard/streamlit_app.py
"""

import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.WARNING)

st.set_page_config(
    page_title="Sentiment Signal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');
    html, body, [class*="css"]  { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3                  { font-family: 'DM Mono', monospace; letter-spacing:-0.02em; }

    /* Top nav bar */
    .nav-bar { display:flex; align-items:center; justify-content:space-between;
               padding:0.6rem 0; margin-bottom:1.5rem; border-bottom:1px solid #1e2130; }
    .nav-logo { font-family:'DM Mono',monospace; font-size:1.1rem; font-weight:500; color:#2563eb; }
    .nav-sub  { font-size:0.78rem; color:#6b7280; margin-left:0.5rem; }

    /* Ticker tab pills */
    .tab-pill { display:inline-flex; align-items:center; gap:0.35rem;
                background:#1e2130; border:1px solid #374151; border-radius:20px;
                padding:0.3rem 0.75rem; font-size:0.82rem; font-family:'DM Mono',monospace;
                cursor:pointer; margin-right:0.4rem; margin-bottom:0.4rem; }
    .tab-pill.active { background:#2563eb; border-color:#2563eb; color:#fff; }
    .tab-pill .close { color:#9ca3af; font-size:0.75rem; margin-left:0.2rem; }

    /* Index cards */
    .index-card { background:#0f1117; border:1px solid #1e2130; border-radius:10px;
                  padding:0.9rem 1rem 0.5rem; }
    .index-name { font-size:0.78rem; color:#6b7280; margin-bottom:0.1rem; }
    .index-val  { font-family:'DM Mono',monospace; font-size:1.25rem; font-weight:500; }
    .index-chg-up   { color:#16a34a; font-size:0.82rem; }
    .index-chg-down { color:#dc2626; font-size:0.82rem; }

    /* Trending cards */
    .t-card { background:#0f1117; border:1px solid #1e2130; border-radius:10px;
              padding:0.9rem 1rem; height:100%; border-top-width:3px; }
    .t-title { font-size:0.88rem; font-weight:500; line-height:1.4; margin-bottom:0.4rem; }
    .t-meta  { font-size:0.75rem; color:#6b7280; }
    .badge   { display:inline-block; padding:0.1rem 0.4rem; border-radius:4px;
               font-size:0.72rem; color:#fff; margin-left:0.3rem; }

    /* Signal blocks */
    .signal-buy  { background:linear-gradient(135deg,#052e16,#14532d); border:1px solid #16a34a; border-radius:12px; padding:1.5rem; text-align:center; }
    .signal-sell { background:linear-gradient(135deg,#2d0a0a,#7f1d1d); border:1px solid #dc2626; border-radius:12px; padding:1.5rem; text-align:center; }
    .signal-hold { background:linear-gradient(135deg,#111827,#1f2937); border:1px solid #4b5563; border-radius:12px; padding:1.5rem; text-align:center; }

    /* Headlines */
    .hl-row { border-left:3px solid; padding:0.4rem 0.75rem; margin-bottom:0.4rem;
              border-radius:0 4px 4px 0; background:rgba(255,255,255,0.02); font-size:0.85rem; }
    .warn-box { background:#2d1b00; border:1px solid #d97706; border-radius:8px;
                padding:1rem 1.25rem; margin-bottom:1rem; }

    /* Search box */
    .search-wrap { position:relative; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

MARKET_CONFIG = {
    "🇺🇸 USA": {
        "suffix": "", "currency": "USD", "symbol": "$", "region": "US",
        "tickers": ["AAPL","MSFT","GOOGL","TSLA","NVDA","AMZN","META","NFLX","JPM"],
        "names":   ["Apple","Microsoft","Google","Tesla","NVIDIA","Amazon","Meta","Netflix","JPMorgan"],
    },
    "🇮🇳 India (NSE)": {
        "suffix": ".NS", "currency": "INR", "symbol": "₹", "region": "IN",
        "tickers": ["RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK","WIPRO","TATAMOTORS","ZOMATO","SBIN"],
        "names":   ["Reliance","TCS","Infosys","HDFC Bank","ICICI Bank","Wipro","Tata Motors","Zomato","SBI"],
    },
    "🇨🇳 China / HK": {
        "suffix": ".HK", "currency": "HKD", "symbol": "HK$", "region": "HK",
        "tickers": ["0700","9988","3690","1810"], "names": ["Tencent","Alibaba HK","Meituan","Xiaomi"],
        "us_adrs": ["BABA","PDD","JD","BIDU","NIO"],
        "adr_names": ["Alibaba","Pinduoduo","JD.com","Baidu","NIO"],
    },
    "🇯🇵 Japan": {
        "suffix": ".T", "currency": "JPY", "symbol": "¥", "region": "JP",
        "tickers": ["7203","6758","9984","6861"], "names": ["Toyota","Sony","SoftBank","Keyence"],
    },
}

EU_EXCHANGES = {
    "🇳🇱 Amsterdam": {"suffix":".AS","currency":"EUR","symbol":"€","tickers":["ASML","PHIA","INGA","HEIA","ADYEN"],"names":["ASML","Philips","ING","Heineken","Adyen"]},
    "🇩🇪 Frankfurt": {"suffix":".DE","currency":"EUR","symbol":"€","tickers":["SAP","BMW","SIE","VOW3","ALV","MBG"],"names":["SAP","BMW","Siemens","Volkswagen","Allianz","Mercedes"]},
    "🇫🇷 Paris":     {"suffix":".PA","currency":"EUR","symbol":"€","tickers":["MC","AIR","OR","TTE","SAN"],"names":["LVMH","Airbus","L'Oréal","TotalEnergies","Sanofi"]},
    "🇬🇧 London":    {"suffix":".L","currency":"GBP","symbol":"£","tickers":["HSBA","SHEL","BP","AZN","GSK"],"names":["HSBC","Shell","BP","AstraZeneca","GSK"]},
    "🇨🇭 Zurich":    {"suffix":".SW","currency":"CHF","symbol":"CHF ","tickers":["NESN","NOVN","ROG","UBSG"],"names":["Nestlé","Novartis","Roche","UBS"]},
    "🇮🇹 Milan":     {"suffix":".MI","currency":"EUR","symbol":"€","tickers":["RACE","UCG","ENI","ENEL"],"names":["Ferrari","UniCredit","ENI","Enel"]},
}

WORLD_INDICES = [
    {"name": "S&P 500",    "ticker": "^GSPC",  "symbol": "$",   "region": "🇺🇸"},
    {"name": "FTSE 100",   "ticker": "^FTSE",  "symbol": "£",   "region": "🇬🇧"},
    {"name": "Nikkei 225", "ticker": "^N225",  "symbol": "¥",   "region": "🇯🇵"},
    {"name": "Sensex",     "ticker": "^BSESN", "symbol": "₹",   "region": "🇮🇳"},
    {"name": "DAX",        "ticker": "^GDAXI", "symbol": "€",   "region": "🇩🇪"},
]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def resolve_ticker(display: str, market: str, exchange: str | None = None) -> str:
    if market == "🇪🇺 Europe" and exchange:
        return display + EU_EXCHANGES[exchange]["suffix"]
    if market == "🇨🇳 China / HK":
        if display in MARKET_CONFIG[market].get("us_adrs", []):
            return display
        return display + ".HK"
    sfx = MARKET_CONFIG[market].get("suffix", "")
    return display + sfx if sfx and not display.endswith(sfx) else display


def get_currency(market: str, exchange: str | None = None) -> tuple[str, str]:
    if market == "🇪🇺 Europe" and exchange:
        ex = EU_EXCHANGES[exchange]
        return ex["currency"], ex["symbol"]
    cfg = MARKET_CONFIG[market]
    return cfg["currency"], cfg["symbol"]


def get_company_name(display_ticker: str, market: str, exchange: str | None = None) -> str:
    if market == "🇪🇺 Europe" and exchange:
        ex = EU_EXCHANGES[exchange]
        for t, n in zip(ex["tickers"], ex["names"]):
            if t == display_ticker:
                return n
        return display_ticker
    cfg = MARKET_CONFIG[market]
    tickers = cfg["tickers"] + cfg.get("us_adrs", [])
    names   = cfg["names"] + cfg.get("adr_names", [])
    for t, n in zip(tickers, names):
        if t == display_ticker:
            return n
    return display_ticker


def validate_custom(raw: str, market: str, exchange: str | None = None) -> tuple[bool, str, str]:
    raw = raw.strip().upper()
    entered_suffix = ("." + raw.rsplit(".", 1)[-1]) if "." in raw else ""
    if market == "🇪🇺 Europe":
        if not exchange:
            return False, raw, "Please select a European exchange first."
        expected = EU_EXCHANGES[exchange]["suffix"]
        if entered_suffix == "":
            return True, raw + expected, ""
        if entered_suffix.upper() != expected.upper():
            detected = next((ex for ex, ec in EU_EXCHANGES.items() if ec["suffix"] == entered_suffix), None)
            det = f" That suffix belongs to **{detected}**." if detected else ""
            return False, raw, f"`{raw}` suffix `{entered_suffix}` doesn't match **{exchange}** (expected `{expected}`).{det}"
        return True, raw, ""
    expected = MARKET_CONFIG[market].get("suffix", "")
    if expected and entered_suffix == "":
        return True, raw + expected, ""
    if expected and entered_suffix.upper() != expected.upper():
        detected_mkt = next((m for m, c in MARKET_CONFIG.items() if c.get("suffix","") == entered_suffix), None)
        detected_eu  = next((ex for ex, ec in EU_EXCHANGES.items() if ec["suffix"] == entered_suffix), None)
        det = (f" Belongs to **{detected_mkt}**." if detected_mkt
               else f" Belongs to **{detected_eu}** (Europe)." if detected_eu else "")
        return False, raw, f"`{raw}` suffix `{entered_suffix}` doesn't match **{market}**.{det}"
    return True, raw, ""


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

if "open_tickers" not in st.session_state:
    st.session_state.open_tickers = []   # list of dicts: {full, display, company, market, exchange}
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "home"


def add_ticker_tab(full_ticker, display_ticker, company, market, exchange):
    exists = any(t["full"] == full_ticker for t in st.session_state.open_tickers)
    if not exists:
        st.session_state.open_tickers.append({
            "full": full_ticker, "display": display_ticker,
            "company": company, "market": market, "exchange": exchange,
        })
    st.session_state.active_tab = full_ticker
    st.rerun()


def close_ticker_tab(full_ticker):
    st.session_state.open_tickers = [
        t for t in st.session_state.open_tickers if t["full"] != full_ticker
    ]
    if st.session_state.active_tab == full_ticker:
        st.session_state.active_tab = "home"
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# CACHED DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def load_index_data(ticker: str) -> pd.DataFrame:
    import yfinance as yf
    end   = datetime.today()
    start = end - timedelta(days=35)
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Close"]].dropna()


@st.cache_data(ttl=3600, show_spinner=False)
def load_trending_news(n: int = 3) -> list[dict]:
    """Fetch recent global financial headlines via RSS."""
    try:
        import feedparser, requests
        HEADERS = {"User-Agent": "Mozilla/5.0"}
        feeds = [
            ("https://news.google.com/rss/search?q=stock+market+finance&hl=en-US&gl=US&ceid=US:en", "Google News"),
            ("https://feeds.reuters.com/reuters/businessNews", "Reuters"),
        ]
        articles = []
        for url, source in feeds:
            try:
                r    = requests.get(url, headers=HEADERS, timeout=8)
                feed = feedparser.parse(r.content)
                for entry in feed.entries[:6]:
                    title = entry.get("title","").strip()
                    if title:
                        articles.append({
                            "title":  title,
                            "source": source,
                            "url":    entry.get("link","#"),
                            "date":   entry.get("published",""),
                        })
            except Exception:
                continue
        return articles[:n]
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def search_global_news(query: str, max_results: int = 10) -> list[dict]:
    """Search global RSS news for any query string."""
    try:
        import feedparser, requests
        HEADERS = {"User-Agent": "Mozilla/5.0"}
        q   = query.replace(" ", "+")
        url = f"https://news.google.com/rss/search?q={q}&hl=en&gl=US&ceid=US:en"
        r   = requests.get(url, headers=HEADERS, timeout=8)
        feed = feedparser.parse(r.content)
        results = []
        for entry in feed.entries[:max_results]:
            title = entry.get("title","").strip()
            if title:
                results.append({
                    "title":  title,
                    "source": entry.get("source",{}).get("title","Google News"),
                    "url":    entry.get("link","#"),
                    "date":   entry.get("published",""),
                })
        return results
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def run_pipeline(ticker: str, days_back: int) -> dict:
    from src.rss_ingester import fetch_rss_and_stock
    from src.sentiment_model import SentimentPipeline
    from src.feature_engineering import FeatureEngineer
    from src.prediction import PredictionModel
    from src.backtesting import Backtester

    news_df, stock_df = fetch_rss_and_stock(ticker, stock_days_back=days_back)
    if stock_df.empty:
        raise ValueError(f"No stock data for `{ticker}`. Check the ticker symbol.")

    pipe   = SentimentPipeline()
    scored = pd.DataFrame()
    daily  = pd.DataFrame()

    if not news_df.empty and "market_date" in news_df.columns:
        scored = pipe.score(news_df)
        if not scored.empty:
            daily = pipe.aggregate_daily(scored)

    fe       = FeatureEngineer()
    features = fe.build(daily, stock_df)
    X, y     = fe.split_features_target(features, target="next_day_direction")

    model   = PredictionModel()
    results = model.train_evaluate(X, y)
    bt      = Backtester()
    report  = bt.run(features, model)

    X_latest   = X.iloc[[-1]]
    signal     = int(model.predict(X_latest)[0])
    confidence = round(float(max(model.predict_proba(X_latest)[0])), 3)

    return {
        "news_df": news_df, "scored": scored, "stock_df": stock_df,
        "daily": daily, "features": features, "X": X,
        "report": report, "signal": signal, "confidence": confidence,
        "model": model, "results": results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# NAV BAR + TAB PILLS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="nav-bar">
    <span>
        <span class="nav-logo">📡 Sentiment Signal</span>
        <span class="nav-sub">FinBERT-powered global trading signals</span>
    </span>
</div>
""", unsafe_allow_html=True)

# Tab pills row
pill_cols = st.columns([1] + [1] * len(st.session_state.open_tickers) + [4])

with pill_cols[0]:
    if st.button("🏠 Home", type="secondary" if st.session_state.active_tab != "home" else "primary",
                 use_container_width=True):
        st.session_state.active_tab = "home"
        st.rerun()

for i, tab in enumerate(st.session_state.open_tickers):
    with pill_cols[i + 1]:
        col_btn, col_x = st.columns([4, 1])
        with col_btn:
            is_active = st.session_state.active_tab == tab["full"]
            if st.button(
                tab["display"],
                key=f"tab_btn_{tab['full']}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                st.session_state.active_tab = tab["full"]
                st.rerun()
        with col_x:
            if st.button("✕", key=f"close_{tab['full']}", use_container_width=True):
                close_ticker_tab(tab["full"])

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.active_tab == "home":

    # ── World indices ──────────────────────────────────────────────────────────
    st.markdown("### 🌍 Global markets")

    idx_cols = st.columns(len(WORLD_INDICES))
    for col, idx in zip(idx_cols, WORLD_INDICES):
        with col:
            try:
                df_idx = load_index_data(idx["ticker"])
                if df_idx.empty:
                    raise ValueError("empty")

                current = float(df_idx["Close"].iloc[-1])
                prev    = float(df_idx["Close"].iloc[-2])
                chg_pct = (current - prev) / prev * 100
                up      = chg_pct >= 0
                line_color = "#22c55e" if up else "#ef4444"
                fill_color = "rgba(34,197,94,0.15)" if up else "rgba(239,68,68,0.15)"
                chg_sym = "▲" if up else "▼"
                chg_color = "#22c55e" if up else "#ef4444"

                # Full 30-day OHLC chart inside the card
                fig_spark = go.Figure()
                fig_spark.add_trace(go.Scatter(
                    x=df_idx.index, y=df_idx["Close"],
                    mode="lines",
                    line=dict(color=line_color, width=2),
                    fill="tozeroy",
                    fillcolor=fill_color,
                    hovertemplate=f"{idx['symbol']}%{{y:,.0f}}<extra></extra>",
                ))
                # Add min/max markers
                min_val = df_idx["Close"].min()
                max_val = df_idx["Close"].max()
                fig_spark.update_layout(
                    height=110,
                    margin=dict(l=4, r=4, t=4, b=4),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(
                        visible=True,
                        showticklabels=True,
                        tickformat="%d %b",
                        nticks=4,
                        tickfont=dict(size=9, color="#6b7280"),
                        showgrid=False,
                        zeroline=False,
                    ),
                    yaxis=dict(
                        visible=True,
                        showticklabels=True,
                        tickformat=",.0f",
                        nticks=3,
                        tickfont=dict(size=9, color="#6b7280"),
                        showgrid=True,
                        gridcolor="rgba(255,255,255,0.05)",
                        zeroline=False,
                    ),
                    showlegend=False,
                    hovermode="x unified",
                )

                # Value + change displayed above chart
                st.markdown(f"""
                <div style="padding:0.7rem 0.5rem 0.2rem">
                    <div style="font-size:0.75rem;color:#9ca3af">{idx['region']} {idx['name']}</div>
                    <div style="font-family:'DM Mono',monospace;font-size:1.3rem;font-weight:500;
                                color:#f9fafb;line-height:1.3">{idx['symbol']}{current:,.0f}</div>
                    <div style="font-size:0.82rem;color:{chg_color};font-weight:500">
                        {chg_sym} {abs(chg_pct):.2f}% today
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.plotly_chart(fig_spark, use_container_width=True, config={"displayModeBar": False})

            except Exception:
                st.markdown(f"""
                <div style="padding:0.7rem 0.5rem">
                    <div style="font-size:0.75rem;color:#9ca3af">{idx['region']} {idx['name']}</div>
                    <div style="color:#6b7280;font-size:0.82rem;margin-top:0.4rem">Data unavailable</div>
                </div>
                """, unsafe_allow_html=True)

    st.divider()

    # ── Trending news ──────────────────────────────────────────────────────────
    st.markdown("### 🔥 Trending financial news")

    with st.spinner("Loading trending news…"):
        trending = load_trending_news(3)

    if trending:
        t_cols = st.columns(3)
        border_colors = ["#2563eb", "#7c3aed", "#0891b2"]
        for i, (col, item) in enumerate(zip(t_cols, trending)):
            with col:
                with st.expander(item["title"], expanded=False):
                    st.markdown(f"""
                    **Source:** {item['source']}

                    🔗 [Read full article]({item['url']})
                    """)
    else:
        st.info("Could not load trending news — check your internet connection.")

    st.divider()

    # ── Global news search ─────────────────────────────────────────────────────
    st.markdown("### 🔍 Search global news")

    search_col, btn_col = st.columns([5, 1])
    with search_col:
        news_query = st.text_input(
            "search", label_visibility="collapsed",
            placeholder="Search any company, topic, or keyword — e.g. Apple, rate cut, Reliance earnings…",
            key="global_news_search",
        )
    with btn_col:
        do_search = st.button("Search", use_container_width=True)

    if news_query.strip() and (do_search or news_query):
        with st.spinner(f"Searching news for '{news_query}'…"):
            results = search_global_news(news_query, max_results=10)
        if results:
            st.caption(f"{len(results)} results for **'{news_query}'**")
            for item in results:
                with st.expander(f"📰 {item['title']}", expanded=False):
                    st.markdown(f"**Source:** {item['source']}\n\n🔗 [Read full article]({item['url']})")
        else:
            st.info(f"No results found for **'{news_query}'**.")

    st.divider()

    # ── Add ticker section ─────────────────────────────────────────────────────
    st.markdown("### 📊 Analyse a stock")
    st.caption("Select a market and ticker to open a new analysis tab.")

    # Market and Exchange MUST be outside the form so they react dynamically
    # when the user changes them. Only the final ticker selection + button go in the form.
    f1, f2 = st.columns([2, 2])
    with f1:
        market = st.selectbox(
            "Market",
            ["— select —"] + list(MARKET_CONFIG.keys()) + ["🇪🇺 Europe"],
            key="home_market",
        )
    with f2:
        exchange = None
        if market == "🇪🇺 Europe":
            exchange = st.selectbox("Exchange", list(EU_EXCHANGES.keys()), key="home_exchange")

    # Ticker list + custom input + button in a form so Enter works
    with st.form("add_ticker_form"):
        f3, f4, f5 = st.columns([3, 3, 1])

        with f3:
            if market == "— select —":
                st.selectbox("Ticker", ["— select a market above first —"],
                             key="home_ticker_empty", disabled=True)
                display_ticker = ""
            elif market == "🇪🇺 Europe" and exchange:
                ex_cfg = EU_EXCHANGES[exchange]
                ticker_opts = ["— select —"] + [
                    f"{t}  —  {n}" for t, n in zip(ex_cfg["tickers"], ex_cfg["names"])
                ]
                sel = st.selectbox("Ticker", ticker_opts, key="home_eu_ticker")
                display_ticker = "" if sel == "— select —" else sel.split("  —  ")[0].strip()
            else:
                cfg   = MARKET_CONFIG[market]
                all_t = cfg["tickers"] + cfg.get("us_adrs", [])
                all_n = cfg["names"] + cfg.get("adr_names", [])
                ticker_opts = ["— select —"] + [
                    f"{t}  —  {n}" for t, n in zip(all_t, all_n)
                ]
                sel = st.selectbox("Ticker", ticker_opts, key="home_ticker")
                display_ticker = "" if sel == "— select —" else sel.split("  —  ")[0].strip()

        with f4:
            custom_input = st.text_input(
                "Or type a ticker directly",
                placeholder="e.g. RELIANCE, BMW, 0700",
                key="home_custom",
            )

        with f5:
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Open →", type="primary", use_container_width=True)

    if submitted:
        if market == "— select —":
            st.warning("Please select a market first.")
        elif custom_input.strip():
            is_valid, full_ticker, custom_error = validate_custom(custom_input, market, exchange)
            display_ticker = full_ticker.split(".")[0]
            if custom_error:
                st.markdown(f'<div class="warn-box">⚠️ {custom_error}</div>', unsafe_allow_html=True)
            else:
                company = get_company_name(display_ticker, market, exchange)
                add_ticker_tab(full_ticker, display_ticker, company, market, exchange)
        elif not display_ticker:
            st.warning("Please select a ticker or type one directly.")
        else:
            full_ticker = resolve_ticker(display_ticker, market, exchange)
            company     = get_company_name(display_ticker, market, exchange)
            add_ticker_tab(full_ticker, display_ticker, company, market, exchange)


# ══════════════════════════════════════════════════════════════════════════════
# TICKER ANALYSIS TAB
# ══════════════════════════════════════════════════════════════════════════════

else:
    tab_info = next(
        (t for t in st.session_state.open_tickers if t["full"] == st.session_state.active_tab),
        None
    )
    if not tab_info:
        st.session_state.active_tab = "home"
        st.rerun()

    full_ticker  = tab_info["full"]
    display_ticker = tab_info["display"]
    company_name   = tab_info["company"]
    market         = tab_info["market"]
    exchange       = tab_info["exchange"]
    currency_name, currency_sym = get_currency(market, exchange)

    # ── Header ─────────────────────────────────────────────────────────────────
    hcol, dcol = st.columns([4, 1])
    with hcol:
        st.markdown(f"## {company_name} `({full_ticker})` — Sentiment Trading Signal")
        caption_parts = [f"Market: {market}"]
        if exchange:
            caption_parts.append(exchange)
        caption_parts.append(f"Currency: {currency_name} ({currency_sym.strip()})")
        st.caption(" · ".join(caption_parts))
    with dcol:
        days_back = st.selectbox("History", [90, 180, 365], index=2,
                                 format_func=lambda x: f"{x} days", key=f"hist_{full_ticker}")

    # ── Run pipeline ────────────────────────────────────────────────────────────
    with st.spinner(f"Running analysis for {full_ticker}… (~20s first run, cached after)"):
        try:
            data = run_pipeline(full_ticker, days_back)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()

    report     = data["report"]
    metrics    = report["metrics"]
    daily      = data["daily"]
    stock_df   = data["stock_df"]
    scored     = data["scored"]
    signal     = data["signal"]
    confidence = data["confidence"]
    model      = data["model"]

    # ── Trending news for this ticker ──────────────────────────────────────────
    if not scored.empty:
        st.markdown("#### 🔥 Latest news")
        trending_t = (
            scored[["published_at","title","source","sentiment_score"]]
            .sort_values("published_at", ascending=False)
            .head(3)
            .reset_index(drop=True)
        )
        t_cols = st.columns(3)
        for i, (_, row) in enumerate(trending_t.iterrows()):
            score = row["sentiment_score"]
            if score > 0.05:
                badge, badge_color = "Bullish 🟢", "#16a34a"
            elif score < -0.05:
                badge, badge_color = "Bearish 🔴", "#dc2626"
            else:
                badge, badge_color = "Neutral ⚪", "#6b7280"
            pub = pd.to_datetime(row["published_at"]).strftime("%b %d")
            url = row.get("url", "#") if "url" in row.index else "#"
            label = f"{badge}  {row['title']}"
            with t_cols[i]:
                with st.expander(label[:80] + ("…" if len(label) > 80 else ""), expanded=False):
                    link = f"\n\n🔗 [Read full article]({url})" if url != "#" else ""
                    st.markdown(f"**{pub}** · {row['source']} · score: `{score:+.3f}`{link}")
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Signal + metrics ───────────────────────────────────────────────────────
    signal_config = {
         1: ("BUY",  "signal-buy",  "🟢", "#16a34a"),
        -1: ("SELL", "signal-sell", "🔴", "#dc2626"),
         0: ("HOLD", "signal-hold", "⚪", "#9ca3af"),
    }
    label, css_class, icon, color = signal_config[signal]
    col_sig, col_metrics = st.columns([1, 3])

    with col_sig:
        latest_close = float(stock_df["close"].iloc[-1])
        latest_sent  = float(daily["mean_score"].iloc[-1]) if not daily.empty else 0.0
        sent_label   = "bullish" if latest_sent > 0.05 else ("bearish" if latest_sent < -0.05 else "neutral")
        st.markdown(f"""
        <div class="{css_class}">
            <div style="font-size:3rem">{icon}</div>
            <div style="font-family:'DM Mono',monospace;font-size:2rem;color:{color};font-weight:500">{label}</div>
            <div style="color:#9ca3af;font-size:0.85rem;margin-top:0.5rem">
                Confidence: {confidence:.0%}<br>
                Sentiment: {sent_label} ({latest_sent:+.3f})<br>
                Close: {currency_sym}{latest_close:,.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_metrics:
        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Total Return",  f"{metrics['total_return']:+.1f}%")
        m2.metric("Annual Return", f"{metrics['annual_return']:+.1f}%")
        m3.metric("Sharpe Ratio",  f"{metrics['sharpe_ratio']:.2f}")
        m4.metric("Max Drawdown",  f"{metrics['max_drawdown']:.1f}%")
        m5.metric("vs Buy & Hold", f"{metrics['vs_buy_and_hold']:+.1f}%")
        w1,w2,w3 = st.columns(3)
        w1.metric("Win Rate",    f"{metrics['win_rate']:.0f}%")
        w2.metric("Trades",      metrics['n_trades'])
        w3.metric("Test Period", f"{metrics['n_days']} days")

    st.divider()

    # ── Analysis tabs ──────────────────────────────────────────────────────────
    atab1, atab2, atab3, atab4 = st.tabs([
        "📈 Price vs Sentiment", "💼 Backtest", "🧠 Model", "📰 Headlines",
    ])

    with atab1:
        from plotly.subplots import make_subplots
        stock_plot = stock_df.copy()
        stock_plot["date"] = pd.to_datetime(stock_plot["date"])
        stock_plot = stock_plot.set_index("date").sort_index()
        daily_plot = daily.copy()
        if not daily_plot.empty:
            daily_plot.index = pd.to_datetime(daily_plot.index)
        has_s = not daily_plot.empty
        fig = make_subplots(
            rows=3 if has_s else 1, cols=1, shared_xaxes=True,
            subplot_titles=[f"Close price ({currency_name})", "Sentiment score", "Article volume"][:3 if has_s else 1],
            row_heights=[0.5,0.3,0.2] if has_s else [1.0],
            vertical_spacing=0.06,
        )
        fig.add_trace(go.Scatter(
            x=stock_plot.index, y=stock_plot["close"],
            name=f"Close ({currency_sym.strip()})", line=dict(color="#2563eb", width=1.5),
            hovertemplate=f"{currency_sym}%{{y:,.2f}}<extra></extra>",
        ), row=1, col=1)
        if has_s:
            sc = ["#16a34a" if s >= 0 else "#dc2626" for s in daily_plot["mean_score"]]
            fig.add_trace(go.Bar(x=daily_plot.index, y=daily_plot["mean_score"],
                marker_color=sc, showlegend=False, hovertemplate="%{y:+.3f}<extra></extra>"), row=2, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="#4b5563", row=2, col=1)
            fig.add_trace(go.Bar(x=daily_plot.index, y=daily_plot["n_articles"],
                marker_color="#6366f1", showlegend=False, hovertemplate="%{y} articles<extra></extra>"), row=3, col=1)
        fig.update_yaxes(title_text=currency_sym.strip(), row=1, col=1)
        fig.update_layout(height=560 if has_s else 340, paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            font=dict(color="#e5e7eb"), hovermode="x unified", legend=dict(orientation="h"))
        fig.update_yaxes(gridcolor="#1e2130", zeroline=False)
        fig.update_xaxes(gridcolor="#1e2130")
        st.plotly_chart(fig, use_container_width=True)
        if not has_s:
            st.info("No news data found — price chart only.")
        st.caption(f"Prices in **{currency_name}** ({currency_sym.strip()})")

    with atab2:
        from plotly.subplots import make_subplots
        daily_bt = report["daily"]
        bnh_bt   = report["bnh"]["daily"]
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
            subplot_titles=[f"Portfolio value ({currency_sym}10,000 starting capital)", "Daily P&L (%)"],
            row_heights=[0.65, 0.35], vertical_spacing=0.08)
        fig2.add_trace(go.Scatter(x=daily_bt.index, y=daily_bt["portfolio"], name="Strategy",
            line=dict(color="#2563eb", width=2),
            hovertemplate=f"{currency_sym}%{{y:,.0f}}<extra>Strategy</extra>"), row=1, col=1)
        fig2.add_trace(go.Scatter(x=bnh_bt.index, y=bnh_bt["portfolio"], name="Buy & Hold",
            line=dict(color="#6b7280", width=1.5, dash="dash"),
            hovertemplate=f"{currency_sym}%{{y:,.0f}}<extra>Buy & Hold</extra>"), row=1, col=1)
        bc2 = ["#16a34a" if r >= 0 else "#dc2626" for r in daily_bt["strat_ret"]]
        fig2.add_trace(go.Bar(x=daily_bt.index, y=daily_bt["strat_ret"]*100,
            marker_color=bc2, showlegend=False, hovertemplate="%{y:+.2f}%<extra></extra>"), row=2, col=1)
        fig2.update_layout(height=500, paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            font=dict(color="#e5e7eb"), hovermode="x unified", legend=dict(orientation="h"))
        fig2.update_yaxes(gridcolor="#1e2130")
        fig2.update_xaxes(gridcolor="#1e2130")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(f"Currency: **{currency_name}** · Transaction cost: 10bps · Starting: {currency_sym}10,000")

    with atab3:
        import plotly.express as px
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Feature importance")
            if hasattr(model.best_model, "feature_importances_"):
                feat_df = pd.DataFrame({
                    "feature": model.feature_cols,
                    "importance": model.best_model.feature_importances_,
                }).sort_values("importance", ascending=True).tail(15)
                fig3 = px.bar(feat_df, x="importance", y="feature", orientation="h",
                              color="importance", color_continuous_scale=["#1e3a5f","#2563eb","#60a5fa"])
                fig3.update_layout(height=420, paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                    font=dict(color="#e5e7eb"), showlegend=False, coloraxis_showscale=False, margin=dict(l=0))
                fig3.update_xaxes(gridcolor="#1e2130")
                fig3.update_yaxes(gridcolor="#1e2130")
                st.plotly_chart(fig3, use_container_width=True)
        with col_b:
            st.markdown("#### Walk-forward results")
            res = data["results"]
            st.dataframe(pd.DataFrame([
                {"Model": n, "Accuracy": f"{s['accuracy_mean']:.3f}±{s['accuracy_std']:.3f}",
                 "F1": f"{s['f1_mean']:.3f}±{s['f1_std']:.3f}"}
                for n, s in res["summary"].items()
            ]), hide_index=True, use_container_width=True)
            st.markdown("#### Fold detail")
            st.dataframe(pd.DataFrame([
                {"Model": n, "Fold": f["fold"], "Acc": round(f["accuracy"],3),
                 "F1": round(f["f1"],3), "Train": f["train_rows"], "Test": f["test_rows"]}
                for n, folds in res["fold_details"].items() for f in folds
            ]), hide_index=True, use_container_width=True)

    with atab4:
        st.markdown("#### Search headlines")
        news_q = st.text_input("Filter", label_visibility="collapsed",
                               placeholder="Search headlines — e.g. earnings, iPhone, acquisition…",
                               key=f"news_search_{full_ticker}")
        if scored.empty:
            st.warning("No headlines available for this ticker.")
        else:
            disp = scored[["published_at","title","source","sentiment_score"]].sort_values("published_at", ascending=False)
            if news_q.strip():
                disp = disp[disp["title"].str.lower().str.contains(news_q.lower(), na=False)].head(10)
                if disp.empty:
                    st.info(f"No headlines matching **'{news_q}'**.")
            else:
                disp = disp.head(50)
            for _, row in disp.iterrows():
                score = row["sentiment_score"]
                em = "🟢" if score > 0.05 else ("🔴" if score < -0.05 else "⚪")
                pub = pd.to_datetime(row["published_at"]).strftime("%b %d")
                url = row.get("url", "#") if "url" in row.index else "#"
                label = f"{em} {row['title']}"
                with st.expander(label, expanded=False):
                    link = f"\n\n🔗 [Read full article]({url})" if url != "#" else ""
                    st.markdown(f"**{pub}** · {row['source']} · score: `{score:+.3f}`{link}")
