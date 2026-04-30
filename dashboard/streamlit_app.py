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
    /* ── Fonts ────────────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"], .stMarkdown, .stText, p, span, div {
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    }
    h1, h2, h3, code, .mono {
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: -0.02em;
    }

    /* ── Global background ────────────────────────────────────────────────── */
    .stApp, .main, section[data-testid="stSidebar"] {
        background-color: #080c14 !important;
    }
    .block-container {
        padding-top: 1rem !important;
        max-width: 1200px !important;
    }

    /* ── Hide Streamlit chrome ────────────────────────────────────────────── */
    #MainMenu, footer, header[data-testid="stHeader"],
    .stDeployButton, [data-testid="stToolbar"] {
        display: none !important;
    }

    /* ── Dividers ─────────────────────────────────────────────────────────── */
    hr { border-color: #1a2035 !important; margin: 1.5rem 0 !important; }

    /* ── Headings ─────────────────────────────────────────────────────────── */
    h1 { font-size: 1.6rem !important; color: #f1f5f9 !important; font-weight: 600 !important; }
    h2 { font-size: 1.25rem !important; color: #cbd5e1 !important; }
    h3 { font-size: 1.05rem !important; color: #94a3b8 !important; font-weight: 500 !important; }

    /* ── Buttons ──────────────────────────────────────────────────────────── */
    .stButton > button {
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        border: 1px solid #1e293b !important;
        background: #0f172a !important;
        color: #94a3b8 !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:hover {
        border-color: #3b82f6 !important;
        color: #e2e8f0 !important;
        background: #1e293b !important;
    }
    .stButton > button[kind="primary"] {
        background: #3b82f6 !important;
        border-color: #3b82f6 !important;
        color: #fff !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #2563eb !important;
        border-color: #2563eb !important;
    }

    /* ── Inputs & selects ─────────────────────────────────────────────────── */
    .stTextInput > div > div > input,
    .stSelectbox > div > div {
        background: #0d1117 !important;
        border: 1px solid #1e293b !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important;
    }

    /* ── Tabs ─────────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid #1a2035 !important;
        gap: 0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #64748b !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
        border-bottom: 2px solid transparent !important;
    }
    .stTabs [aria-selected="true"] {
        color: #3b82f6 !important;
        border-bottom: 2px solid #3b82f6 !important;
        background: transparent !important;
    }

    /* ── Metrics ──────────────────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: #0d1117 !important;
        border: 1px solid #1a2035 !important;
        border-radius: 10px !important;
        padding: 0.8rem 1rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.72rem !important;
        color: #475569 !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.3rem !important;
        color: #f1f5f9 !important;
        font-weight: 500 !important;
    }
    [data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

    /* ── Expanders ────────────────────────────────────────────────────────── */
    .streamlit-expanderHeader, [data-testid="stExpanderToggleIcon"],
    details > summary {
        background: #0d1117 !important;
        border: 1px solid #1a2035 !important;
        border-radius: 8px !important;
        color: #94a3b8 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.84rem !important;
        list-style: none !important;
    }
    .streamlit-expanderContent, details > div {
        background: #090e18 !important;
        border: 1px solid #1a2035 !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
    }
    /* Hide the _arrow_right material icon text that leaks in some Streamlit versions */
    [data-testid="stExpanderToggleIcon"] { display: none !important; }
    details summary svg { display: none !important; }
    /* Suppress any raw icon text nodes */
    .streamlit-expanderHeader p::before { content: none !important; }

    /* ── Dataframes ───────────────────────────────────────────────────────── */
    [data-testid="stDataFrame"] {
        border: 1px solid #1a2035 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }

    /* ── Captions ─────────────────────────────────────────────────────────── */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #64748b !important;
        font-size: 0.75rem !important;
    }

    /* ── Cards — index ────────────────────────────────────────────────────── */
    .index-card {
        background: linear-gradient(145deg, #0d1117, #111827);
        border: 1px solid #1a2035;
        border-radius: 12px;
        padding: 1rem 1rem 0.4rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    }
    .index-name { font-size: 0.72rem; color: #475569; margin-bottom: 0.15rem;
                  text-transform: uppercase; letter-spacing: 0.06em; }
    .index-val  { font-family: 'JetBrains Mono', monospace; font-size: 1.4rem;
                  font-weight: 500; color: #f1f5f9; line-height: 1.2; }
    .index-chg-up   { color: #22c55e; font-size: 0.8rem; font-weight: 500; }
    .index-chg-down { color: #ef4444; font-size: 0.8rem; font-weight: 500; }
    .index-status   { font-size: 0.7rem; color: #334155; margin-top: 0.2rem; }

    /* ── Cards — trending news ────────────────────────────────────────────── */
    .t-card {
        background: linear-gradient(145deg, #0d1117, #0f1623);
        border: 1px solid #1a2035;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        height: 100%;
        border-top-width: 2px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        transition: border-color 0.2s;
    }
    .t-title { font-size: 0.87rem; font-weight: 500; line-height: 1.45;
               margin-bottom: 0.5rem; color: #cbd5e1; }
    .t-meta  { font-size: 0.72rem; color: #334155; }
    .badge   { display:inline-block; padding:0.1rem 0.4rem; border-radius:4px;
               font-size:0.7rem; color:#fff; margin-left:0.3rem;
               font-weight: 500; letter-spacing: 0.02em; }

    /* ── Signal blocks ────────────────────────────────────────────────────── */
    .signal-buy {
        background: linear-gradient(145deg, #021a0e, #052e16);
        border: 1px solid #16a34a;
        border-radius: 14px; padding: 1.75rem; text-align: center;
        box-shadow: 0 0 30px rgba(22,163,74,0.12);
    }
    .signal-sell {
        background: linear-gradient(145deg, #1a0505, #2d0a0a);
        border: 1px solid #dc2626;
        border-radius: 14px; padding: 1.75rem; text-align: center;
        box-shadow: 0 0 30px rgba(220,38,38,0.12);
    }
    .signal-hold {
        background: linear-gradient(145deg, #0a0e18, #111827);
        border: 1px solid #334155;
        border-radius: 14px; padding: 1.75rem; text-align: center;
        box-shadow: 0 0 20px rgba(0,0,0,0.3);
    }

    /* ── Headlines ────────────────────────────────────────────────────────── */
    .hl-row {
        border-left: 2px solid;
        padding: 0.5rem 0.85rem;
        margin-bottom: 0.35rem;
        border-radius: 0 6px 6px 0;
        background: rgba(255,255,255,0.015);
        font-size: 0.84rem;
        transition: background 0.15s;
    }
    .hl-row:hover { background: rgba(255,255,255,0.04); }

    /* ── Warn box ─────────────────────────────────────────────────────────── */
    .warn-box {
        background: #1a0e00;
        border: 1px solid #92400e;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        color: #fcd34d;
    }

    /* ── Sentiment cards ──────────────────────────────────────────────────── */
    .sent-card {
        border-radius: 12px;
        padding: 0.85rem 0.75rem;
        text-align: center;
        border: 1px solid;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }

    /* ── Spinner ──────────────────────────────────────────────────────────── */
    .stSpinner > div { border-color: #3b82f6 transparent transparent !important; }

    /* ── Info / warning / success boxes ──────────────────────────────────── */
    .stAlert { border-radius: 10px !important; border-width: 1px !important; }

    /* ── Slider ───────────────────────────────────────────────────────────── */
    .stSlider [data-baseweb="slider"] { margin-top: 0.5rem; }

    /* ── Section headers with subtle rule ────────────────────────────────── */
    .section-header {
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #64748b;
        margin-bottom: 0.75rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #1a2035;
    }
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

# Market hours per index — (open_hour, close_hour) in local time, timezone string
MARKET_HOURS = {
    "^GSPC":  {"tz": "America/New_York",  "open": (9,  30), "close": (16, 0),  "label": "ET"},
    "^FTSE":  {"tz": "Europe/London",     "open": (8,  0),  "close": (16, 30), "label": "BST/GMT"},
    "^N225":  {"tz": "Asia/Tokyo",        "open": (9,  0),  "close": (15, 30), "label": "JST"},
    "^BSESN": {"tz": "Asia/Kolkata",      "open": (9,  15), "close": (15, 30), "label": "IST"},
    "^GDAXI": {"tz": "Europe/Berlin",     "open": (9,  0),  "close": (17, 30), "label": "CET/CEST"},
}


def get_market_open_status(ticker: str) -> dict:
    """
    Return whether a market is currently open, its local time,
    and a human-readable status string.
    """
    import pytz
    hours = MARKET_HOURS.get(ticker)
    if not hours:
        return {"is_open": False, "local_time": "", "status": ""}

    tz       = pytz.timezone(hours["tz"])
    now_local = datetime.now(tz)
    weekday   = now_local.weekday()   # 0=Mon, 6=Sun

    oh, om = hours["open"]
    ch, cm = hours["close"]

    open_mins  = oh * 60 + om
    close_mins = ch * 60 + cm
    now_mins   = now_local.hour * 60 + now_local.minute

    is_weekday = weekday < 5
    in_hours   = open_mins <= now_mins < close_mins
    is_open    = is_weekday and in_hours

    local_time_str = now_local.strftime(f"%H:%M {hours['label']}")

    if is_open:
        # Time until close
        mins_left = close_mins - now_mins
        h_left, m_left = divmod(mins_left, 60)
        if h_left > 0:
            time_left = f"{h_left}h {m_left}m left"
        else:
            time_left = f"{m_left}m left"
        status = f"🟢 OPEN  {local_time_str}  ({time_left})"
    elif not is_weekday:
        # Weekend — show next open
        days_to_monday = 7 - weekday if weekday >= 5 else 0
        next_open_dt   = now_local + timedelta(days=days_to_monday)
        next_open_str  = next_open_dt.strftime(f"%a %d %b  {oh:02d}:{om:02d} {hours['label']}")
        status = f"🔴 CLOSED  Opens {next_open_str}"
    else:
        # Weekday but outside hours
        if now_mins < open_mins:
            opens_in = open_mins - now_mins
            h_in, m_in = divmod(opens_in, 60)
            if h_in > 0:
                wait = f"in {h_in}h {m_in}m"
            else:
                wait = f"in {m_in}m"
            status = f"🟡 PRE-MARKET  {local_time_str}  (opens {wait})"
        else:
            status = f"🔴 CLOSED  {local_time_str}"

    return {
        "is_open":    is_open,
        "local_time": local_time_str,
        "status":     status,
    }


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

def get_market_status() -> dict:
    """
    Detect whether today is a weekend — markets closed.
    Returns dict with is_weekend, next_open, note.
    """
    import pytz
    today   = datetime.now(pytz.utc)
    weekday = today.weekday()   # 0=Mon ... 6=Sun
    is_weekend = weekday >= 5

    if is_weekend:
        days_to_monday = 7 - weekday   # Sat→2, Sun→1
        next_open = (today + timedelta(days=days_to_monday)).strftime("%A %b %d")
        note = (
            "Markets are closed today. Signals are based on Friday's closing prices. "
            f"Weekend news is included in sentiment scoring and will feed into "
            f"Monday's signal when markets reopen ({next_open})."
        )
    else:
        next_open = None
        note = None

    return {"is_weekend": is_weekend, "weekday": weekday,
            "next_open": next_open, "note": note}


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
def load_market_sentiment() -> dict:
    """Score sentiment across all markets — cached for 1 hour."""
    try:
        from src.market_sentiment import MarketSentimentScorer
        scorer = MarketSentimentScorer()
        return scorer.score_all_markets()
    except Exception as e:
        return {}


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

# Nav bar + live clock
# Strategy: render the entire nav bar including clock inside components.html
# so the script runs. height=38 makes it visible. Negative margin pulls it flush.
import streamlit.components.v1 as components

_nav_html = (
    "<style>"
    "*{box-sizing:border-box;margin:0;padding:0;}"
    "body{background:#080c14;font-family:'Inter',system-ui,sans-serif;}"
    ".nav{"
    "display:flex;align-items:center;justify-content:space-between;"
    "padding:0.6rem 0.25rem;border-bottom:1px solid #1a2035;"
    "}"
    ".logo{font-family:'JetBrains Mono','DM Mono',monospace;font-size:1rem;font-weight:600;color:#3b82f6;letter-spacing:-0.01em;}"
    ".sub{font-size:0.73rem;color:#475569;margin-left:0.6rem;letter-spacing:0.01em;}"
    ".clk{font-family:'JetBrains Mono','DM Mono',monospace;font-size:0.75rem;color:#64748b;letter-spacing:0.02em;}"
    ".dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:#22c55e;margin-right:6px;vertical-align:middle;}"
    "</style>"
    "<div class='nav'>"
    "<span><span class='logo'>[ICON] Sentiment Signal</span>"
    "<span class='sub'>FinBERT-powered global trading signals</span></span>"
    "<span class='clk' id='clk'>--</span>"
    "</div>"
    "<script>"
    "(function(){"
    "var D=['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];"
    "var M=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];"
    "function tick(){"
    "var el=document.getElementById('clk');"
    "if(!el)return;"
    "var n=new Date();"
    "var hh=String(n.getHours()).padStart(2,'0');"
    "var mm=String(n.getMinutes()).padStart(2,'0');"
    "var tz='Local';"
    "try{tz=new Intl.DateTimeFormat('en',{timeZoneName:'short'})"
    ".formatToParts(n)"
    ".find(function(p){return p.type==='timeZoneName';}).value;"
    "}catch(e){}"
    "el.textContent='[CLK] '+D[n.getDay()]+' '+n.getDate()+' '+M[n.getMonth()]"
    "+'  '+hh+':'+mm+' '+tz;"
    "}"
    "tick();setInterval(tick,1000);"
    "})();"
    "</script>"
)
# Insert emoji after building string (avoids encoding issues in Python 3.14)
_nav_html = _nav_html.replace("[ICON]", "📡").replace("[CLK]", "🕐")
components.html(_nav_html, height=42, scrolling=False)

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

    # ── Weekend / market closed banner ───────────────────────────────────────
    mkt_status = get_market_status()
    if mkt_status["is_weekend"]:
        st.info(
            f"📅 **Weekend mode** — {mkt_status['note']}",
            icon="🏖️",
        )

    # ── Analyse a stock — primary action at top ────────────────────────────────
    st.markdown("### 📊 Analyse a stock")
    st.caption("Select a market and ticker to open a new analysis tab. Results persist as tabs above.")

    # Market and Exchange MUST be outside the form so they react dynamically
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

    st.divider()

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
                mkt_open = get_market_open_status(idx["ticker"])
                st.markdown(f"""
                <div style="padding:0.7rem 0.5rem 0.2rem">
                    <div style="font-size:0.75rem;color:#9ca3af">{idx['region']} {idx['name']}</div>
                    <div style="font-family:'DM Mono',monospace;font-size:1.3rem;font-weight:500;
                                color:#f9fafb;line-height:1.3">{idx['symbol']}{current:,.0f}</div>
                    <div style="font-size:0.82rem;color:{chg_color};font-weight:500">
                        {chg_sym} {abs(chg_pct):.2f}% today
                    </div>
                    <div style="font-size:0.72rem;color:#6b7280;margin-top:0.25rem">
                        {mkt_open['status']}
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

    # ── Market sentiment ──────────────────────────────────────────────────────
    st.markdown("#### 🧠 Market sentiment (FinBERT)")
    st.caption("Aggregate sentiment scored from representative stock baskets — updates hourly.")

    with st.spinner("Scoring market sentiment…"):
        mkt_sentiment = load_market_sentiment()

    if mkt_sentiment:
        SENTIMENT_MARKETS = ["🇺🇸 USA", "🇮🇳 India", "🇪🇺 Europe", "🇨🇳 China / HK", "🇯🇵 Japan"]
        sent_cols = st.columns(len(SENTIMENT_MARKETS))
        for col, mkt in zip(sent_cols, SENTIMENT_MARKETS):
            s = mkt_sentiment.get(mkt, {})
            score = s.get("score", 0.0)
            label = s.get("label", "Neutral")
            n     = s.get("n_articles", 0)
            stocks = s.get("stocks", [])
            error  = s.get("error")

            if label == "Bullish":
                icon, color, bg = "🟢", "#22c55e", "rgba(34,197,94,0.08)"
            elif label == "Bearish":
                icon, color, bg = "🔴", "#ef4444", "rgba(239,68,68,0.08)"
            else:
                icon, color, bg = "⚪", "#9ca3af", "rgba(156,163,175,0.08)"

            with col:
                if error and not stocks:
                    st.markdown(f"""
                    <div style="background:{bg};border:1px solid {color}33;border-radius:8px;
                                padding:0.7rem;text-align:center;">
                        <div style="font-size:0.75rem;color:#9ca3af">{mkt}</div>
                        <div style="color:#6b7280;font-size:0.78rem;margin-top:0.3rem">Unavailable</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    stock_detail = " · ".join(
                        f"{s['ticker']} {s['score']:+.2f}" for s in stocks
                    ) if stocks else ""
                    st.markdown(f"""
                    <div style="background:{bg};border:1px solid {color}55;border-radius:8px;
                                padding:0.7rem 0.8rem;text-align:center;">
                        <div style="font-size:0.75rem;color:#9ca3af;margin-bottom:0.2rem">{mkt}</div>
                        <div style="font-size:1.5rem">{icon}</div>
                        <div style="font-family:'DM Mono',monospace;font-size:1rem;
                                    color:{color};font-weight:500">{label}</div>
                        <div style="font-family:'DM Mono',monospace;font-size:0.9rem;
                                    color:{color}">{score:+.3f}</div>
                        <div style="font-size:0.72rem;color:#6b7280;margin-top:0.3rem">
                            {n} articles
                        </div>
                        <div style="font-size:0.68rem;color:#4b5563;margin-top:0.15rem">
                            {stock_detail}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("Market sentiment unavailable — FinBERT is loading or RSS feeds are down.")

    st.divider()

    # ── Trending news ──────────────────────────────────────────────────────────
    st.markdown("### 🔥 Trending financial news")

    with st.spinner("Loading trending news…"):
        trending = load_trending_news(3)

    if trending:
        t_cols = st.columns(3)
        border_colors = ["#3b82f6", "#8b5cf6", "#06b6d4"]
        for i, (col, item) in enumerate(zip(t_cols, trending)):
            bc = border_colors[i]
            url = item.get("url", "#")
            with col:
                st.markdown(f"""
                <div class="t-card" style="border-top-color:{bc}">
                    <div class="t-title">{item['title']}</div>
                    <div class="t-meta">{item['source']}</div>
                </div>
                """, unsafe_allow_html=True)
                if url and url != "#":
                    st.markdown(f"[Read article →]({url})", unsafe_allow_html=False)
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
                url = item.get("url","#")
                link_html = f"&nbsp;<a href='{url}' target='_blank' style='color:#3b82f6;font-size:0.75rem'>Read →</a>" if url != "#" else ""
                st.markdown(f"""
                <div class="hl-row" style="border-color:#3b82f6">
                    📰 <strong>{item['title']}</strong>{link_html}<br>
                    <span style="color:#475569;font-size:0.75rem">{item['source']}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(f"No results found for **'{news_query}'**.")




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
        _mkt = get_market_status()
        if _mkt["is_weekend"]:
            caption_parts.append("📅 Weekend — prices as of Friday close")
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
                badge, bc, badge_color = "Bullish", "#22c55e", "#16a34a"
            elif score < -0.05:
                badge, bc, badge_color = "Bearish", "#ef4444", "#dc2626"
            else:
                badge, bc, badge_color = "Neutral", "#64748b", "#6b7280"
            pub = pd.to_datetime(row["published_at"]).strftime("%b %d")
            url = row.get("url", "#") if "url" in row.index else "#"
            with t_cols[i]:
                st.markdown(f"""
                <div class=\"t-card\" style=\"border-top-color:{bc}\">
                    <div style=\"margin-bottom:0.3rem\">
                        <span class=\"badge\" style=\"background:{badge_color}\">{badge}</span>
                        <span style=\"font-size:0.7rem;color:#475569;margin-left:0.4rem\">{score:+.3f}</span>
                    </div>
                    <div class=\"t-title\">{row['title']}</div>
                    <div class=\"t-meta\">{pub} · {row['source']}</div>
                </div>
                """, unsafe_allow_html=True)
                if url and url != "#":
                    st.markdown(f"[Read →]({url})")
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Log prediction to Supabase (silently, non-blocking) ───────────────────
    try:
        from src.feedback_logger import FeedbackLogger
        from datetime import date as _date
        _fb = FeedbackLogger()
        _today = _date.today().isoformat()
        _sent  = float(daily["mean_score"].iloc[-1]) if not daily.empty else 0.0
        _fb.log_prediction(
            ticker=full_ticker, trade_date=_today,
            predicted=signal, confidence=confidence, sentiment=_sent,
        )
        # Also update yesterday's actual if we have price data
        if len(stock_df) >= 2:
            _yesterday = stock_df.iloc[-2]
            _actual    = int(_yesterday.get("direction", 0))
            _ydate     = str(stock_df.iloc[-2]["date"])
            _fb.update_actual(ticker=full_ticker, trade_date=_ydate, actual=_actual)
    except Exception:
        pass  # Supabase not configured — continue without logging

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

    # ── Sentiment disagreement panel ──────────────────────────────────────────
    if not daily.empty:
        last = daily.iloc[-1]

        # Compute disagreement metrics from latest day
        std_score   = float(last.get("std_score",  0.0) or 0.0)
        sent_range  = float(last.get("sentiment_range", 0.0) or 0.0)
        pct_pos     = float(last.get("pct_positive", 0.0) or 0.0) * 100
        pct_neg     = float(last.get("pct_negative", 0.0) or 0.0) * 100
        pct_neu     = float(last.get("pct_neutral",  0.0) or 0.0) * 100
        n_arts      = int(last.get("n_articles", 0) or 0)
        vol_spike   = float(last.get("volume_spike", 1.0) or 1.0)

        # Disagreement level
        if std_score > 0.4:
            dis_label, dis_color, dis_note = "High", "#ef4444", "Headlines sharply divided — market hasn't priced in the uncertainty yet. Larger price move possible."
        elif std_score > 0.2:
            dis_label, dis_color, dis_note = "Moderate", "#f59e0b", "Some disagreement between headlines — mixed signals, interpret with caution."
        else:
            dis_label, dis_color, dis_note = "Low", "#22c55e", "Headlines largely agree — sentiment signal is more reliable today."

        st.markdown("#### 📊 Today's sentiment breakdown")
        dis_col1, dis_col2, dis_col3 = st.columns([2, 2, 3])

        with dis_col1:
            # Bullish / Bearish / Neutral breakdown bar
            st.markdown(f"""
            <div style="background:#0d1117;border:1px solid #1a2035;border-radius:10px;padding:0.9rem 1rem;">
                <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;
                            letter-spacing:0.08em;margin-bottom:0.6rem">Headline split</div>
                <div style="display:flex;height:10px;border-radius:5px;overflow:hidden;margin-bottom:0.6rem">
                    <div style="width:{pct_pos:.0f}%;background:#22c55e"></div>
                    <div style="width:{pct_neu:.0f}%;background:#475569"></div>
                    <div style="width:{pct_neg:.0f}%;background:#ef4444"></div>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:0.75rem">
                    <span style="color:#22c55e">🟢 {pct_pos:.0f}% bullish</span>
                    <span style="color:#475569">⚪ {pct_neu:.0f}%</span>
                    <span style="color:#ef4444">🔴 {pct_neg:.0f}% bearish</span>
                </div>
                <div style="margin-top:0.5rem;font-size:0.72rem;color:#334155">
                    {n_arts} articles · {"⚡ Volume spike" if vol_spike > 1.5 else "Normal volume"}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with dis_col2:
            # Disagreement gauge
            gauge_pct = min(int(std_score / 0.6 * 100), 100)
            filled    = "█" * (gauge_pct // 10)
            empty     = "░" * (10 - gauge_pct // 10)
            st.markdown(f"""
            <div style="background:#0d1117;border:1px solid #1a2035;border-radius:10px;padding:0.9rem 1rem;">
                <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;
                            letter-spacing:0.08em;margin-bottom:0.5rem">Disagreement</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;
                            color:{dis_color};letter-spacing:0.05em;margin-bottom:0.3rem">
                    {filled}{empty}
                </div>
                <div style="font-size:1rem;font-weight:600;color:{dis_color};
                            font-family:'JetBrains Mono',monospace">
                    {dis_label}
                </div>
                <div style="font-size:0.72rem;color:#475569;margin-top:0.25rem">
                    std={std_score:.3f} · range={sent_range:.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with dis_col3:
            # Interpretation note
            st.markdown(f"""
            <div style="background:#0d1117;border:1px solid {dis_color}44;border-radius:10px;
                        padding:0.9rem 1rem;height:100%;border-left:3px solid {dis_color}">
                <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;
                            letter-spacing:0.08em;margin-bottom:0.4rem">What this means</div>
                <div style="font-size:0.82rem;color:#cbd5e1;line-height:1.5">{dis_note}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Analysis tabs ──────────────────────────────────────────────────────────
    atab1, atab2, atab3, atab4, atab5 = st.tabs([
        "📈 Price vs Sentiment", "💼 Backtest", "🧠 Model", "📰 Headlines", "📊 Model Performance",
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
            # Colour bars by both direction AND disagreement:
            # bright = consensus, muted/orange = high disagreement
            def _bar_color(row):
                std = row.get("std_score", 0) or 0
                score = row["mean_score"]
                if std > 0.4:
                    return "#f59e0b"   # amber = high disagreement regardless of direction
                return "#22c55e" if score >= 0 else "#ef4444"

            if "std_score" in daily_plot.columns:
                sc = [_bar_color(row) for _, row in daily_plot.iterrows()]
                hover = [
                    f"Score: {row['mean_score']:+.3f}<br>Std: {row.get('std_score',0):.3f}<br>Articles: {int(row.get('n_articles',0))}"
                    for _, row in daily_plot.iterrows()
                ]
            else:
                sc = ["#22c55e" if s >= 0 else "#ef4444" for s in daily_plot["mean_score"]]
                hover = [f"{s:+.3f}" for s in daily_plot["mean_score"]]

            fig.add_trace(go.Bar(x=daily_plot.index, y=daily_plot["mean_score"],
                marker_color=sc, showlegend=False,
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover), row=2, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="#4b5563", row=2, col=1)

            # Add a subtle legend note
            fig.add_annotation(
                text="🟡 amber = high disagreement between headlines",
                xref="paper", yref="paper", x=1, y=0.42,
                showarrow=False, font=dict(size=9, color="#64748b"),
                xanchor="right",
            )
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
        h_search_col, h_mode_col = st.columns([4, 1])
        with h_search_col:
            news_q = st.text_input("Filter", label_visibility="collapsed",
                                   placeholder="Search headlines — e.g. earnings, iPhone, acquisition…",
                                   key=f"news_search_{full_ticker}")
        with h_mode_col:
            group_mode = st.toggle("Group by sentiment", value=True, key=f"group_{full_ticker}")

        if scored.empty:
            st.warning("No headlines available for this ticker.")
        else:
            disp = scored[["published_at","title","source","sentiment_score"]].sort_values("published_at", ascending=False)
            if "url" in scored.columns:
                disp["url"] = scored["url"]

            if news_q.strip():
                disp = disp[disp["title"].str.lower().str.contains(news_q.lower(), na=False)].head(10)
                if disp.empty:
                    st.info(f"No headlines matching **'{news_q}'**.")
                    group_mode = False
            else:
                disp = disp.head(50)

            def _render_headline(row):
                score = row["sentiment_score"]
                bc  = "#22c55e" if score > 0.05 else ("#ef4444" if score < -0.05 else "#475569")
                em  = "🟢" if score > 0.05 else ("🔴" if score < -0.05 else "⚪")
                pub = pd.to_datetime(row["published_at"]).strftime("%b %d")
                url = row.get("url", "#") if "url" in row.index else "#"
                link_html = f"&nbsp;<a href='{url}' target='_blank' style='color:#3b82f6;font-size:0.75rem'>Read →</a>" if url and url != "#" else ""
                st.markdown(f"""
                <div class="hl-row" style="border-color:{bc}">
                    {em} <strong>{row['title']}</strong>{link_html}<br>
                    <span style="color:#475569;font-size:0.75rem">{pub} · {row['source']} · {score:+.3f}</span>
                </div>
                """, unsafe_allow_html=True)

            if group_mode and not news_q.strip():
                bullish = disp[disp["sentiment_score"] >  0.05]
                bearish = disp[disp["sentiment_score"] < -0.05]
                neutral = disp[(disp["sentiment_score"] >= -0.05) & (disp["sentiment_score"] <= 0.05)]

                # Summary disagreement note
                total = len(disp)
                if total > 0:
                    pct_b = len(bullish)/total*100
                    pct_n = len(neutral)/total*100
                    pct_r = len(bearish)/total*100
                    st.markdown(f"""
                    <div style="background:#0d1117;border:1px solid #1a2035;border-radius:8px;
                                padding:0.7rem 1rem;margin-bottom:0.75rem;
                                display:flex;gap:1.5rem;align-items:center">
                        <span style="font-size:0.75rem;color:#64748b">{total} headlines</span>
                        <span style="color:#22c55e;font-size:0.82rem;font-weight:500">🟢 {pct_b:.0f}% bullish</span>
                        <span style="color:#94a3b8;font-size:0.82rem">⚪ {pct_n:.0f}% neutral</span>
                        <span style="color:#ef4444;font-size:0.82rem;font-weight:500">🔴 {pct_r:.0f}% bearish</span>
                    </div>
                    """, unsafe_allow_html=True)

                if not bullish.empty:
                    st.markdown(f"**🟢 Bullish** ({len(bullish)})")
                    for _, row in bullish.iterrows():
                        _render_headline(row)
                if not bearish.empty:
                    st.markdown(f"**🔴 Bearish** ({len(bearish)})")
                    for _, row in bearish.iterrows():
                        _render_headline(row)
                if not neutral.empty:
                    st.markdown(f"**⚪ Neutral** ({len(neutral)})")
                    for _, row in neutral.iterrows():
                        _render_headline(row)
            else:
                for _, row in disp.iterrows():
                    _render_headline(row)

    # ── Tab 5: Performance & Retrain ───────────────────────────────────────────
    with atab5:
        st.markdown("#### 🔁 Model performance & feedback loop")
        st.caption(
            "Predictions are logged to Supabase automatically. "
            "As actual prices arrive, accuracy is computed and stored. "
            "Use the Retrain button to incorporate new feedback into the model."
        )

        # ── Supabase setup check ───────────────────────────────────────────────
        supabase_ok = False
        try:
            from src.feedback_logger import FeedbackLogger
            fb = FeedbackLogger()
            supabase_ok = True
        except Exception as e:
            st.warning(
                f"⚠️ Supabase not configured — feedback logging is disabled.\n\n"
                f"To enable: add `[supabase]` credentials to your Streamlit secrets.\n\n"
                f"Error: `{e}`"
            )

        if supabase_ok:
            # ── Connection status + table health ───────────────────────────────
            ok, msg = fb.test_connection()
            if ok:
                st.success(f"🟢 {msg}")
                stats = fb.table_stats()
                if stats:
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Total rows",    stats.get("total", 0))
                    s2.metric("Pending actuals", stats.get("pending", 0))
                    s3.metric("Retrained rows", stats.get("retrained", 0))
                    s4.metric("Row limit",      f"{stats.get('max_rows',0):,}")
                    if stats.get("total", 0) > stats.get("max_rows", 10000) * 0.8:
                        st.warning("⚠️ Table is 80%+ full — cleanup will run automatically on next prediction.")
            else:
                st.error(f"🔴 {msg}")

            st.markdown("---")

            # ── Performance metrics ────────────────────────────────────────────
            perf_df = fb.get_performance(ticker=full_ticker, days_back=90)

            if perf_df.empty:
                st.info(
                    "No feedback data yet for this ticker. "
                    "Predictions are being logged — check back after the next trading day "
                    "when actual prices arrive."
                )
            else:
                # Summary stats
                total    = len(perf_df)
                accuracy = perf_df["correct"].mean() * 100
                avg_conf = perf_df["confidence"].mean()

                p1, p2, p3 = st.columns(3)
                p1.metric("Predictions logged",  total)
                p2.metric("Live accuracy",        f"{accuracy:.1f}%")
                p3.metric("Avg confidence",       f"{avg_conf:.1%}")

                # Accuracy over time chart
                daily_acc = (
                    perf_df.groupby("trade_date")["correct"]
                    .mean()
                    .reset_index()
                    .rename(columns={"correct": "accuracy"})
                )
                daily_acc["accuracy"] *= 100

                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(
                    x=daily_acc["trade_date"], y=daily_acc["accuracy"],
                    mode="lines+markers",
                    name="Daily accuracy",
                    line=dict(color="#2563eb", width=2),
                    hovertemplate="%{y:.1f}%<extra></extra>",
                ))
                fig_acc.add_hline(
                    y=50, line_dash="dash", line_color="#6b7280",
                    annotation_text="Random baseline (50%)",
                    annotation_position="bottom right",
                )
                fig_acc.update_layout(
                    height=280,
                    title="Prediction accuracy over time",
                    paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                    font=dict(color="#e5e7eb"),
                    yaxis=dict(range=[0, 100], gridcolor="#1e2130", ticksuffix="%"),
                    xaxis=dict(gridcolor="#1e2130"),
                    margin=dict(l=0, r=0, t=40, b=0),
                )
                st.plotly_chart(fig_acc, use_container_width=True)

                # Prediction history table
                st.markdown("#### Recent predictions")
                display_perf = perf_df[["trade_date","predicted","actual","correct","confidence","sentiment"]].copy()
                display_perf["trade_date"] = display_perf["trade_date"].dt.strftime("%Y-%m-%d")
                display_perf["predicted"]  = display_perf["predicted"].map({1:"▲ UP", -1:"▼ DOWN"})
                display_perf["actual"]     = display_perf["actual"].map({1:"▲ UP", -1:"▼ DOWN"})
                display_perf["correct"]    = display_perf["correct"].map({True:"✓", False:"✗"})
                display_perf["confidence"] = display_perf["confidence"].apply(lambda x: f"{x:.1%}")
                display_perf["sentiment"]  = display_perf["sentiment"].apply(lambda x: f"{x:+.3f}")
                st.dataframe(display_perf.head(20), hide_index=True, use_container_width=True)

            st.divider()

            # ── Automated retraining info ──────────────────────────────────────
            st.markdown("#### 🤖 Automated retraining")
            st.info(
                "The model retrains automatically every weekday at **8:00 AM ET** "
                "via GitHub Actions — one hour before US market open. "
                "Indian and Asian market tickers retrain on the same schedule, "
                "capturing overnight news. No manual action needed."
            )
            st.caption(
                "Retraining incorporates yesterday's actual outcomes, "
                "updates today's prediction, marks feedback rows as retrained, "
                "and cleans up rows older than 30 days."
            )
