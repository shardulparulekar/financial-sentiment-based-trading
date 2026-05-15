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


if "sage_open" not in st.session_state:
    st.session_state.sage_open = False

st.set_page_config(
    page_title="Sentiment Signal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded" if st.session_state.sage_open else "collapsed",
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
        "tickers": ["NVDA","AAPL","MSFT","GOOGL","AMZN","META","TSLA","BRK-B","AVGO","LLY"],
        "names":   ["NVIDIA","Apple","Microsoft","Alphabet","Amazon","Meta","Tesla","Berkshire Hathaway","Broadcom","Eli Lilly"],
    },
    "🇮🇳 India (NSE)": {
        "suffix": ".NS", "currency": "INR", "symbol": "₹", "region": "IN",
        "tickers": ["RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","BHARTIARTL","LT","SBIN","ADANIENT"],
        "names":   ["Reliance","TCS","HDFC Bank","Infosys","ICICI Bank","Hindustan Unilever","Bharti Airtel","L&T","SBI","Adani Enterprises"],
    },
    "🇨🇳 China / HK": {
        "suffix": ".HK", "currency": "HKD", "symbol": "HK$", "region": "HK",
        "tickers":   ["0700","9988","1398","0939","1211","2318","0857","3690"],
        "names":     ["Tencent","Alibaba HK","ICBC","CCB","BYD","Ping An","PetroChina","Meituan"],
        "us_adrs":   ["JD","BIDU","PDD","NIO","BABA"],
        "adr_names": ["JD.com","Baidu","Pinduoduo","NIO","Alibaba"],
    },
    "🇯🇵 Japan": {
        "suffix": ".T", "currency": "JPY", "symbol": "¥", "region": "JP",
        "tickers": ["7203","6758","8306","9984","6861","7974","8035","6501","7267","8316"],
        "names":   ["Toyota","Sony","Mitsubishi UFJ","SoftBank","Keyence","Nintendo","Tokyo Electron","Hitachi","Honda","SMFG"],
    },
}

EU_EXCHANGES = {
    "🇳🇱 Amsterdam": {
        "suffix": ".AS", "currency": "EUR", "symbol": "€",
        "tickers": ["ASML","PRX","INGA","HEIA","AGN","ASRNL","RAND","ABN","WKL","ADYEN"],
        "names":   ["ASML","Prosus","ING","Heineken","Aegon","ASR Nederland","Randstad","ABN Amro","Wolters Kluwer","Adyen"],
        # argenx primary listing moved to Nasdaq — ticker is plain ARGX (no .AS suffix)
        "us_exceptions":      ["ARGX"],
        "us_exception_names": ["Argenx"],
    },
    "🇩🇪 Frankfurt": {
        "suffix": ".DE", "currency": "EUR", "symbol": "€",
        "tickers": ["SAP","SIE","VOW3","MBG","ALV","BAS","BMW","DTE","IFX","ADS"],
        "names":   ["SAP","Siemens","Volkswagen","Mercedes-Benz","Allianz","BASF","BMW","Deutsche Telekom","Infineon","Adidas"],
    },
    "🇫🇷 Paris": {
        "suffix": ".PA", "currency": "EUR", "symbol": "€",
        "tickers": ["MC","TTE","OR","SU","AIR","SAN","BNP","CS","SAF","RMS"],
        "names":   ["LVMH","TotalEnergies","L'Oréal","Schneider Electric","Airbus","Sanofi","BNP Paribas","AXA","Safran","Hermès"],
    },
    "🇬🇧 London": {
        "suffix": ".L", "currency": "GBP", "symbol": "£",
        "tickers": ["SHEL","HSBA","ULVR","BP","AZN","GSK","DGE","RIO","BATS","BARC"],
        "names":   ["Shell","HSBC","Unilever","BP","AstraZeneca","GSK","Diageo","Rio Tinto","British American Tobacco","Barclays"],
    },
    "🇨🇭 Zurich": {
        "suffix": ".SW", "currency": "CHF", "symbol": "CHF ",
        "tickers": ["NESN","ROG","NOVN","UBSG","ZURN","ABBN","CFR","SREN","LONN","HOLN"],
        "names":   ["Nestlé","Roche","Novartis","UBS","Zurich Insurance","ABB","Richemont","Swiss Re","Lonza","Holcim"],
    },
    "🇮🇹 Milan": {
        "suffix": ".MI", "currency": "EUR", "symbol": "€",
        "tickers": ["ENEL","ENI","RACE","ISP","UCG","STLAM","PIRC","TIT","LDO","MONC"],
        "names":   ["Enel","Eni","Ferrari","Intesa Sanpaolo","UniCredit","Stellantis","Pirelli","Telecom Italia","Leonardo","Moncler"],
    },
}

WORLD_INDICES = [
    {"name": "S&P 500",    "ticker": "^GSPC",  "symbol": "$",   "region": "🇺🇸"},
    {"name": "FTSE 100",   "ticker": "^FTSE",  "symbol": "£",   "region": "🇬🇧"},
    {"name": "Nikkei 225", "ticker": "^N225",  "symbol": "¥",   "region": "🇯🇵"},
    {"name": "Sensex",     "ticker": "^BSESN", "symbol": "₹",   "region": "🇮🇳"},
    {"name": "DAX",        "ticker": "^GDAXI", "symbol": "€",   "region": "🇩🇪"},
    {"name": "Hang Seng",  "ticker": "^HSI",   "symbol": "HK$", "region": "🇭🇰"},
]

# Market hours per index — (open_hour, close_hour) in local time, timezone string
MARKET_HOURS = {
    "^GSPC":  {"tz": "America/New_York",  "open": (9,  30), "close": (16, 0),  "label": "ET"},
    "^FTSE":  {"tz": "Europe/London",     "open": (8,  0),  "close": (16, 30), "label": "BST/GMT"},
    "^N225":  {"tz": "Asia/Tokyo",        "open": (9,  0),  "close": (15, 30), "label": "JST"},
    "^BSESN": {"tz": "Asia/Kolkata",      "open": (9,  15), "close": (15, 30), "label": "IST"},
    "^GDAXI": {"tz": "Europe/Berlin",     "open": (9,  0),  "close": (17, 30), "label": "CET/CEST"},
    "^HSI":   {"tz": "Asia/Hong_Kong",    "open": (9,  30), "close": (16, 0),  "label": "HKT"},
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
        ex_cfg = EU_EXCHANGES[exchange]
        # Some tickers in a European exchange config trade on US markets (e.g. ARGX on Nasdaq)
        # — they must NOT get the local suffix appended.
        if display in ex_cfg.get("us_exceptions", []):
            return display
        return display + ex_cfg["suffix"]
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
        all_t = ex["tickers"] + ex.get("us_exceptions", [])
        all_n = ex["names"]   + ex.get("us_exception_names", [])
        for t, n in zip(all_t, all_n):
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


def resolve_market_for_ticker(full_ticker: str) -> tuple[str, str | None, str, str]:
    """
    Given a full ticker (e.g. 'AAPL', 'RELIANCE.NS', 'ASML.AS', '7203.T'),
    return (market, exchange, display_ticker, company_name).
    Falls back to (USA, None, full_ticker, full_ticker) if not found.
    """
    # EU exchanges first — suffix lookup
    for ex_name, ex_cfg in EU_EXCHANGES.items():
        sfx = ex_cfg["suffix"]
        # Normal suffixed tickers
        all_t = ex_cfg["tickers"] + ex_cfg.get("us_exceptions", [])
        all_n = ex_cfg["names"]   + ex_cfg.get("us_exception_names", [])
        for t, n in zip(all_t, all_n):
            resolved = t if t in ex_cfg.get("us_exceptions", []) else t + sfx
            if full_ticker.upper() == resolved.upper():
                return "🇪🇺 Europe", ex_name, t, n
    # Non-EU MARKET_CONFIG markets
    for mkt_name, cfg in MARKET_CONFIG.items():
        sfx      = cfg.get("suffix", "")
        all_t    = cfg["tickers"] + cfg.get("us_adrs", [])
        all_n    = cfg["names"]   + cfg.get("adr_names", [])
        for t, n in zip(all_t, all_n):
            resolved = t + sfx if sfx else t
            if full_ticker.upper() == resolved.upper():
                return mkt_name, None, t, n
    # Fallback — treat as US
    display = full_ticker.split(".")[0]
    return "🇺🇸 USA", None, display, display


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
        # Clear signal cache so home page shows fresh updated_at timestamps
        # after the user has run an analysis and closed the tab.
        load_top_signals.clear()
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


@st.cache_data(ttl=3600, show_spinner=False)
def load_index_data(ticker: str) -> pd.DataFrame:
    """
    Download 35 days of closing prices for a world index.

    TTL is 1 hour (3600s) — indices don't need more frequent updates on the
    homepage, and the longer cache dramatically reduces how often we hit Yahoo.

    Retries up to 3 times with exponential back-off (1s, 2s, 4s) to handle
    Yahoo Finance rate-limit bursts (YFRateLimitError) that occur when all 6
    world-index cards load simultaneously on a cold start.  On permanent
    failure we return an empty DataFrame so the caller's except-branch can
    show a graceful "Data unavailable" card instead of a traceback.
    """
    import time
    import random
    import yfinance as yf

    end   = datetime.today()
    start = end - timedelta(days=35)
    df    = pd.DataFrame()

    for attempt in range(3):
        try:
            df = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
            )
            if not df.empty:
                break          # success — stop retrying
        except Exception:
            pass               # rate-limit or network error — wait then retry

        # Exponential back-off: 1s, 2s, 4s  +  small random jitter
        if attempt < 2:
            time.sleep(2 ** attempt + random.uniform(0.1, 0.9))

    if df.empty:
        return pd.DataFrame()  # caller handles this gracefully

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


@st.cache_data(ttl=1800, show_spinner=False)
def load_top_signals(n: int = 10) -> pd.DataFrame:
    """
    Fetch the top-N highest-confidence pre-computed signals from Supabase.
    TTL = 30 min — fresh enough for a home page preview, avoids hammering
    Supabase on every rerun.  Returns empty DataFrame if Supabase is not
    configured or the table has no retrained rows yet.
    """
    try:
        from src.feedback_logger import FeedbackLogger
        return FeedbackLogger().get_top_signals(n=n)
    except Exception:
        return pd.DataFrame()


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
        "model_info": pipe.model_info() if hasattr(pipe, "model_info") else {},
    }


# ══════════════════════════════════════════════════════════════════════════════
# NAV BAR + TAB PILLS
# ══════════════════════════════════════════════════════════════════════════════

# Nav bar — pure st.markdown (no components.html; Streamlit Cloud safe)
import datetime as _dt
_now_utc   = _dt.datetime.now(_dt.timezone.utc)
_days_nav  = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
_mons_nav  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
_clock_str = (f"{_days_nav[_now_utc.weekday()]} {_now_utc.day} "
              f"{_mons_nav[_now_utc.month-1]}  {_now_utc.strftime('%H:%M')} UTC")

_nav_left, _nav_right = st.columns([9, 1])
with _nav_left:
    st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
            padding:0.55rem 0.25rem;border-bottom:1px solid #1a2035;
            font-family:'Inter',system-ui,sans-serif;margin-bottom:0.5rem;">
  <span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:1rem;font-weight:600;
                 color:#3b82f6;letter-spacing:-0.01em;">📡 Sentiment Signal</span>
    <span style="font-size:0.73rem;color:#475569;margin-left:0.6rem;">
      FinBERT-powered global trading signals</span>
  </span>
  <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;
               color:#64748b;letter-spacing:0.02em;">🕐 {_clock_str}</span>
</div>
""", unsafe_allow_html=True)

with _nav_right:
    st.markdown("""<style>
/* SAGE button in nav — always visible top-right */
div[data-testid="stHorizontalBlock"]:first-of-type > div:last-child button {
    background: #1e3a6e !important;
    border: 1px solid rgba(56,189,248,0.5) !important;
    color: #38bdf8 !important;
    font-family: monospace !important;
    font-weight: 800 !important;
    font-size: 0.78rem !important;
    letter-spacing: 1.5px !important;
    border-radius: 8px !important;
    padding: 0.3rem 0.7rem !important;
    margin-top: 0.2rem !important;
}
div[data-testid="stHorizontalBlock"]:first-of-type > div:last-child button:hover {
    background: #2a52a0 !important;
    border-color: #38bdf8 !important;
}
</style>""", unsafe_allow_html=True)
    _sage_label = "✕ SAGE" if st.session_state.sage_open else "SAGE ›"
    if st.button(_sage_label, key="sage_toggle_btn", help="Toggle SAGE Signal Assistant"):
        st.session_state.sage_open = not st.session_state.sage_open
        st.rerun()


# Tab pills row
pill_cols = st.columns([1] + [1] * len(st.session_state.open_tickers) + [4])

with pill_cols[0]:
    if st.button("🏠 Home", type="secondary" if st.session_state.active_tab != "home" else "primary",
                 width='stretch'):
        st.session_state.active_tab = "home"
        # Clear signal cache so the home page re-fetches updated_at from
        # Supabase — by the time the user clicks Home the pipeline has finished
        # and written the fresh timestamp, so the card now shows the new time.
        load_top_signals.clear()
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
                width='stretch',
            ):
                st.session_state.active_tab = tab["full"]
                st.rerun()
        with col_x:
            if st.button("✕", key=f"close_{tab['full']}", width='stretch'):
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
                ex_cfg  = EU_EXCHANGES[exchange]
                all_t   = ex_cfg["tickers"] + ex_cfg.get("us_exceptions", [])
                all_n   = ex_cfg["names"]   + ex_cfg.get("us_exception_names", [])
                ticker_opts = ["— select —"] + [
                    f"{t}  —  {n}" for t, n in zip(all_t, all_n)
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
            submitted = st.form_submit_button("Open →", type="primary", width='stretch')

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

    # ── Top 10 pre-computed signals ────────────────────────────────────────────
    st.markdown("### 🏆 Top signals today")
    st.caption(
        "Highest-confidence buy/sell signals from the latest batch retrain — "
        "across all 10 markets. Click any card to open a live analysis tab "
        "and recalculate the signal fresh."
    )

    top_df = load_top_signals(n=10)

    if top_df.empty:
        st.info(
            "No pre-computed signals available yet — the daily batch hasn't run today, "
            "or Supabase is not configured. Signals appear here after the 09:00 UTC batch."
        )
        # Surface any hidden error for debugging
        try:
            from src.feedback_logger import FeedbackLogger
            _test = FeedbackLogger().get_top_signals(n=1)
        except Exception as _e:
            st.caption(f"⚠️ Debug: {type(_e).__name__}: {_e}")
    else:
        import pytz
        now_utc = datetime.now(pytz.utc)

        # Pre-compute all card data first — avoids any column context issues later
        cards = []
        for _, sig in top_df.iterrows():
            ticker     = sig["ticker"]
            predicted  = int(sig["predicted"])
            confidence = float(sig["confidence"])
            sentiment  = float(sig["sentiment"])
            updated_at = sig["updated_at"]
            trade_date = sig["trade_date"]

            mkt, exch, display_t, company = resolve_market_for_ticker(ticker)

            is_buy     = predicted == 1
            sig_label  = "BUY"  if is_buy else "SELL"
            sig_color  = "#22c55e" if is_buy else "#ef4444"
            sig_bg     = "rgba(34,197,94,0.07)" if is_buy else "rgba(239,68,68,0.07)"
            sig_border = "#16a34a" if is_buy else "#dc2626"
            sig_icon   = "▲" if is_buy else "▼"
            sent_color = "#22c55e" if sentiment > 0.05 else ("#ef4444" if sentiment < -0.05 else "#94a3b8")

            flag = mkt.split(" ")[0] if mkt else "🌐"
            if exch:
                flag = exch.split(" ")[0]

            # Per-card age — each ticker has its own updated_at
            age_secs  = (now_utc - updated_at).total_seconds()
            if age_secs < 3600:
                age_label = f"{int(age_secs // 60)}m ago"
            elif age_secs < 86400:
                age_label = f"{int(age_secs // 3600)}h {int((age_secs % 3600) // 60)}m ago"
            else:
                age_label = updated_at.strftime("%d %b %H:%M UTC")

            # Per-card trade date label
            td_label = trade_date.strftime("%d %b %Y")

            cards.append({
                "ticker": ticker, "display_t": display_t, "company": company,
                "mkt": mkt, "exch": exch, "flag": flag,
                "sig_label": sig_label, "sig_color": sig_color, "sig_bg": sig_bg,
                "sig_border": sig_border, "sig_icon": sig_icon,
                "confidence": confidence, "sentiment": sentiment,
                "sent_color": sent_color, "age_label": age_label, "td_label": td_label,
            })

        # Render cards in a CSS grid via injected styles on st.columns.
        # We use a unique wrapper class per row and override Streamlit's
        # flex layout with CSS grid so the browser handles responsiveness.
        # Buttons sit directly under their card inside the same st.container,
        # which avoids the markdown flush bug.

        # Inject global CSS once — targets our wrapper divs by data attribute
        st.markdown("""
<style>
/* Signal card containers */
[data-sig-card] {
    background: var(--sig-bg);
    border: 1px solid rgba(255,255,255,0.07);
    border-top: 3px solid var(--sig-border);
    border-radius: 12px 12px 0 0;
    padding: 0.8rem 0.6rem 0.6rem;
    text-align: center;
    margin-bottom: 0;
}
/* Make the Open button flush under the card */
[data-sig-btn] > div > button {
    border-radius: 0 0 10px 10px !important;
    border-top: none !important;
    margin-top: -1px !important;
    font-size: 0.72rem !important;
    padding: 0.2rem 0.5rem !important;
}
</style>
        """, unsafe_allow_html=True)

        # ── Expand/collapse state ─────────────────────────────────────────────
        if "signals_expanded" not in st.session_state:
            st.session_state.signals_expanded = False

        # Always show top 2 cards (highest confidence — already sorted desc).
        # Remaining 8 are behind an expand toggle.
        cards_top  = cards[:2]
        cards_rest = cards[2:]

        def _render_card_row(card_pair):
            """Render one row of 2 cards with their Open buttons."""
            cols = st.columns(2)
            for col, card in zip(cols, card_pair):
                with col:
                    with st.container():
                        st.markdown(f"""
<div style="background:{card['sig_bg']};
     border:1px solid {card['sig_border']}44;
     border-top:3px solid {card['sig_border']};
     border-radius:12px 12px 0 0;
     padding:0.8rem 0.6rem 0.5rem;
     text-align:center;">
  <div style="font-size:0.85rem">{card['flag']}</div>
  <div style="font-family:monospace;font-size:0.9rem;font-weight:600;color:#f1f5f9">{card['display_t']}</div>
  <div style="font-size:0.65rem;color:#64748b;margin-bottom:0.35rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{card['company'] if card['company'] != card['display_t'] else '&nbsp;'}</div>
  <div style="font-size:1.1rem;font-weight:700;color:{card['sig_color']}">{card['sig_icon']} {card['sig_label']}</div>
  <div style="font-size:0.73rem;color:#94a3b8;margin-top:0.15rem">{card['confidence']:.0%} confidence</div>
  <div style="font-size:0.69rem;color:{card['sent_color']};margin-top:0.1rem">sentiment {card['sentiment']:+.3f}</div>
  <div style="font-size:0.61rem;color:#475569;margin-top:0.35rem;border-top:1px solid #1a2035;padding-top:0.28rem">📅 {card['td_label']} · ⏱ {card['age_label']}</div>
</div>""", unsafe_allow_html=True)
                        if st.button(
                            f"Open {card['display_t']} →",
                            key=f"top_sig_{card['ticker']}",
                            width='stretch',
                            type="secondary",
                        ):
                            load_top_signals.clear()
                            add_ticker_tab(
                                card["ticker"], card["display_t"],
                                card["company"], card["mkt"], card["exch"],
                            )
            # Pad with empty column if odd card in last row
            if len(card_pair) == 1:
                with cols[1]:
                    st.empty()

        # ── Render top 2 always ────────────────────────────────────────────────
        _render_card_row(cards_top)

        # ── Expand toggle (only shown if there are more cards) ─────────────────
        if cards_rest:
            remaining = len(cards_rest)
            toggle_label = (
                f"▲ Show less"
                if st.session_state.signals_expanded
                else f"▼ Show {remaining} more signals"
            )
            # Centre the toggle button with narrow columns
            _, btn_col, _ = st.columns([2, 3, 2])
            with btn_col:
                if st.button(
                    toggle_label,
                    key="signals_toggle",
                    width='stretch',
                    type="secondary",
                ):
                    st.session_state.signals_expanded = not st.session_state.signals_expanded
                    st.rerun()

        # ── Render remaining cards if expanded ─────────────────────────────────
        if st.session_state.signals_expanded and cards_rest:
            for i in range(0, len(cards_rest), 2):
                _render_card_row(cards_rest[i:i+2])

    st.divider()

    # ── World indices ──────────────────────────────────────────────────────────
    st.markdown("### 🌍 Global markets")

    # Two rows of 3 for 6 indices — cleaner than one cramped row of 6
    idx_rows = [WORLD_INDICES[:3], WORLD_INDICES[3:]]
    for idx_row in idx_rows:
        idx_cols = st.columns(3)
        for col, idx in zip(idx_cols, idx_row):
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
                st.plotly_chart(fig_spark, width='stretch', config={"displayModeBar": False})

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
        do_search = st.button("Search", width='stretch')

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
            force=True,   # dashboard recalc → always refresh updated_at
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
        st.plotly_chart(fig, width='stretch')
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
        st.plotly_chart(fig2, width='stretch')
        st.caption(f"Currency: **{currency_name}** · Transaction cost: 10bps · Starting: {currency_sym}10,000")

    with atab3:
        import plotly.express as px

        # Show which sentiment model is active
        try:
            from src.sentiment_model import FINBERT_MODEL_PRIMARY, FINBERT_MODEL_FALLBACK
            info = data.get("model_info", {})
            active_model = info.get("model_name", FINBERT_MODEL_PRIMARY)
            is_fallback  = active_model == FINBERT_MODEL_FALLBACK
            if is_fallback:
                st.warning(
                    f"⚠️ **Fallback model active** — using `{FINBERT_MODEL_FALLBACK}` (~110MB) "
                    f"instead of the primary FinBERT model due to memory constraints. "
                    f"Sentiment accuracy may be slightly lower."
                )
            else:
                st.success(f"✅ Primary model active — `{active_model}`")
        except Exception:
            pass

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
                st.plotly_chart(fig3, width='stretch')
        with col_b:
            st.markdown("#### Walk-forward results")
            res = data["results"]
            st.dataframe(pd.DataFrame([
                {"Model": n, "Accuracy": f"{s['accuracy_mean']:.3f}±{s['accuracy_std']:.3f}",
                 "F1": f"{s['f1_mean']:.3f}±{s['f1_std']:.3f}"}
                for n, s in res["summary"].items()
            ]), hide_index=True, width='stretch')
            st.markdown("#### Fold detail")
            st.dataframe(pd.DataFrame([
                {"Model": n, "Fold": f["fold"], "Acc": round(f["accuracy"],3),
                 "F1": round(f["f1"],3), "Train": f["train_rows"], "Test": f["test_rows"]}
                for n, folds in res["fold_details"].items() for f in folds
            ]), hide_index=True, width='stretch')

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
                st.plotly_chart(fig_acc, width='stretch')

                # Prediction history table
                st.markdown("#### Recent predictions")
                display_perf = perf_df[["trade_date","predicted","actual","correct","confidence","sentiment"]].copy()
                display_perf["trade_date"] = display_perf["trade_date"].dt.strftime("%Y-%m-%d")
                display_perf["predicted"]  = display_perf["predicted"].map({1:"▲ UP", -1:"▼ DOWN"})
                display_perf["actual"]     = display_perf["actual"].map({1:"▲ UP", -1:"▼ DOWN"})
                display_perf["correct"]    = display_perf["correct"].map({True:"✓", False:"✗"})
                display_perf["confidence"] = display_perf["confidence"].apply(lambda x: f"{x:.1%}")
                display_perf["sentiment"]  = display_perf["sentiment"].apply(lambda x: f"{x:+.3f}")
                st.dataframe(display_perf.head(20), hide_index=True, width='stretch')

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


# ══════════════════════════════════════════════════════════════════════════════
# SAGE — Persistent right-side chat assistant
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# SAGE — Persistent right-side chat assistant
# ══════════════════════════════════════════════════════════════════════════════























# ── Desktop sidebar CSS (only injected on desktop) ────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: #080f1e !important;
    border-left: 1px solid rgba(56,189,248,0.18) !important;
    border-right: none !important;
}
[data-testid="stSidebar"] > div:first-child {
    background: #080f1e !important;
    padding-top: 0 !important;
    padding-left: 0.6rem !important; padding-right: 0.6rem !important;
    overflow-y: auto !important;
}
[data-testid="stSidebar"] [data-testid="stChatInput"] textarea {
    background: #0c1824 !important;
    border: 1px solid rgba(56,189,248,0.25) !important;
    color: #f1f5f9 !important; font-size: 0.8rem !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] [data-testid="stChatInput"] button {
    background: rgba(56,189,248,0.15) !important;
    border: 1px solid rgba(56,189,248,0.3) !important;
    color: #38bdf8 !important;
}
[data-testid="stSidebar"] .stButton > button { font-size: 0.75rem !important; }
[data-testid="stSidebar"] [data-testid="chatAvatarIcon-user"],
[data-testid="stSidebar"] [data-testid="chatAvatarIcon-assistant"],
[data-testid="stSidebar"] [class*="avatarIcon"],
[data-testid="stSidebar"] [class*="Avatar"] { display: none !important; }
</style>
""", unsafe_allow_html=True)



















































































if "sage_msgs"         not in st.session_state: st.session_state.sage_msgs         = []
if "sage_pending"      not in st.session_state: st.session_state.sage_pending      = None
if "sage_tickers"      not in st.session_state: st.session_state.sage_tickers      = {}
if "sage_pick"         not in st.session_state: st.session_state.sage_pick         = None
if "sage_flow"         not in st.session_state: st.session_state.sage_flow         = None
# sage_open already initialised before set_page_config above


# ── Universal ticker resolution — works for ANY yfinance symbol ───────────────
# _SK is still built for general-question context (top signals list), but
# SAGE ticker lookup no longer requires a symbol to be in _SK.
_SK: dict = {}
for _m, _cfg in MARKET_CONFIG.items():
    _sfx = _cfg.get("suffix", "")
    for _t, _n in zip(_cfg["tickers"] + _cfg.get("us_adrs", []),
                      _cfg["names"]   + _cfg.get("adr_names", [])):
        _f = _t + _sfx if _sfx and not _t.endswith(_sfx) else _t
        _SK[_f.upper()] = (_m, None, _t, _n)
for _ex, _ecfg in EU_EXCHANGES.items():
    _sfx = _ecfg["suffix"]
    for _t, _n in zip(_ecfg["tickers"] + _ecfg.get("us_exceptions", []),
                      _ecfg["names"]   + _ecfg.get("us_exception_names", [])):
        _f = _t if _t in _ecfg.get("us_exceptions", []) else _t + _sfx
        _SK[_f.upper()] = ("🇪🇺 Europe", _ex, _t, _n)

# Suffix map for wizard — market/exchange → yfinance suffix
_MKT_SUFFIX: dict = {m: cfg.get("suffix","") for m, cfg in MARKET_CONFIG.items()}
_MKT_SUFFIX["🇪🇺 Europe"] = ""  # exchange-level suffix applied separately
_EX_SUFFIX: dict  = {ex: cfg.get("suffix","") for ex, cfg in EU_EXCHANGES.items()}

def _build_yf_symbol(ticker: str, market: str, exchange: str | None) -> str:
    """Build the correct yfinance symbol for any ticker given market/exchange."""
    t = ticker.upper().strip()
    if exchange:
        sfx = _EX_SUFFIX.get(exchange, "")
    else:
        sfx = _MKT_SUFFIX.get(market, "")
    # Don't double-add suffix
    if sfx and not t.endswith(sfx):
        return t + sfx
    return t

def _sk_find(msg: str) -> list:
    """Find tickers from _SK mentioned in a message (used for general questions only)."""
    words = [w.strip(".,!?()[]'\"") for w in msg.upper().split()]
    found = []
    for w in words:
        if w in _SK and w not in found:
            found.append(w)
    if not found:
        root_map: dict = {}
        for key in _SK:
            root_map.setdefault(key.split(".")[0], []).append(key)
        for w in words:
            if w in root_map:
                for k in root_map[w]:
                    if k not in found: found.append(k)
    return found

def _sig_from_df(ft, df):
    if df is None or df.empty: return None
    m = df[df["ticker"].str.upper() == ft.upper()]
    if m.empty: return None
    r = m.iloc[0]
    return {"predicted": int(r["predicted"]), "confidence": float(r["confidence"]),
            "sentiment": float(r["sentiment"]), "updated_at": r["updated_at"]}

def _live_sig_universal(yf_symbol: str, market: str, exchange: str | None):
    """
    Fetch a FRESH signal for ANY ticker using:
      1. FinBERT news sentiment pipeline (same as main app)
      2. yfinance price-momentum fallback if FinBERT unavailable
    Always runs fresh — no cached table lookup.
    """
    # ── Attempt 1: Full FinBERT pipeline (same path as main app) ──────────────
    try:
        from src.data_ingestion      import DataIngestion
        from src.sentiment_model     import SentimentModel
        from src.feature_engineering import FeatureEngineer
        import pytz as _ptz_u
        # Strip suffix for news fetch (e.g. ABN from ABN.AS)
        bare = yf_symbol.split(".")[0]
        arts = DataIngestion().fetch_news(bare, market=market, exchange=exchange)
        sc   = SentimentModel().score_articles(arts) if arts else []
        fe   = FeatureEngineer().build_features(sc)  if sc   else None
        if fe is not None and not fe.empty:
            s = float(fe.iloc[-1].get("mean_score", 0))
            return {
                "predicted":  1 if s > 0 else -1,
                "confidence": min(abs(s) * 2 + 0.5, 0.99),
                "sentiment":  s,
                "updated_at": datetime.now(_ptz_u.utc),
                "live": True, "source": "finbert",
                "articles": len(arts)
            }
    except Exception:
        pass

    # ── Attempt 2: yfinance price-momentum fallback ────────────────────────────
    try:
        import yfinance as _yf
        import datetime as _dt2
        _hist = _yf.Ticker(yf_symbol).history(period="15d", interval="1d")
        if _hist is None or _hist.empty:
            return None
        _closes = _hist["Close"].dropna()
        if len(_closes) < 3:
            return None
        _ret5 = float((_closes.iloc[-1] - _closes.iloc[0])  / _closes.iloc[0])
        _ret1 = float((_closes.iloc[-1] - _closes.iloc[-2]) / _closes.iloc[-2])
        _s    = max(-1.0, min(1.0, (_ret5 * 0.6 + _ret1 * 0.4) * 10))
        _conf = min(0.5 + abs(_s) * 0.35, 0.88)
        # Also grab company info for display name
        try:
            _info = _yf.Ticker(yf_symbol).info
            _name = _info.get("shortName") or _info.get("longName") or yf_symbol
        except Exception:
            _name = yf_symbol
        return {
            "predicted":  1 if _s >= 0 else -1,
            "confidence": _conf,
            "sentiment":  round(_s, 4),
            "updated_at": _dt2.datetime.now(_dt2.timezone.utc),
            "live": True, "source": "price", "name": _name
        }
    except Exception:
        return None

def _synth_answer(msgs, ctx):
    """Data-driven answer built directly from signal/sentiment context."""
    if not ctx or ctx == "No data.":
        return "No live data right now. Type a ticker to get its signal."

    lines  = {l.split(":")[0]: l for l in ctx.split("\n") if ":" in l}
    q      = (msgs[-1]["content"] if msgs else "").lower()

    # Parse signals line
    sigs, buys, sells = [], [], []
    if "Signals" in lines:
        sigs  = lines["Signals"].replace("Signals: ", "").split(", ")
        buys  = [s for s in sigs if "BUY"  in s]
        sells = [s for s in sigs if "SELL" in s]

    # Parse sentiment line
    bearish_mkts, bullish_mkts, neutral_mkts = [], [], []
    if "Sentiment" in lines:
        for entry in lines["Sentiment"].replace("Sentiment: ", "").split(" | "):
            if "bearish" in entry.lower(): bearish_mkts.append(entry.split(":")[0].strip())
            elif "bullish" in entry.lower(): bullish_mkts.append(entry.split(":")[0].strip())
            else: neutral_mkts.append(entry.split(":")[0].strip())

    # Route by intent — mutually exclusive
    if any(w in q for w in ["bearish", "bear", "which market"]):
        parts = [f"Bearish markets: {', '.join(bearish_mkts) if bearish_mkts else 'none right now'}."]
        if bullish_mkts: parts.append(f"Bullish: {', '.join(bullish_mkts)}.")
        if neutral_mkts: parts.append(f"Neutral: {', '.join(neutral_mkts)}.")

    elif any(w in q for w in ["bullish", "bull"]):
        parts = [f"Bullish markets: {', '.join(bullish_mkts) if bullish_mkts else 'none right now'}."]
        if bearish_mkts: parts.append(f"Bearish: {', '.join(bearish_mkts)}.")

    elif any(w in q for w in ["sell", "short"]):
        parts = [f"Top SELL signals: {', '.join(sells[:5]) if sells else 'none in top signals'}."]

    elif any(w in q for w in ["buy", "long"]):
        parts = [f"Top BUY signals: {', '.join(buys[:5]) if buys else 'none in top signals'}."]

    elif any(w in q for w in ["strongest", "best", "top signal", "highest"]):
        parts = []
        if sigs:  parts.append(f"Strongest: {sigs[0]}.")
        if buys:  parts.append(f"Top BUYs: {', '.join(buys[:3])}.")
        if sells: parts.append(f"Top SELLs: {', '.join(sells[:3])}.")

    elif any(w in q for w in ["sentiment", "market", "feel", "overall"]):
        parts = []
        if bearish_mkts: parts.append(f"Bearish: {', '.join(bearish_mkts)}.")
        if bullish_mkts: parts.append(f"Bullish: {', '.join(bullish_mkts)}.")
        if neutral_mkts: parts.append(f"Neutral: {', '.join(neutral_mkts)}.")

    else:
        # Generic fallback
        parts = []
        if buys:  parts.append(f"Top BUYs: {', '.join(buys[:3])}.")
        if sells: parts.append(f"Top SELLs: {', '.join(sells[:3])}.")

    return "\n".join(parts) + "\n\n_Data-driven — AI model offline._" if parts else ctx

def _hf_call(msgs, ctx):
    """HuggingFace LLM call with data-driven fallback."""
    try:
        import requests as _rq, os as _os
        try:    tok = st.secrets["huggingface"]["token"]
        except Exception: tok = _os.getenv("HUGGINGFACE_TOKEN", "")
        if not tok:
            return _synth_answer(msgs, ctx)
        sys_p = ("You are Sage, a concise financial assistant for Sentiment Signal. "
                 "Answer in 2-3 sentences. Never give direct investment advice.\n\n"
                 f"Context:\n{ctx}")
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{sys_p}<|eot_id|>"
        for m in msgs[-6:]:
            r = "user" if m["role"] == "user" else "assistant"
            c = m["content"] if "<div" not in m["content"] else "[signal shown]"
            prompt += f"<|start_header_id|>{r}<|end_header_id|>\n{c}<|eot_id|>"
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
        resp = _rq.post(
            "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct",
            headers={"Authorization": f"Bearer {tok}"},
            json={"inputs": prompt, "parameters": {
                "max_new_tokens": 160, "temperature": 0.4,
                "return_full_text": False,
                "stop": ["<|eot_id|>", "<|start_header_id|>"]}},
            timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            txt = (data[0].get("generated_text", "") if isinstance(data, list) and data else "").strip()
            for s in ["<|eot_id|>", "<|start_header_id|>"]:
                txt = txt[:txt.index(s)].strip() if s in txt else txt
            return txt or _synth_answer(msgs, ctx)
        else:
            return _synth_answer(msgs, ctx)
    except Exception:
        return _synth_answer(msgs, ctx)

# ── Helper: fetch signal for a resolved full ticker key ───────────────────────
def _resolve_and_fetch(ft_key):
    """Resolve a tracked _SK key and fetch fresh signal."""
    _mkt, _exch, _dt, _co = _SK.get(ft_key, (None, None, ft_key, ft_key))
    _yfsym = _build_yf_symbol(_dt, _mkt or "", _exch)
    return _fetch_signal_for(_yfsym, _dt, _co, _mkt or "", _exch)

def _fetch_signal_for(yf_symbol: str, display_ticker: str, company: str,
                      market: str, exchange):
    """
    Fetch a FRESH signal for ANY yfinance symbol — no tracked list required.
    Always runs live analysis (FinBERT → price fallback).
    """
    import pytz as _ptz3
    with st.spinner(f"Running fresh analysis for {display_ticker}…"):
        _sig = _live_sig_universal(yf_symbol, market, exchange)

    if _sig:
        _is_up = _sig["predicted"] == 1
        _s     = _sig["sentiment"]
        _sl    = "bullish" if _s > 0.05 else ("bearish" if _s < -0.05 else "neutral")
        try:    _sc = (datetime.now(_ptz3.utc) - _sig["updated_at"]).total_seconds()
        except: _sc = 0
        _age   = f"{int(_sc//60)}m ago" if _sc < 3600 else f"{int(_sc//3600)}h {int((_sc%3600)//60)}m ago"
        _exlb  = f" ({exchange})" if exchange else f" ({market})" if market else ""
        _src   = {"finbert": " · FinBERT live", "price": " · price momentum"}.get(
                  _sig.get("source", ""), "")
        _arts  = f" · {_sig['articles']} articles" if _sig.get("articles") else ""
        _name  = _sig.get("name") or company or display_ticker
        _rep   = (f"{display_ticker}{_exlb} — {_name}\n\n"
                  f"{'🟢 BUY' if _is_up else '🔴 SELL'} · {_sig['confidence']:.0%} confidence\n\n"
                  f"Sentiment: {_sl} ({_s:+.3f}){_arts} · ⏱ {_age}{_src}")
        _next  = len(st.session_state.sage_msgs) + 1
        return _rep, (market, exchange, display_ticker, _name, _next), yf_symbol
    else:
        _exlb = f" on {exchange}" if exchange else f" ({market})" if market else ""
        return (f"Couldn't fetch data for {display_ticker}{_exlb}. "
                f"Check the symbol and try again.", None, None)

# ── Markets and exchanges available ───────────────────────────────────────────
_ALL_MARKETS   = list(MARKET_CONFIG.keys()) + (["🇪🇺 Europe"] if "🇪🇺 Europe" not in MARKET_CONFIG else [])
_EU_MKT        = "🇪🇺 Europe"
_ALL_EXCHANGES = list(EU_EXCHANGES.keys())


# ── Process pending free-text message ─────────────────────────────────────────
if st.session_state.sage_pending:
    _pend = st.session_state.sage_pending
    st.session_state.sage_pending = None

    _GENERAL_WORDS = {"strongest","signal","signals","market","markets","bearish",
                      "bullish","sell","buy","confidence","what","which","top",
                      "high","low","best","worst","today","now","right","explain",
                      "tell","show","give","list","any","all","current","overall"}
    _pend_words   = set(_pend.lower().replace("?","").replace("-"," ").split())
    _is_general_q = len(_pend_words & _GENERAL_WORDS) >= 2 or len(_pend.split()) > 4

    _hits = _sk_find(_pend)

    if _hits and len(_hits) == 1:
        # Unambiguous ticker match — answer directly
        st.session_state.sage_msgs.append({"role": "user", "content": _pend})
        _rep, _ttup, _tkey = _resolve_and_fetch(_hits[0])
        st.session_state.sage_msgs.append({"role": "assistant", "content": _rep})
        if _ttup and _tkey:
            st.session_state.sage_tickers[_tkey] = _ttup

    elif _is_general_q:
        # General question — answer from data context
        _top  = load_top_signals(n=20)
        _sent = load_market_sentiment()
        _ctx  = []
        if _top is not None and not _top.empty:
            _ctx.append("Signals: " + ", ".join(
                f"{r['ticker']} {'BUY' if int(r['predicted'])==1 else 'SELL'} {float(r['confidence']):.0%}"
                for _, r in _top.head(10).iterrows()))
        if _sent:
            _ctx.append("Sentiment: " + " | ".join(
                f"{mk}: {d.get('label','?')} ({d.get('score',0):+.3f})"
                for mk, d in _sent.items()))
        _rep = _hf_call(st.session_state.sage_msgs + [{"role":"user","content":_pend}],
                        "\n".join(_ctx) or "No data.")
        st.session_state.sage_msgs.append({"role": "user",      "content": _pend})
        st.session_state.sage_msgs.append({"role": "assistant", "content": _rep})

    else:
        # Short unknown word — assume it's a ticker — ask which market
        st.session_state.sage_msgs.append({"role": "user", "content": _pend})
        st.session_state.sage_msgs.append({
            "role": "assistant",
            "content": f"Which market is **{_pend.upper()}** traded on?",
            "wizard": "market"
        })
        st.session_state.sage_flow = {"step": "market", "ticker": _pend.upper()}



# ── SAGE icon ──────────────────────────────────────────────────────────────────
_SAGE_URI = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0MCA0MCI+CiAgPHJlY3Qgd2lkdGg9IjQwIiBoZWlnaHQ9IjQwIiByeD0iMTAiIGZpbGw9IiMyNTYzZWIiLz4KICA8cmVjdCB4PSIxMCIgeT0iMTAiIHdpZHRoPSIyMCIgaGVpZ2h0PSIzIiByeD0iMS41IiBmaWxsPSJ3aGl0ZSIvPgogIDxyZWN0IHg9IjEwIiB5PSIxOC41IiB3aWR0aD0iMjAiIGhlaWdodD0iMyIgcng9IjEuNSIgZmlsbD0id2hpdGUiLz4KICA8cmVjdCB4PSIxMCIgeT0iMjciIHdpZHRoPSIyMCIgaGVpZ2h0PSIzIiByeD0iMS41IiBmaWxsPSJ3aGl0ZSIvPgogIDxyZWN0IHg9IjEwIiB5PSIxMCIgd2lkdGg9IjMiIGhlaWdodD0iMTEuNSIgcng9IjEuNSIgZmlsbD0id2hpdGUiLz4KICA8cmVjdCB4PSIyNyIgeT0iMTguNSIgd2lkdGg9IjMiIGhlaWdodD0iMTEuNSIgcng9IjEuNSIgZmlsbD0id2hpdGUiLz4KPC9zdmc+"

# Migrate old 4-tuple ticker entries
for _k in list(st.session_state.sage_tickers.keys()):
    _v = st.session_state.sage_tickers[_k]
    if len(_v) == 4:
        st.session_state.sage_tickers[_k] = (_v[0], _v[1], _v[2], _v[3], -1)

# ── Shared SAGE UI — called from sidebar (desktop) or dialog (mobile) ──────────
def _render_sage_panel():
    """Render the full SAGE chat UI. Works inside st.sidebar or st.dialog."""
    # Header
    st.markdown(f"""
<div style="display:flex;align-items:center;gap:0.65rem;
            padding:0.9rem 0.6rem 0.8rem;margin:-0.5rem -0.6rem 0.7rem;
            border-bottom:1px solid rgba(56,189,248,0.18);
            background:#0f2044;">
  <img src="{_SAGE_URI}" width="36" height="36"
       style="border-radius:9px;flex-shrink:0;display:block;"/>
  <div>
    <div style="font-size:0.9rem;font-weight:800;color:#f1f5f9;
                font-family:monospace;letter-spacing:2.5px;line-height:1.2;">SAGE</div>
    <div style="font-size:0.63rem;color:#38bdf8;margin-top:1px;">Signal Assistant</div>
  </div>
</div>""", unsafe_allow_html=True)

    # Quick-start buttons when empty
    if not st.session_state.sage_msgs and not st.session_state.sage_flow:
        st.caption("Quick questions:")
        for _sg in ["What's the strongest signal?",
                    "Which markets are bearish?",
                    "Top BUY signals right now",
                    "High-confidence SELL signals?"]:
            if st.button(_sg, key=f"sg_{abs(hash(_sg))}", width='stretch', type="secondary"):
                st.session_state.sage_pending = _sg
                st.rerun()
        st.caption("Or type any ticker below ↓")

    # Messages + wizard buttons
    _ticker_by_msgidx = {v[4]: (k, v) for k, v in st.session_state.sage_tickers.items() if len(v) > 4}
    _recent = st.session_state.sage_msgs[-40:]

    for _mi, _msg in enumerate(_recent):
        _is_user = _msg["role"] == "user"
        _bg   = "rgba(30,41,59,0.8)"  if _is_user else "rgba(8,20,40,0.7)"
        _br   = "12px 12px 3px 12px"  if _is_user else "12px 12px 12px 3px"
        _ml   = "1.2rem" if _is_user else "0"
        _mr   = "0"      if _is_user else "1.2rem"
        _lbl  = "You"    if _is_user else "Sage"
        _lclr = "#475569" if _is_user else "#38bdf8"
        _content = _msg["content"].replace(chr(10), "<br>")
        st.markdown(f"""
<div style="margin-bottom:0.4rem;margin-left:{_ml};margin-right:{_mr}">
  <div style="font-size:0.6rem;color:{_lclr};margin-bottom:2px;
              text-align:{'right' if _is_user else 'left'};font-family:monospace;">{_lbl}</div>
  <div style="background:{_bg};border-radius:{_br};padding:0.45rem 0.65rem;
              font-size:0.79rem;line-height:1.6;color:#e2e8f0;
              border:1px solid rgba(255,255,255,0.07);">{_content}</div>
</div>""", unsafe_allow_html=True)

        # Wizard: market buttons
        if not _is_user and _msg.get("wizard") == "market" and _mi == len(_recent) - 1:
            _ticker_raw = (st.session_state.sage_flow or {}).get("ticker", "")
            for _mkt_opt in _ALL_MARKETS:
                _mflag = _mkt_opt.split(" ")[0]
                _mname = " ".join(_mkt_opt.split(" ")[1:])
                if st.button(f"{_mflag} {_mname}", key=f"wiz_mkt_{_mkt_opt}_{_mi}",
                             width='stretch', type="secondary"):
                    st.session_state.sage_msgs.append({"role": "user", "content": _mkt_opt})
                    st.session_state.sage_flow = {**(st.session_state.sage_flow or {}),
                                                  "market": _mkt_opt}
                    if _mkt_opt == _EU_MKT:
                        st.session_state.sage_msgs.append({
                            "role": "assistant",
                            "content": "Which European exchange?",
                            "wizard": "exchange"
                        })
                        st.session_state.sage_flow["step"] = "exchange"
                    else:
                        _yfsym2 = _build_yf_symbol(_ticker_raw, _mkt_opt, None)
                        _rep2, _ttup2, _tkey2 = _fetch_signal_for(
                            _yfsym2, _ticker_raw, "", _mkt_opt, None)
                        st.session_state.sage_msgs.append({"role": "assistant", "content": _rep2})
                        if _ttup2 and _tkey2:
                            st.session_state.sage_tickers[_tkey2] = _ttup2
                        st.session_state.sage_flow = None
                    st.rerun()

        # Wizard: exchange buttons (Europe only)
        elif not _is_user and _msg.get("wizard") == "exchange" and _mi == len(_recent) - 1:
            _ticker_raw = (st.session_state.sage_flow or {}).get("ticker", "")
            _mkt_sel    = (st.session_state.sage_flow or {}).get("market", _EU_MKT)
            _EX_FLAGS   = {"🇳🇱 Amsterdam":"🇳🇱","🇩🇪 Frankfurt":"🇩🇪","🇫🇷 Paris":"🇫🇷",
                           "🇬🇧 London":"🇬🇧","🇨🇭 Zurich":"🇨🇭","🇮🇹 Milan":"🇮🇹",
                           "🇪🇸 Madrid":"🇪🇸","🇵🇹 Lisbon":"🇵🇹","🇧🇪 Brussels":"🇧🇪"}
            for _ex_opt in _ALL_EXCHANGES:
                _eflag  = _EX_FLAGS.get(_ex_opt, "🇪🇺")
                _exname = _ex_opt.split(" ")[-1] if " " in _ex_opt else _ex_opt
                if st.button(f"{_eflag} {_exname}", key=f"wiz_ex_{_ex_opt}_{_mi}",
                             width='stretch', type="secondary"):
                    st.session_state.sage_msgs.append({"role": "user", "content": _ex_opt})
                    _yfsym3 = _build_yf_symbol(_ticker_raw, _mkt_sel, _ex_opt)
                    _rep2, _ttup2, _tkey2 = _fetch_signal_for(
                        _yfsym3, _ticker_raw, "", _mkt_sel, _ex_opt)
                    st.session_state.sage_msgs.append({"role": "assistant", "content": _rep2})
                    if _ttup2 and _tkey2:
                        st.session_state.sage_tickers[_tkey2] = _ttup2
                    st.session_state.sage_flow = None
                    st.rerun()

        # Open-in-analysis button after signal replies
        if not _is_user and "wizard" not in _msg:
            _abs_idx = len(st.session_state.sage_msgs) - len(_recent) + _mi
            if _abs_idx in _ticker_by_msgidx:
                _ft2, (_m2, _e2, _d2, _c2, _) = _ticker_by_msgidx[_abs_idx]
                if st.button(f"📊 Open {_d2} →", key=f"sop_{_ft2}_{_mi}",
                             width='stretch', type="primary"):
                    load_top_signals.clear()
                    add_ticker_tab(_ft2, _d2, _c2, _m2, _e2)

    # Clear + chat input
    if st.session_state.sage_msgs:
        st.divider()
        if st.button("🗑 Clear", key="sage_clr", type="secondary", width='stretch'):
            st.session_state.sage_msgs    = []
            st.session_state.sage_tickers = {}
            st.session_state.sage_pick    = None
            st.session_state.sage_flow    = None
            st.rerun()

    _inp = st.chat_input("Stock ticker or question…", key="sage_inp")
    if _inp:
        st.session_state.sage_pending = _inp
        st.rerun()


# ── Render SAGE in sidebar (collapsed by default) ─────────────────────────────
with st.sidebar:
    _render_sage_panel()
























































