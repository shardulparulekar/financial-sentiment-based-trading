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
    initial_sidebar_state="expanded",
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
        "tickers": ["ASML","PRX","INGA","HEIA","AGN","ASRNL","RAND"],
        "names":   ["ASML","Prosus","ING","Heineken","Aegon","ASR Nederland","Randstad"],
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

# Nav bar + live clock
# Nav bar — pure st.markdown (no components.html needed; clock shows time at last rerun)
import datetime as _dt
_now_utc = _dt.datetime.now(_dt.timezone.utc)
_days  = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
_mons  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
_clock_str = f"{_days[_now_utc.weekday()]} {_now_utc.day} {_mons[_now_utc.month-1]}  {_now_utc.strftime('%H:%M')} UTC"
st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
            padding:0.6rem 0.25rem;border-bottom:1px solid #1a2035;
            font-family:'Inter',system-ui,sans-serif;margin-bottom:0.5rem;">
  <span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:1rem;font-weight:600;
                 color:#3b82f6;letter-spacing:-0.01em;">📡 Sentiment Signal</span>
    <span style="font-size:0.73rem;color:#475569;margin-left:0.6rem;letter-spacing:0.01em;">
      FinBERT-powered global trading signals</span>
  </span>
  <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;
               color:#64748b;letter-spacing:0.02em;">🕐 {_clock_str}</span>
</div>
""", unsafe_allow_html=True)

# ── Global Sage assistant (fixed top-right, persists across all tabs) ─────────
if "sage_pending" not in st.session_state:
    st.session_state.sage_pending = None
if "sage_msgs"    not in st.session_state:
    st.session_state.sage_msgs    = []

# ── Handle sage_msg query param at top level (works on any tab) ──────────────
_sage_msg_qp = st.query_params.get("sage_msg")
if _sage_msg_qp:
    import urllib.parse as _up_qp
    st.session_state.sage_pending = _up_qp.unquote(_sage_msg_qp)
    st.query_params.clear()
    st.rerun()

if st.session_state.sage_pending:
    _pending = st.session_state.sage_pending
    st.session_state.sage_pending = None

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

    def _sk_find(msg):
        found = []
        for w in msg.upper().split():
            w = w.strip(".,!?()[]'\"")
            if w in _SK and w not in found:
                found.append(w)
        return found

    _found = _sk_find(_pending)
    _reply = ""
    if _found:
        _ft = _found[0]
        _mkt, _exch, _dt, _co = _SK.get(_ft, (None, None, _ft, _ft))
        _top = load_top_signals(n=20)
        _sig = None
        if _top is not None and not _top.empty:
            _match = _top[_top["ticker"].str.upper() == _ft.upper()]
            if not _match.empty:
                _r = _match.iloc[0]
                _sig = {"predicted": int(_r["predicted"]), "confidence": float(_r["confidence"]),
                        "sentiment": float(_r["sentiment"]), "updated_at": _r["updated_at"],
                        "trade_date": _r["trade_date"]}
        if not _sig and _mkt:
            try:
                from src.data_ingestion      import DataIngestion
                from src.sentiment_model     import SentimentModel
                from src.feature_engineering import FeatureEngineer
                import pytz as _ptz
                _arts = DataIngestion().fetch_news(_ft, market=_mkt, exchange=_exch)
                _sc   = SentimentModel().score_articles(_arts) if _arts else []
                _fe   = FeatureEngineer().build_features(_sc) if _sc else None
                if _fe is not None and not _fe.empty:
                    _sc2   = float(_fe.iloc[-1].get("mean_score", 0))
                    _now2  = datetime.now(_ptz.utc)
                    _sig   = {"predicted": 1 if _sc2 > 0 else -1,
                              "confidence": min(abs(_sc2)*2+0.5,0.99),
                              "sentiment": _sc2, "updated_at": _now2,
                              "trade_date": _now2, "live": True}
            except Exception:
                pass
        if _sig and _mkt:
            import pytz as _ptz2
            _now3  = datetime.now(_ptz2.utc)
            _is_up = _sig["predicted"] == 1
            _scol  = "#22c55e" if _is_up else "#ef4444"
            _slbl  = "bullish" if _sig["sentiment"] > 0.05 else ("bearish" if _sig["sentiment"] < -0.05 else "neutral")
            _secs  = (_now3 - _sig["updated_at"]).total_seconds()
            _age   = (f"{int(_secs//60)}m ago" if _secs < 3600
                      else f"{int(_secs//3600)}h {int((_secs%3600)//60)}m ago")
            _flag  = (_exch or _mkt or "").split(" ")[0]
            _live  = " · live" if _sig.get("live") else ""
            _reply = (
                f"<div style='border-top:2px solid {_scol};border-radius:8px;"
                f"padding:0.55rem 0.65rem;background:rgba(255,255,255,0.03)'>"
                f"<div style='font-size:0.78rem;color:#94a3b8'>{_flag} <b>{_dt}</b>"
                f" <span style='color:#475569'>{_co}</span></div>"
                f"<div style='font-size:1rem;font-weight:700;color:{_scol}'>"
                f"{'&#9650;' if _is_up else '&#9660;'} {'BUY' if _is_up else 'SELL'}</div>"
                f"<div style='font-size:0.72rem;color:#94a3b8;margin-top:0.1rem'>"
                f"{_sig['confidence']:.0%} conf · {_slbl} ({_sig['sentiment']:+.3f})</div>"
                f"<div style='font-size:0.65rem;color:#475569;margin-top:0.15rem'>"
                f"⏱ {_age}{_live}</div>"
                f"<a href='?sage_open={_ft}' style='display:inline-block;margin-top:0.35rem;"
                f"background:#1e3a5f;color:#60a5fa;font-size:0.72rem;"
                f"padding:0.2rem 0.55rem;border-radius:6px;text-decoration:none'>"
                f"📊 Open full analysis →</a></div>"
            )
        elif _mkt:
            _reply = f"Couldn't fetch a live signal for <b>{_dt}</b> right now."
        else:
            _reply = f"<b>{_ft}</b> isn't in our tracked list."
    else:
        try:
            import requests as _rq, os as _os
            # Read token — secrets must be [huggingface] / token = "hf_..."
            try:    _hf = st.secrets["huggingface"]["token"]
            except Exception: _hf = _os.getenv("HUGGINGFACE_TOKEN", "")

            if not _hf:
                _reply = "HuggingFace token not configured. Add [huggingface]\ntoken='hf_xxx' to Streamlit secrets."
            else:
                _top2  = load_top_signals(n=10)
                _sent2 = load_market_sentiment()
                _ctx   = []
                if _top2 is not None and not _top2.empty:
                    _ctx.append("Signals: " + ", ".join(
                        f"{r['ticker']}: {'BUY' if int(r['predicted'])==1 else 'SELL'} {float(r['confidence']):.0%}"
                        for _,r in _top2.head(5).iterrows()))
                if _sent2:
                    _ctx.append("Sentiment: " + " | ".join(
                        f"{mk}: {d.get('label','?')} ({d.get('score',0):+.3f})"
                        for mk,d in _sent2.items()))
                _system = ("You are Sage, a concise financial signal assistant. "
                           "Answer in 2-3 sentences. Never give direct investment advice. "
                           "Reply in plain text only, no HTML tags.\n\n"
                           f"Context:\n{chr(10).join(_ctx) or 'No data.'}")
                # Use HF chat completions endpoint (works with free tier as of 2025)
                # Model: Qwen2.5-7B-Instruct — fast, free, reliable on HF router
                _messages = [{"role": "system", "content": _system}]
                for _hm in st.session_state.sage_msgs[-6:]:
                    _messages.append({"role": _hm["role"], "content": _hm["content"]})
                _messages.append({"role": "user", "content": _pending})

                import concurrent.futures as _cf
                def _call_hf():
                    return _rq.post(
                        "https://router.huggingface.co/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {_hf}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": "Qwen/Qwen3-8B",
                            "messages": _messages,
                            "max_tokens": 200,
                            "temperature": 0.4,
                            "stream": False,
                        },
                        timeout=60)

                with st.spinner("Sage is thinking…"):
                    with _cf.ThreadPoolExecutor(max_workers=1) as _ex:
                        _future = _ex.submit(_call_hf)
                        _resp   = _future.result(timeout=65)

                if _resp.status_code == 200:
                    _data = _resp.json()
                    _text = _data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    _reply = _text or "I couldn't generate a response."
                elif _resp.status_code == 503:
                    _reply = "⏳ Model is warming up. Please try again in ~20 seconds."
                elif _resp.status_code == 403:
                    _reply = "🔑 HuggingFace token lacks access. Check your token at huggingface.co/settings/tokens."
                else:
                    _reply = f"⚠️ Model error ({_resp.status_code}): {_resp.text[:200]}"
        except Exception as _e:
            _reply = f"Error: {_e}"

    # Strip any HTML tags the model may have returned
    import re as _re
    _reply_clean = _re.sub(r'<[^>]+>', '', _reply).strip()
    st.session_state.sage_msgs.append({"role":"user","content":_pending})
    st.session_state.sage_msgs.append({"role":"assistant","content":_reply_clean})

# _sage_hist removed — sidebar now uses st.chat_message directly



# -- Sage: right-side panel using st.dialog (native Streamlit overlay, no JS tricks) --
# st.dialog is fully supported in Streamlit 1.37+ and works on Streamlit Cloud.
# The FAB button opens it; the dialog renders as a modal overlay on the right.

# ── SAGE: right sidebar chat (native Streamlit — reliable, persistent) ────────
# Using st.sidebar which is already fixed-positioned, scrollable independently,
# and does NOT close on rerun. This is the correct Streamlit pattern.

# Style sidebar to look like SAGE panel (right side)
st.markdown("""
<style>
/* Move sidebar to RIGHT side and style as SAGE panel */
[data-testid="stSidebar"] {
    left: auto !important;
    right: 0 !important;
    background: #080f1e !important;
    border-left: 1px solid rgba(56,189,248,0.22) !important;
    border-right: none !important;
    min-width: 320px !important;
    max-width: 320px !important;
}
[data-testid="stSidebar"] > div:first-child {
    background: #080f1e !important;
    padding-top: 0.75rem !important;
}
/* Hide default sidebar nav toggle arrow on left — we have our own FAB */
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
/* When sidebar open, push main content left */
[data-testid="stSidebar"][aria-expanded="true"] ~ [data-testid="stMainBlockContainer"] {
    margin-right: 320px !important;
    transition: margin-right 0.3s ease !important;
}
/* Animated FAB — fixed bottom right, always visible */
.sage-fab-wrapper {
    position: fixed !important;
    bottom: 4.5rem !important;
    right: 0.5rem !important;
    width: 90px !important;
    height: 90px !important;
    z-index: 99999 !important;
    cursor: pointer !important;
}
/* Style chat input inside sidebar */
[data-testid="stSidebar"] [data-testid="stChatInput"] textarea {
    background: #0c1824 !important;
    border: 1px solid rgba(56,189,248,0.22) !important;
    color: #f1f5f9 !important;
}
/* No avatars on chat messages */
[data-testid="stSidebar"] [data-testid="chatAvatarIcon-user"],
[data-testid="stSidebar"] [data-testid="chatAvatarIcon-assistant"] {
    display: none !important;
}
[data-testid="stSidebar"] [data-testid="stChatMessage"] {
    padding: 0.2rem 0 !important;
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SAGE — persistent right sidebar chat assistant
# Always visible, independent scroll, no toggle tricks needed.
# ══════════════════════════════════════════════════════════════════════════════

# Session state
if "sage_msgs"    not in st.session_state: st.session_state.sage_msgs    = []
if "sage_pending" not in st.session_state: st.session_state.sage_pending = None
if "sage_tickers" not in st.session_state: st.session_state.sage_tickers = {}

# Sidebar CSS — right side, dark theme, independent scroll
st.markdown("""
<style>
[data-testid="stSidebar"] {
    left: auto !important;
    right: 0 !important;
    width: 320px !important;
    min-width: 320px !important;
    max-width: 320px !important;
    background: #080f1e !important;
    border-left: 1px solid rgba(56,189,248,0.2) !important;
    border-right: none !important;
}
[data-testid="stSidebar"] > div:first-child {
    background: #080f1e !important;
    padding: 0 !important;
    overflow-y: auto !important;
    height: 100vh !important;
}
/* Hide Streamlit's own sidebar collapse arrow — we keep it always open */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
button[data-testid="baseButton-headerNoPadding"] {
    display: none !important;
}
/* Main content — don't overlap with sidebar */
[data-testid="stMainBlockContainer"] {
    margin-right: 325px !important;
}
/* Chat input styling */
[data-testid="stSidebar"] [data-testid="stChatInput"] textarea {
    background: #0c1824 !important;
    border: 1px solid rgba(56,189,248,0.22) !important;
    color: #f1f5f9 !important;
    font-size: 0.82rem !important;
}
[data-testid="stSidebar"] [data-testid="stChatInput"] button {
    background: #1e3a5f !important;
    border: none !important;
    color: #38bdf8 !important;
}
/* Chat messages */
[data-testid="stSidebar"] [data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 0.1rem 0 !important;
}
[data-testid="stSidebar"] [data-testid="chatAvatarIcon-user"],
[data-testid="stSidebar"] [data-testid="chatAvatarIcon-assistant"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# ── Ticker lookup ─────────────────────────────────────────────────────────────
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

def _sk_find(msg):
    found = []
    for w in msg.upper().split():
        w = w.strip(".,!?()[]'\"")
        if w in _SK and w not in found:
            found.append(w)
    return found

def _sage_signal(ft, top_df):
    if top_df is None or top_df.empty: return None
    m = top_df[top_df["ticker"].str.upper() == ft.upper()]
    return None if m.empty else {
        "predicted":  int(m.iloc[0]["predicted"]),
        "confidence": float(m.iloc[0]["confidence"]),
        "sentiment":  float(m.iloc[0]["sentiment"]),
        "updated_at": m.iloc[0]["updated_at"],
    }

def _sage_live(ft, mkt, exch):
    try:
        from src.data_ingestion      import DataIngestion
        from src.sentiment_model     import SentimentModel
        from src.feature_engineering import FeatureEngineer
        import pytz as _ptz
        arts = DataIngestion().fetch_news(ft, market=mkt, exchange=exch)
        sc   = SentimentModel().score_articles(arts) if arts else []
        fe   = FeatureEngineer().build_features(sc)  if sc   else None
        if fe is None or fe.empty: return None
        s   = float(fe.iloc[-1].get("mean_score", 0))
        now = datetime.now(_ptz.utc)
        return {"predicted": 1 if s>0 else -1,
                 "confidence": min(abs(s)*2+0.5,0.99),
                 "sentiment": s, "updated_at": now, "live": True}
    except Exception:
        return None

def _sage_hf(messages, context):
    try:
        import requests as _rq, os as _os
        try:    _hf = st.secrets["huggingface"]["token"]
        except Exception: _hf = _os.getenv("HUGGINGFACE_TOKEN","")
        if not _hf:
            return "HuggingFace token not configured. Add [huggingface]\ntoken='hf_xxx' to Streamlit secrets."
        system = ("You are Sage, a concise financial signal assistant for Sentiment Signal. "
                  "Answer in 2-3 sentences. Never give direct investment advice.\n\n"
                  f"Context:\n{context}")
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|>"
        for m in messages[-6:]:
            r = "user" if m["role"]=="user" else "assistant"
            c = m["content"] if "<div" not in m["content"] else "[signal card]"
            prompt += f"<|start_header_id|>{r}<|end_header_id|>\n{c}<|eot_id|>"
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
        resp = _rq.post(
            "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct",
            headers={"Authorization": f"Bearer {_hf}"},
            json={"inputs": prompt, "parameters": {
                "max_new_tokens": 160, "temperature": 0.4,
                "return_full_text": False,
                "stop": ["<|eot_id|>","<|start_header_id|>"]}},
            timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            text = (data[0].get("generated_text","") if isinstance(data,list) and data else "").strip()
            for s2 in ["<|eot_id|>","<|start_header_id|>"]:
                text = text[:text.index(s2)].strip() if s2 in text else text
            return text or "I couldn't generate a response. Please try again."
        elif resp.status_code == 503:
            return "Model is warming up (~20s). Please try again shortly."
        else:
            return f"Model error ({resp.status_code}). Ticker spotlights still work — just type a ticker."
    except Exception as e:
        return f"Error: {e}"

# ── Process pending message ───────────────────────────────────────────────────
if st.session_state.sage_pending:
    _pending = st.session_state.sage_pending
    st.session_state.sage_pending = None
    _top = load_top_signals(n=20)
    _sent = load_market_sentiment()
    _found = _sk_find(_pending)

    if _found:
        _ft = _found[0]
        _mkt, _exch, _dt, _co = _SK.get(_ft, (None, None, _ft, _ft))
        _sig = _sage_signal(_ft, _top)
        if not _sig and _mkt:
            with st.spinner(f"Fetching live signal for {_dt}…"):
                _sig = _sage_live(_ft, _mkt, _exch)
        if _sig and _mkt:
            _is_up = _sig["predicted"] == 1
            _sc    = _sig["sentiment"]
            _col   = "#22c55e" if _is_up else "#ef4444"
            _slbl  = "bullish" if _sc>0.05 else ("bearish" if _sc<-0.05 else "neutral")
            import pytz as _ptz2
            _age_s = (datetime.now(_ptz2.utc) - _sig["updated_at"]).total_seconds()
            _age   = f"{int(_age_s//60)}m ago" if _age_s<3600 else f"{int(_age_s//3600)}h {int((_age_s%3600)//60)}m ago"
            _flag  = (_exch or _mkt or "").split(" ")[0]
            _live  = " · live" if _sig.get("live") else ""
            _reply = (
                f"**{_flag} {_dt}** ({_co})\n\n"
                f"{':green[▲ BUY]' if _is_up else ':red[▼ SELL]'} "
                f"· {_sig['confidence']:.0%} confidence\n\n"
                f"Sentiment: {_slbl} ({_sc:+.3f})\n\n"
                f"⏱ {_age}{_live}"
            )
            st.session_state.sage_tickers[_ft] = (_mkt, _exch, _dt, _co)
        elif _mkt:
            _reply = f"Couldn't fetch a live signal for **{_dt}** right now."
            st.session_state.sage_tickers[_ft] = (_mkt, _exch, _dt, _co)
        else:
            _reply = f"**{_ft}** isn't in our tracked list."
    else:
        # Conversational
        _ctx_lines = []
        if _top is not None and not _top.empty:
            _ctx_lines.append("Top signals: " + ", ".join(
                f"{r['ticker']} {'BUY' if int(r['predicted'])==1 else 'SELL'} {float(r['confidence']):.0%}"
                for _,r in _top.head(5).iterrows()))
        if _sent:
            _ctx_lines.append("Sentiment: " + " | ".join(
                f"{mk}: {d.get('label','?')} ({d.get('score',0):+.3f})"
                for mk,d in _sent.items()))
        _reply = _sage_hf(st.session_state.sage_msgs, "\n".join(_ctx_lines) or "No data.")

    st.session_state.sage_msgs.append({"role":"user",    "content":_pending})
    st.session_state.sage_msgs.append({"role":"assistant","content":_reply})

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
# CHATBOT_PLACEHOLDER


# ── Render sidebar ────────────────────────────────────────────────────────────
_SAGE_URI = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0naHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnIHZpZXdCb3g9JzAgMCA3MiA3Mic+PGRlZnM+PHJhZGlhbEdyYWRpZW50IGlkPSdzYmcnIGN4PSc1MCUnIGN5PSc1MCUnIHI9JzUwJSc+PHN0b3Agb2Zmc2V0PScwJScgc3RvcC1jb2xvcj0nJTIzMDgyMDQwJy8+PHN0b3Agb2Zmc2V0PScxMDAlJyBzdG9wLWNvbG9yPSclMjMwNTBmMWYnLz48L3JhZGlhbEdyYWRpZW50PjxmaWx0ZXIgaWQ9J3NnbG93JyB4PSctMzAlJyB5PSctMzAlJyB3aWR0aD0nMTYwJScgaGVpZ2h0PScxNjAlJz48ZmVHYXVzc2lhbkJsdXIgc3RkRGV2aWF0aW9uPScxLjUnIHJlc3VsdD0nYmx1cicvPjxmZU1lcmdlPjxmZU1lcmdlTm9kZSBpbj0nYmx1cicvPjxmZU1lcmdlTm9kZSBpbj0nU291cmNlR3JhcGhpYycvPjwvZmVNZXJnZT48L2ZpbHRlcj48L2RlZnM+PHJlY3Qgd2lkdGg9JzcyJyBoZWlnaHQ9JzcyJyByeD0nMTgnIGZpbGw9J3VybCglMjNzYmcpJy8+PHJlY3Qgd2lkdGg9JzcyJyBoZWlnaHQ9JzcyJyByeD0nMTgnIGZpbGw9J25vbmUnIHN0cm9rZT0nJTIzMzhiZGY4JyBzdHJva2Utd2lkdGg9JzEnIG9wYWNpdHk9JzAuMycvPjxwb2x5bGluZSBwb2ludHM9JzYsMzggMTIsMzggMTUsMjYgMTguNSw1MCAyMiwyOCAyNS41LDQ0IDI5LDIzIDMyLjUsNDcgMzYsMzAgMzksNDIgNDIsMzggNDgsMzgnIGZpbGw9J25vbmUnIHN0cm9rZT0nJTIzMzhiZGY4JyBzdHJva2Utd2lkdGg9JzInIHN0cm9rZS1saW5lY2FwPSdyb3VuZCcgc3Ryb2tlLWxpbmVqb2luPSdyb3VuZCcgZmlsdGVyPSd1cmwoJTIzc2dsb3cpJyBvcGFjaXR5PScwLjk1Jy8+PGNpcmNsZSBjeD0nNDgnIGN5PSczOCcgcj0nMycgZmlsbD0nJTIzMzhiZGY4JyBmaWx0ZXI9J3VybCglMjNzZ2xvdyknIG9wYWNpdHk9JzAuOTUnLz48Y2lyY2xlIGN4PSc0OCcgY3k9JzM4JyByPSc2JyBmaWxsPSclMjMzOGJkZjgnIG9wYWNpdHk9JzAuMTInLz48dGV4dCB4PSczNicgeT0nNjInIHRleHQtYW5jaG9yPSdtaWRkbGUnIGZvbnQtZmFtaWx5PSdtb25vc3BhY2UnIGZvbnQtc2l6ZT0nOScgZm9udC13ZWlnaHQ9JzcwMCcgbGV0dGVyLXNwYWNpbmc9JzMnIGZpbGw9JyUyMzM4YmRmOCcgb3BhY2l0eT0nMC45Jz5TQUdFPC90ZXh0Pjwvc3ZnPg=="
with st.sidebar:
    # Header
    st.markdown(f"""
<div style="display:flex;align-items:center;gap:0.6rem;
            padding:0.75rem 1rem 0.6rem;
            border-bottom:1px solid rgba(56,189,248,0.15);
            background:linear-gradient(135deg,#0c1f3d,#080f1e);">
  <img src="{_SAGE_URI}" style="width:36px;height:36px;border-radius:10px;"/>
  <div>
    <div style="font-size:0.9rem;font-weight:700;color:#f1f5f9;font-family:monospace;letter-spacing:1px">SAGE</div>
    <div style="font-size:0.67rem;color:#38bdf8">Sentiment Signal Assistant</div>
  </div>
</div>
    """, unsafe_allow_html=True)

    # Suggestion chips — show only when no messages yet
    if not st.session_state.sage_msgs:
        st.markdown("<div style='padding:0.5rem 0.25rem 0.25rem;font-size:0.72rem;color:#475569'>Quick questions:</div>",
                    unsafe_allow_html=True)
        _suggs = [
            "What's the strongest signal today?",
            "Which markets are bearish?",
            "Explain the top BUY signal",
            "Any high-confidence SELL signals?",
        ]
        for _s in _suggs:
            if st.button(_s, key=f"sage_sugg_{hash(_s)}", use_container_width=True, type="secondary"):
                st.session_state.sage_pending = _s
                st.rerun()

    # Chat history
    for _msg in st.session_state.sage_msgs[-20:]:
        with st.chat_message(_msg["role"]):
            st.markdown(_msg["content"], unsafe_allow_html=False)

    # Open analysis buttons for any ticker spotlights
    if st.session_state.sage_tickers:
        st.divider()
        for _ft2, (_m2,_e2,_d2,_c2) in st.session_state.sage_tickers.items():
            if st.button(f"📊 Open {_d2} analysis →", key=f"sage_open_btn_{_ft2}",
                         use_container_width=True, type="primary"):
                load_top_signals.clear()
                add_ticker_tab(_ft2, _d2, _c2, _m2, _e2)

    # Clear button
    if st.session_state.sage_msgs:
        if st.button("🗑 Clear chat", key="sage_clear", type="secondary"):
            st.session_state.sage_msgs    = []
            st.session_state.sage_tickers = {}
            st.rerun()

    # Chat input — always at bottom
    _user_input = st.chat_input("Ask about a stock or market…", key="sage_chat_input")
    if _user_input:
        st.session_state.sage_pending = _user_input
        st.rerun()




# ══════════════════════════════════════════════════════════════════════════════
# TICKER ANALYSIS TAB
# ══════════════════════════════════════════════════════════════════════════════
