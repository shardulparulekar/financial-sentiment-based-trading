"""
ticker_utils.py
---------------
Utilities for handling international stock tickers.

Different exchanges use different ticker formats in yfinance:
    India  (NSE)  : RELIANCE.NS, TCS.NS, INFY.NS
    India  (BSE)  : RELIANCE.BO
    UK     (LSE)  : HSBA.L, BP.L
    Germany(XETRA): SAP.DE, BMW.DE
    Japan  (TSE)  : 7203.T  (Toyota)
    China  (SSE)  : 600519.SS (Moutai)
    China  (SZSE) : 000858.SZ
    HK     (HKEX) : 0700.HK  (Tencent)
    Europe (Euronext): ASML.AS, MC.PA

Usage:
    from src.ticker_utils import resolve_ticker, get_market_info

    info = get_market_info("RELIANCE.NS")
    # {"market": "India NSE", "currency": "INR", "company": "Reliance Industries",
    #  "news_lang": "en", "news_region": "IN"}
"""

# ── Known ticker → company name mapping ───────────────────────────────────────
# Used for RSS news search queries.
# Format: "TICKER.SUFFIX" or plain "TICKER" for US stocks.

TICKER_COMPANY_MAP = {
    # ── USA ───────────────────────────────────────────────────────────────────
    "AAPL":    "Apple",
    "MSFT":    "Microsoft",
    "GOOGL":   "Google Alphabet",
    "AMZN":    "Amazon",
    "META":    "Meta Facebook",
    "TSLA":    "Tesla",
    "NVDA":    "NVIDIA",
    "NFLX":    "Netflix",
    "JPM":     "JPMorgan Chase",
    "BAC":     "Bank of America",

    # ── India NSE ─────────────────────────────────────────────────────────────
    "RELIANCE.NS":  "Reliance Industries",
    "TCS.NS":       "Tata Consultancy Services TCS",
    "INFY.NS":      "Infosys",
    "HDFCBANK.NS":  "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "WIPRO.NS":     "Wipro",
    "HINDUNILVR.NS":"Hindustan Unilever HUL",
    "ITC.NS":       "ITC Limited",
    "SBIN.NS":      "State Bank of India SBI",
    "BAJFINANCE.NS":"Bajaj Finance",
    "TATAMOTORS.NS":"Tata Motors",
    "ADANIENT.NS":  "Adani Enterprises",
    "MARUTI.NS":    "Maruti Suzuki",
    "ZOMATO.NS":    "Zomato",
    "PAYTM.NS":     "Paytm One97 Communications",

    # ── China ─────────────────────────────────────────────────────────────────
    "BABA":         "Alibaba",
    "PDD":          "PDD Pinduoduo Temu",
    "JD":           "JD.com",
    "BIDU":         "Baidu",
    "NIO":          "NIO electric vehicle",
    "9988.HK":      "Alibaba Hong Kong",
    "0700.HK":      "Tencent",
    "3690.HK":      "Meituan",

    # ── Europe ────────────────────────────────────────────────────────────────
    "SAP.DE":       "SAP",
    "ASML.AS":      "ASML semiconductor",
    "MC.PA":        "LVMH Louis Vuitton",
    "NESN.SW":      "Nestle",
    "NOVN.SW":      "Novartis",
    "BP.L":         "BP oil",
    "HSBA.L":       "HSBC bank",
    "SHEL.L":       "Shell energy",

    # ── Japan ─────────────────────────────────────────────────────────────────
    "7203.T":       "Toyota Motor",
    "6758.T":       "Sony",
    "9984.T":       "SoftBank",
    "6861.T":       "Keyence",
}

# ── Market detection by ticker suffix ─────────────────────────────────────────

MARKET_CONFIG = {
    ".NS": {
        "market":      "India NSE",
        "currency":    "INR",
        "news_region": "IN",
        "news_lang":   "en",
        "timezone":    "Asia/Kolkata",
        "close_hour":  15,   # 3:30pm IST
    },
    ".BO": {
        "market":      "India BSE",
        "currency":    "INR",
        "news_region": "IN",
        "news_lang":   "en",
        "timezone":    "Asia/Kolkata",
        "close_hour":  15,
    },
    ".HK": {
        "market":      "Hong Kong HKEX",
        "currency":    "HKD",
        "news_region": "HK",
        "news_lang":   "en",
        "timezone":    "Asia/Hong_Kong",
        "close_hour":  16,
    },
    ".SS": {
        "market":      "China Shanghai SSE",
        "currency":    "CNY",
        "news_region": "CN",
        "news_lang":   "en",
        "timezone":    "Asia/Shanghai",
        "close_hour":  15,
    },
    ".SZ": {
        "market":      "China Shenzhen SZSE",
        "currency":    "CNY",
        "news_region": "CN",
        "news_lang":   "en",
        "timezone":    "Asia/Shanghai",
        "close_hour":  15,
    },
    ".T": {
        "market":      "Japan TSE",
        "currency":    "JPY",
        "news_region": "JP",
        "news_lang":   "en",
        "timezone":    "Asia/Tokyo",
        "close_hour":  15,
    },
    ".L": {
        "market":      "UK LSE",
        "currency":    "GBP",
        "news_region": "GB",
        "news_lang":   "en",
        "timezone":    "Europe/London",
        "close_hour":  16,
    },
    ".DE": {
        "market":      "Germany XETRA",
        "currency":    "EUR",
        "news_region": "DE",
        "news_lang":   "en",
        "timezone":    "Europe/Berlin",
        "close_hour":  17,
    },
    ".AS": {
        "market":      "Netherlands Euronext",
        "currency":    "EUR",
        "news_region": "NL",
        "news_lang":   "en",
        "timezone":    "Europe/Amsterdam",
        "close_hour":  17,
    },
    ".PA": {
        "market":      "France Euronext",
        "currency":    "EUR",
        "news_region": "FR",
        "news_lang":   "en",
        "timezone":    "Europe/Paris",
        "close_hour":  17,
    },
    ".SW": {
        "market":      "Switzerland SIX",
        "currency":    "CHF",
        "news_region": "CH",
        "news_lang":   "en",
        "timezone":    "Europe/Zurich",
        "close_hour":  17,
    },
}

# Default for plain US tickers (no suffix)
US_MARKET = {
    "market":      "USA NASDAQ/NYSE",
    "currency":    "USD",
    "news_region": "US",
    "news_lang":   "en",
    "timezone":    "America/New_York",
    "close_hour":  16,
}


def get_market_info(ticker: str) -> dict:
    """
    Detect which market a ticker belongs to based on its suffix.

    Args:
        ticker : e.g. "RELIANCE.NS", "AAPL", "0700.HK"

    Returns:
        dict with market, currency, news_region, news_lang, timezone, close_hour
    """
    ticker = ticker.upper()
    for suffix, config in MARKET_CONFIG.items():
        if ticker.endswith(suffix.upper()):
            return {"ticker": ticker, **config}
    return {"ticker": ticker, **US_MARKET}


def get_company_name(ticker: str) -> str:
    """
    Look up a human-readable company name for RSS news search queries.
    Falls back to the base ticker symbol (without suffix) if not found.
    """
    ticker = ticker.upper()
    if ticker in TICKER_COMPANY_MAP:
        return TICKER_COMPANY_MAP[ticker]

    # Strip exchange suffix and use base symbol
    base = ticker.split(".")[0]
    return base


def get_rss_sources_for_market(market_info: dict) -> list[str]:
    """
    Return the appropriate RSS source keys for a given market.
    Different regions have different reliable news sources.
    """
    region = market_info.get("news_region", "US")

    if region == "IN":
        return ["economic_times", "moneycontrol", "google_news_india"]
    elif region in ("HK", "CN"):
        return ["google_news_asia", "seeking_alpha"]
    elif region in ("GB", "DE", "FR", "NL", "CH"):
        return ["google_news_europe", "seeking_alpha"]
    elif region == "JP":
        return ["google_news_asia", "seeking_alpha"]
    else:
        return ["seeking_alpha", "google_news"]
