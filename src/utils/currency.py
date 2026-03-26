"""
Currency Converter
------------------
Fetches live exchange rates and converts between EUR, NGN, USD.
Used by daily_advisor.py so users can enter amounts in any currency.

Rates fetched from yfinance (free, no API key needed).
Fallback to hardcoded rates if fetch fails.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

# Fallback rates (updated manually if needed)
FALLBACK_RATES = {
    "EUR_NGN": 1650.0,
    "USD_NGN": 1550.0,
    "EUR_USD": 1.08,
    "NGN_EUR": 1 / 1650.0,
    "NGN_USD": 1 / 1550.0,
    "USD_EUR": 1 / 1.08,
}


def fetch_rate(from_ccy: str, to_ccy: str) -> float:
    """
    Fetch live exchange rate from_ccy → to_ccy.
    Returns float rate or fallback if unavailable.
    """
    pair_key = f"{from_ccy}_{to_ccy}"

    if not YF_AVAILABLE:
        return FALLBACK_RATES.get(pair_key, 1.0)

    # yfinance forex pairs
    pair_map = {
        "EUR_NGN": "EURNGN=X",
        "USD_NGN": "USDNGN=X",
        "EUR_USD": "EURUSD=X",
        "USD_EUR": "USDEUR=X",
        "NGN_EUR": "NGNEUR=X",
        "NGN_USD": "NGNUSD=X",
    }

    ticker_sym = pair_map.get(pair_key)
    if not ticker_sym:
        # Try reverse
        rev_key    = f"{to_ccy}_{from_ccy}"
        rev_ticker = pair_map.get(rev_key)
        if rev_ticker:
            rate = _fetch_ticker(rev_ticker)
            return 1 / rate if rate else FALLBACK_RATES.get(pair_key, 1.0)
        return FALLBACK_RATES.get(pair_key, 1.0)

    rate = _fetch_ticker(ticker_sym)
    return rate if rate else FALLBACK_RATES.get(pair_key, 1.0)


def _fetch_ticker(symbol: str) -> float:
    """Fetch latest price for a yfinance ticker."""
    try:
        t  = yf.Ticker(symbol)
        px = t.fast_info.get("last_price") or \
             t.fast_info.get("regularMarketPrice")
        if px and float(px) > 0:
            return float(px)
        # Fallback: download 1d
        import pandas as pd
        df = yf.download(symbol, period="5d",
                         interval="1d", progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            return float(df["Close"].iloc[-1])
    except Exception:
        pass
    return 0.0


def convert(amount: float,
            from_ccy: str,
            to_ccy: str,
            rate: float = None) -> float:
    """Convert amount from one currency to another."""
    if from_ccy == to_ccy:
        return amount
    r = rate or fetch_rate(from_ccy.upper(), to_ccy.upper())
    return amount * r


class Portfolio:
    """
    Holds portfolio value in multiple currencies.
    Always stores internally in EUR.
    """

    def __init__(self, amount: float, currency: str,
                 rates: dict = None):
        currency  = currency.upper()
        self.rates = rates or {}

        if currency == "EUR":
            self.eur = amount
        elif currency == "NGN":
            rate      = self.rates.get("NGN_EUR") or \
                        fetch_rate("NGN", "EUR")
            self.eur  = amount * rate
        elif currency == "USD":
            rate      = self.rates.get("USD_EUR") or \
                        fetch_rate("USD", "EUR")
            self.eur  = amount * rate
        else:
            print(f"  ⚠️  Unknown currency {currency} — treating as EUR")
            self.eur  = amount

        self.original_amount   = amount
        self.original_currency = currency

    @property
    def ngn(self) -> float:
        rate = self.rates.get("EUR_NGN") or fetch_rate("EUR", "NGN")
        return self.eur * rate

    @property
    def usd(self) -> float:
        rate = self.rates.get("EUR_USD") or fetch_rate("EUR", "USD")
        return self.eur * rate

    def display(self) -> str:
        return (f"₦{self.ngn:,.0f}  /  "
                f"€{self.eur:,.2f}  /  "
                f"${self.usd:,.2f}")


def get_all_rates() -> dict:
    """Fetch all rates needed by the system in one call."""
    print("  Fetching live exchange rates...", end=" ")
    rates = {
        "EUR_NGN": fetch_rate("EUR", "NGN"),
        "USD_NGN": fetch_rate("USD", "NGN"),
        "EUR_USD": fetch_rate("EUR", "USD"),
        "NGN_EUR": fetch_rate("NGN", "EUR"),
        "NGN_USD": fetch_rate("NGN", "USD"),
        "USD_EUR": fetch_rate("USD", "EUR"),
    }
    print(f"done  (€1 = ₦{rates['EUR_NGN']:,.0f}  /  "
          f"$1 = ₦{rates['USD_NGN']:,.0f})")
    return rates


def parse_amount_input(raw: str, rates: dict = None) -> tuple:
    """
    Parse user input like '10000', '₦10000', '€500', '$200'.
    Returns (amount_eur, original_amount, original_currency).
    """
    raw = raw.strip().replace(",", "").replace(" ", "")

    if raw.startswith("₦") or raw.upper().endswith("NGN"):
        ccy    = "NGN"
        amount = float(raw.replace("₦","").replace("NGN","")
                          .replace("ngn",""))
    elif raw.startswith("€") or raw.upper().endswith("EUR"):
        ccy    = "EUR"
        amount = float(raw.replace("€","").replace("EUR","")
                          .replace("eur",""))
    elif raw.startswith("$") or raw.upper().endswith("USD"):
        ccy    = "USD"
        amount = float(raw.replace("$","").replace("USD","")
                          .replace("usd",""))
    else:
        # Bare number — ask if large (likely NGN) or small (likely EUR)
        amount = float(raw)
        if amount > 10000:
            ccy = "NGN"
        else:
            ccy = "EUR"

    p = Portfolio(amount, ccy, rates)
    return p.eur, p.original_amount, p.original_currency, p
