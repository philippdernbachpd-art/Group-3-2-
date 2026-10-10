"""Equity Valuation Studio — Phase B (single-file edition for GitHub + Streamlit Community Cloud).

Residual income + trading comparables for any listed company, using Yahoo Finance data via yfinance.
Everything (valuation engine, Yahoo data layer, optional Claude features, Streamlit interface) is in this one file.

Deploy:  upload this file + requirements.txt to a GitHub repo  ->  share.streamlit.io  ->  Create app  ->  main file: streamlit_app.py
Local:   pip install -r requirements.txt   then   streamlit run streamlit_app.py
Optional AI tab: set ANTHROPIC_API_KEY in the Streamlit Cloud "Secrets" (never in this file).
"""
from __future__ import annotations

import sys as _sys
import types as _types


def _namespace(name, before):
    """Collect the names defined since `before` into a module object (E, MK, ai), registered so cached objects can be pickled."""
    mod = _types.ModuleType(name)
    for key, val in list(globals().items()):
        if key not in before and not key.startswith("_"):
            setattr(mod, key, val)
            if isinstance(val, type) and val.__module__ == __name__:
                val.__module__ = name
                val.__qualname__ = key
    _sys.modules[name] = mod
    return mod


# ============================================================================
# 1. VALUATION ENGINE — residual income, comps maths, football field  (formerly engine.py)
# ============================================================================
_before = set(globals())
from dataclasses import dataclass, replace
from typing import Dict, List, Optional


# ---------------------------------------------------------------- residual income
@dataclass
class RIAssumptions:
    revenue_growth: float
    net_margin: float
    payout: float
    cost_of_equity: float
    terminal_growth: float
    years: int = 5
    fade_growth: bool = False              # glide growth linearly from year 1 to terminal growth in the final year
    target_margin: Optional[float] = None  # if set, margin glides linearly to this level by the final year


def growth_path(a: RIAssumptions) -> List[float]:
    n = a.years
    if not a.fade_growth or n == 1:
        return [a.revenue_growth] * n
    return [a.revenue_growth + (a.terminal_growth - a.revenue_growth) * t / (n - 1) for t in range(n)]


def margin_path(a: RIAssumptions) -> List[float]:
    n = a.years
    if a.target_margin is None:
        return [a.net_margin] * n
    return [a.net_margin + (a.target_margin - a.net_margin) * (t + 1) / n for t in range(n)]


def capm(risk_free: float, beta: float, erp: float) -> float:
    return risk_free + beta * erp


def forecast(revenue0: float, book0: float, a: RIAssumptions, base_year: int = 0) -> List[Dict[str, float]]:
    rows, rev, bv = [], revenue0, book0
    for t, (g, m) in enumerate(zip(growth_path(a), margin_path(a)), start=1):
        rev = rev * (1 + g)
        ni = rev * m
        div = ni * a.payout
        bv0, bv = bv, bv + ni - div
        charge = a.cost_of_equity * bv0
        ri = ni - charge
        df = 1 / (1 + a.cost_of_equity) ** t
        rows.append({"year": base_year + t, "t": t, "revenue": rev, "growth": g, "net_margin": m, "net_income": ni, "dividends": div,
                     "retained": ni - div, "bv_begin": bv0, "bv_end": bv, "roe_begin": (ni / bv0) if bv0 > 0 else None,
                     "capital_charge": charge, "residual_income": ri, "discount_factor": df, "pv_ri": ri * df})
    return rows


def validate(a: RIAssumptions, shares: float) -> List[str]:
    errs = []
    if a.years < 1 or a.years > 30:
        errs.append("Forecast period must be between 1 and 30 years.")
    if a.cost_of_equity <= a.terminal_growth:
        errs.append(f"Cost of equity ({a.cost_of_equity:.2%}) must be higher than terminal growth ({a.terminal_growth:.2%}).")
    if not shares or shares <= 0:
        errs.append("Share count must be positive.")
    if not 0 <= a.payout <= 1.5:
        errs.append("Payout ratio should be between 0% and 150%.")
    return errs


def residual_income_value(revenue0: float, book0: float, shares: float, a: RIAssumptions, base_year: int = 0) -> Dict[str, object]:
    errs = validate(a, shares)
    if errs:
        raise ValueError(" ".join(errs))
    rows = forecast(revenue0, book0, a, base_year)
    pv_ri = sum(r["pv_ri"] for r in rows)
    last = rows[-1]
    tv = last["residual_income"] * (1 + a.terminal_growth) / (a.cost_of_equity - a.terminal_growth)
    pv_tv = tv * last["discount_factor"]
    equity = book0 + pv_ri + pv_tv
    return {"rows": rows, "book0": book0, "pv_ri": pv_ri, "terminal_value": tv, "pv_tv": pv_tv, "equity_value": equity,
            "shares": shares, "per_share": equity / shares,
            "parts": {"Book value today": book0 / shares, "PV of residual income": pv_ri / shares, "PV of terminal value": pv_tv / shares}}


def sensitivity_grid(revenue0, book0, shares, a: RIAssumptions, field_x: str, xs, field_y: str, ys):
    """Value per share for each (y, x) pair of two assumption fields; None where invalid (e.g. ke ≤ g)."""
    out = []
    for y in ys:
        row = []
        for x in xs:
            b = replace(a, **{field_x: x, field_y: y})
            try:
                row.append(residual_income_value(revenue0, book0, shares, b)["per_share"])
            except ValueError:
                row.append(None)
        out.append(row)
    return out


SCENARIO_SHIFTS = {"Bear": {"revenue_growth": -0.03, "net_margin": -0.02, "terminal_growth": -0.005},
                   "Base": {}, "Bull": {"revenue_growth": 0.03, "net_margin": 0.02, "terminal_growth": 0.005}}


def scenario(a: RIAssumptions, name: str, shifts: Optional[Dict[str, Dict[str, float]]] = None) -> RIAssumptions:
    sh = (shifts or SCENARIO_SHIFTS)[name]
    kw = {k: getattr(a, k) + v for k, v in sh.items()}
    if a.target_margin is not None and "net_margin" in sh:
        kw["target_margin"] = a.target_margin + sh["net_margin"]
    return replace(a, **kw)


# ---------------------------------------------------------------- history-based defaults
def cagr(first: Optional[float], last: Optional[float], years: int) -> Optional[float]:
    if first is None or last is None or first <= 0 or last <= 0 or years <= 0:
        return None
    return (last / first) ** (1 / years) - 1


def historical_defaults(revenue: List[Optional[float]], net_income: List[Optional[float]], dividends: List[Optional[float]]) -> Dict[str, Optional[float]]:
    """Inputs are oldest → newest annual values. Returns suggested starting assumptions plus what they were based on."""
    rv = [x for x in revenue if x is not None]
    g = cagr(rv[0], rv[-1], len(rv) - 1) if len(rv) >= 2 else None
    margins = [n / r for r, n in zip(revenue, net_income) if r and n is not None and r > 0]
    m = sum(margins[-3:]) / len(margins[-3:]) if margins else None
    pays = [d / n for n, d in zip(net_income, dividends) if n and d is not None and n > 0]
    p = min(max(sum(pays) / len(pays), 0.0), 1.0) if pays else None
    return {"revenue_growth": g, "net_margin": m, "payout": p, "years_of_revenue": len(rv), "years_of_margin": len(margins[-3:]), "years_of_payout": len(pays)}


# ---------------------------------------------------------------- trading comparables
MULTIPLES = ("EV/Revenue", "EV/EBITDA", "P/E", "P/Book")


def _pos(x) -> Optional[float]:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if x == x and x > 0 else None


def enterprise_value(d: Dict[str, object]) -> Optional[float]:
    ev = d.get("enterprise_value")
    if ev is not None:
        try:
            return float(ev)
        except (TypeError, ValueError):
            pass
    mc, debt, cash = d.get("market_cap"), d.get("total_debt"), d.get("total_cash")
    if mc is None:
        return None
    return float(mc) + float(debt or 0) - float(cash or 0)


def multiples(d: Dict[str, object]) -> Dict[str, Optional[float]]:
    """Each multiple, or None when the denominator is missing, zero or negative (not meaningful)."""
    ev, price = enterprise_value(d), _pos(d.get("price"))
    out = {}
    rev, ebitda, eps, bvps = _pos(d.get("revenue")), _pos(d.get("ebitda")), _pos(d.get("eps")), _pos(d.get("bvps"))
    out["EV/Revenue"] = ev / rev if (ev is not None and ev > 0 and rev) else None
    out["EV/EBITDA"] = ev / ebitda if (ev is not None and ev > 0 and ebitda) else None
    out["P/E"] = price / eps if (price and eps) else None
    out["P/Book"] = price / bvps if (price and bvps) else None
    return out


def quantile(values: List[float], q: float) -> float:
    v = sorted(values)
    if len(v) == 1:
        return v[0]
    pos = (len(v) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def summarize(values: List[Optional[float]]) -> Optional[Dict[str, float]]:
    v = [x for x in values if x is not None]
    if not v:
        return None
    return {"n": len(v), "low": quantile(v, 0.25), "median": quantile(v, 0.5), "high": quantile(v, 0.75), "mean": sum(v) / len(v), "min": min(v), "max": max(v)}


def implied_price(target: Dict[str, object], method: str, multiple: float) -> Optional[float]:
    """Value per share implied for the target by one multiple; None when the target metric is not meaningful."""
    shares = _pos(target.get("shares"))
    if method in ("EV/Revenue", "EV/EBITDA"):
        metric = _pos(target.get("revenue" if method == "EV/Revenue" else "ebitda"))
        if not (metric and shares):
            return None
        net_debt = float(target.get("total_debt") or 0) - float(target.get("total_cash") or 0)
        return (multiple * metric - net_debt) / shares
    if method == "P/E":
        eps = _pos(target.get("eps"))
        return multiple * eps if eps else None
    if method == "P/Book":
        bvps = _pos(target.get("bvps"))
        return multiple * bvps if bvps else None
    raise ValueError(method)


def comps_valuation(target: Dict[str, object], peers: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    """Per multiple: peer statistics and the implied low / mid / high value per share for the target."""
    table = {p["ticker"]: multiples(p) for p in peers}
    out = {}
    for m in MULTIPLES:
        stats = summarize([table[t][m] for t in table])
        if not stats:
            out[m] = {"stats": None, "implied": None, "reason": "No peer has a meaningful value"}
            continue
        imp = {k: implied_price(target, m, stats[k]) for k in ("low", "median", "high")}
        if imp["median"] is None:
            out[m] = {"stats": stats, "implied": None, "reason": "Target metric is missing or negative"}
        else:
            out[m] = {"stats": stats, "implied": imp, "reason": ""}
    return out


# ---------------------------------------------------------------- football field
def football_field(ri_low, ri_mid, ri_high, comps: Dict[str, Dict[str, object]], low52=None, high52=None,
                   target_low=None, target_mean=None, target_high=None) -> List[Dict[str, object]]:
    rows = []
    if ri_mid is not None:
        rows.append({"method": "Residual income (bear–bull)", "low": min(ri_low, ri_high), "mid": ri_mid, "high": max(ri_low, ri_high), "group": "Intrinsic"})
    for m in MULTIPLES:
        c = comps.get(m) or {}
        imp = c.get("implied")
        if imp:
            vals = [v for v in imp.values() if v is not None]
            rows.append({"method": f"{m} (peer 25th–75th)", "low": min(vals), "mid": imp["median"], "high": max(vals), "group": "Trading comps"})
    if low52 and high52:
        rows.append({"method": "52-week trading range", "low": low52, "mid": None, "high": high52, "group": "Market"})
    if target_low and target_high:
        rows.append({"method": "Analyst price targets", "low": target_low, "mid": target_mean, "high": target_high, "group": "Market"})
    return rows


def blended_value(ri_mid: Optional[float], comps: Dict[str, Dict[str, object]], ri_weight: float = 0.5) -> Optional[float]:
    """Weighted blend: residual income vs the average of available comps medians."""
    meds = [c["implied"]["median"] for c in comps.values() if c.get("implied")]
    if ri_mid is None and not meds:
        return None
    if not meds:
        return ri_mid
    comp_avg = sum(meds) / len(meds)
    if ri_mid is None:
        return comp_avg
    return ri_weight * ri_mid + (1 - ri_weight) * comp_avg
E = _namespace("evs_engine", _before)


# ============================================================================
# 2. MARKET DATA — Yahoo Finance via yfinance, sample data  (formerly market.py)
# ============================================================================
_before = set(globals())
import datetime as dt
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

M = 1e6

INFO_FIELDS = {
    "price": ["currentPrice", "regularMarketPrice", "previousClose"],
    "shares": ["sharesOutstanding", "impliedSharesOutstanding"],
    "market_cap": ["marketCap"],
    "enterprise_value": ["enterpriseValue"],
    "total_debt": ["totalDebt"],
    "total_cash": ["totalCash"],
    "revenue_ttm": ["totalRevenue"],
    "ebitda_ttm": ["ebitda"],
    "eps_ttm": ["trailingEps"],
    "bvps": ["bookValue"],
    "beta": ["beta"],
    "high52": ["fiftyTwoWeekHigh"],
    "low52": ["fiftyTwoWeekLow"],
    "target_low": ["targetLowPrice"],
    "target_mean": ["targetMeanPrice"],
    "target_high": ["targetHighPrice"],
    "dividend_rate": ["dividendRate"],
}
MONEY = {"market_cap", "enterprise_value", "total_debt", "total_cash", "revenue_ttm", "ebitda_ttm"}  # converted to millions
SHARE_FIELDS = {"shares"}  # converted to millions

STATEMENT_ROWS = {
    "revenue": ("income_stmt", ["Total Revenue", "Operating Revenue"]),
    "net_income": ("income_stmt", ["Net Income", "Net Income Common Stockholders"]),
    "ebitda": ("income_stmt", ["EBITDA", "Normalized EBITDA"]),
    "diluted_eps": ("income_stmt", ["Diluted EPS"]),
    "equity": ("balance_sheet", ["Stockholders Equity", "Common Stock Equity"]),
    "dividends": ("cashflow", ["Cash Dividends Paid", "Common Stock Dividend Paid"]),
    "buybacks": ("cashflow", ["Repurchase Of Capital Stock", "Common Stock Payments"]),
}


@dataclass
class Company:
    ticker: str
    name: str
    currency: str
    financial_currency: str
    sector: str
    industry: str
    fields: Dict[str, Optional[float]]
    sources: Dict[str, str]
    history: Dict[str, Dict[int, Optional[float]]]  # field -> {fiscal year: value in millions}
    fiscal_year_end: Optional[dt.date]
    mode: str = "live"  # live | sample
    missing: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    retrieved_at: str = ""

    def f(self, k):
        return self.fields.get(k)

    def series(self, k) -> List[Optional[float]]:
        years = self.years()
        h = self.history.get(k, {})
        return [h.get(y) for y in years]

    def years(self) -> List[int]:
        ys = set()
        for h in self.history.values():
            ys |= set(h)
        return sorted(ys)

    def latest(self, k) -> Optional[float]:
        h = self.history.get(k, {})
        return h[max(h)] if h else None

    def comps_row(self) -> Dict[str, object]:
        """The inputs engine.multiples / implied_price expect (millions; per-share in trading currency)."""
        return {"ticker": self.ticker, "name": self.name, "price": self.f("price"), "market_cap": self.f("market_cap"),
                "enterprise_value": self.f("enterprise_value"), "total_debt": self.f("total_debt"), "total_cash": self.f("total_cash"),
                "revenue": self.f("revenue_ttm"), "ebitda": self.f("ebitda_ttm"), "eps": self.f("eps_ttm"), "bvps": self.f("bvps"),
                "shares": self.f("shares"), "currency": self.currency}


def _num(x) -> Optional[float]:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def from_yahoo(ticker: str, info: Dict[str, object], statements: Dict[str, object], via: Optional[Dict[str, str]] = None) -> Company:
    """Map a yfinance `info` dict and annual statement frames to a Company. Testable without network access."""
    fields, sources, missing, notes = {}, {}, [], []
    for k, keys in INFO_FIELDS.items():
        val, src = None, None
        for key in keys:
            v = _num(info.get(key))
            if v is not None:
                val, src = v, key
                break
        if val is not None and (k in MONEY or k in SHARE_FIELDS):
            val = val / M
        fields[k] = val
        sources[k] = (f"{(via or {}).get(src, 'Yahoo info')} '{src}'") if src else "Not returned by Yahoo"
        if val is None and k not in ("target_low", "target_mean", "target_high", "dividend_rate", "enterprise_value"):
            missing.append(k)
    if fields["enterprise_value"] is None and fields["market_cap"] is not None:
        fields["enterprise_value"] = fields["market_cap"] + (fields["total_debt"] or 0) - (fields["total_cash"] or 0)
        sources["enterprise_value"] = "Calculated: market cap + total debt − total cash"
    if fields["market_cap"] is None and fields["price"] and fields["shares"]:
        fields["market_cap"] = fields["price"] * fields["shares"]
        sources["market_cap"] = "Calculated: price × shares"

    history: Dict[str, Dict[int, Optional[float]]] = {}
    fye = None
    for k, (stmt, labels) in STATEMENT_ROWS.items():
        frame = statements.get(stmt)
        if frame is None or getattr(frame, "empty", True):
            continue
        lab = next((l for l in labels if l in frame.index), None)
        if lab is None:
            continue
        row = frame.loc[lab]
        h = {}
        for col in row.index:
            v = _num(row[col])
            if v is None:
                continue
            year = col.year if hasattr(col, "year") else int(str(col)[:4])
            scale = 1 if k == "diluted_eps" else 1 / M
            h[year] = abs(v) * scale if k in ("dividends", "buybacks") else v * scale
            if hasattr(col, "date") and (fye is None or col.date() > fye):
                fye = col.date()
        history[k] = h
        sources[f"history:{k}"] = f"Yahoo {stmt} '{lab}'"
    # Fill missing trailing figures from the latest annual statements, labelled as such (never zero).
    def _latest(k):
        h = history.get(k) or {}
        return (max(h), h[max(h)]) if h else (None, None)
    for fk, hk, label in (("revenue_ttm", "revenue", "revenue"), ("ebitda_ttm", "ebitda", "EBITDA"), ("eps_ttm", "diluted_eps", "diluted EPS")):
        y, v = _latest(hk)
        if fields.get(fk) is None and v is not None:
            fields[fk] = v
            sources[fk] = f"Fallback: fiscal {y} annual {label} (trailing figure not returned)"
            if fk in missing:
                missing.remove(fk)
    y, eq = _latest("equity")
    if fields.get("bvps") is None and eq and fields.get("shares"):
        fields["bvps"] = eq / fields["shares"]
        sources["bvps"] = f"Calculated: fiscal {y} equity ÷ shares outstanding"
        if "bvps" in missing:
            missing.remove("bvps")
    for k in ("revenue", "net_income", "equity"):
        if not history.get(k):
            missing.append(f"annual {k}")
    if "dividends" not in history:
        notes.append("No dividends line in Yahoo's cash flow statement; the payout ratio defaults to 0% (editable).")
    cur, fcur = str(info.get("currency") or ""), str(info.get("financialCurrency") or info.get("currency") or "")
    if cur and fcur and cur != fcur:
        notes.append(f"Shares trade in {cur} but statements are in {fcur}. Per-share values from statements are not converted; "
                     f"compare with the price only after adjusting for the exchange rate.")
    return Company(ticker=ticker.upper(), name=str(info.get("longName") or info.get("shortName") or ticker.upper()), currency=cur or "USD",
                   financial_currency=fcur or cur or "USD", sector=str(info.get("sector") or "—"), industry=str(info.get("industry") or "—"),
                   fields=fields, sources=sources, history=history, fiscal_year_end=fye, mode="live", missing=missing, notes=notes,
                   retrieved_at=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))


class YahooError(RuntimeError):
    """Raised when Yahoo Finance gives no usable quote. The message is shown to the user."""


RATE_LIMIT_HINTS = ("too many requests", "rate limit", "ratelimit", "429")


def _is_rate_limit(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(h in text for h in RATE_LIMIT_HINTS)


def _retry(fn, waits=(0.0, 1.5, 4.0)):
    """Call fn, retrying after short pauses. Yahoo often answers the second or third request from cloud servers."""
    last = None
    for w in waits:
        if w:
            time.sleep(w)
        try:
            return fn()
        except Exception as exc:  # network errors, rate limits, Yahoo hiccups
            last = exc
    raise last


def _has_price(info: Dict[str, object]) -> bool:
    return any(_num(info.get(k)) for k in ("currentPrice", "regularMarketPrice", "previousClose"))


def _quick_quote(t) -> Dict[str, object]:
    """Fallback quote from yfinance's fast_info (Yahoo chart endpoint), used when the full quote summary fails."""
    out = {}
    try:
        fi = t.fast_info
    except Exception:
        return out
    for info_key, attr in (("currentPrice", "last_price"), ("previousClose", "previous_close"), ("marketCap", "market_cap"),
                           ("sharesOutstanding", "shares"), ("fiftyTwoWeekHigh", "year_high"), ("fiftyTwoWeekLow", "year_low"),
                           ("currency", "currency")):
        try:
            v = getattr(fi, attr)
        except Exception:
            v = None
        if v is not None and (info_key == "currency" or _num(v) is not None):
            out[info_key] = v
    return out


def _history_quote(t) -> Dict[str, object]:
    """Second fallback: last close and 52-week range from one year of daily prices."""
    try:
        h = t.history(period="1y", auto_adjust=False)
    except Exception:
        return {}
    if h is None or getattr(h, "empty", True) or "Close" not in h:
        return {}
    close = h["Close"].dropna()
    if close.empty:
        return {}
    return {"currentPrice": float(close.iloc[-1]), "fiftyTwoWeekHigh": float(h.get("High", close).max()),
            "fiftyTwoWeekLow": float(h.get("Low", close).min())}


def fetch(ticker: str) -> Company:
    """Live data for one ticker. Tries Yahoo's full quote, then fast_info, then price history; statements are fetched separately."""
    import yfinance as yf  # imported lazily so the rest of the app imports without it
    ticker = ticker.strip().upper()
    t = yf.Ticker(ticker)
    info, info_err, via = {}, None, {}
    try:
        info = dict(_retry(lambda: t.info or {}))
    except Exception as exc:
        info_err = exc
    if not _has_price(info):
        for label, getter in (("Yahoo fast_info", _quick_quote), ("Yahoo price history", _history_quote)):
            extra = getter(t)
            for k_, v in extra.items():
                if info.get(k_) in (None, ""):
                    info[k_] = v
                    via[k_] = label
            if _has_price(info):
                break
    if not _has_price(info):
        if info_err is not None and _is_rate_limit(info_err):
            raise YahooError("Yahoo Finance is limiting requests from this server right now (\"Too Many Requests\"). "
                             "Wait about a minute, then press Refresh live data.")
        if info_err is not None:
            raise YahooError(f"Yahoo Finance did not respond for {ticker} ({type(info_err).__name__}: {info_err}). Press Refresh live data to try again.")
        raise YahooError(f"Yahoo Finance has no quote for {ticker}. Check the symbol, e.g. NKE, AAPL, MSFT, ADS.DE or 7203.T.")
    statements = {}
    for name in ("income_stmt", "balance_sheet", "cashflow"):
        try:
            statements[name] = _retry(lambda n=name: getattr(t, n), waits=(0.0, 1.5))
        except Exception:
            statements[name] = None
    co = from_yahoo(ticker, info, statements, via)
    if via:
        co.notes.append("Yahoo's full quote was unavailable, so the price and market data come from Yahoo's chart feed. "
                        "Fields such as debt, EBITDA, EPS or beta may be missing until a refresh succeeds.")
    return co


def fetch_risk_free() -> Optional[float]:
    """10-year US Treasury yield from Yahoo's ^TNX (quoted in percent)."""
    try:
        import yfinance as yf
        h = _retry(lambda: yf.Ticker("^TNX").history(period="5d"), waits=(0.0, 1.5))
        return float(h["Close"].dropna().iloc[-1]) / 100 if not h.empty else None
    except Exception:
        return None


def fetch_price_history(ticker: str, period: str = "2y"):
    try:
        import yfinance as yf
        h = _retry(lambda: yf.Ticker(ticker).history(period=period), waits=(0.0, 1.5))
        return h[["Close"]].reset_index() if not h.empty else None
    except Exception:
        return None


def sample_nike() -> Company:
    """Offline sample: Nike FY2026 Form 10-K figures (verified in Phase A) and the 9 Oct 2026 price. Labelled as sample data in the app."""
    src = "Sample: Nike FY2026 Form 10-K / Phase A"
    fields = {"price": 34.71, "shares": 1483.0, "market_cap": 34.71 * 1483.0, "enterprise_value": None, "total_debt": 7942.0, "total_cash": 9027.0,
              "revenue_ttm": 46398.0, "ebitda_ttm": 3850.0 + 747.0, "eps_ttm": 2.10, "bvps": 14865.0 / 1483.0, "beta": 1.0445,
              "high52": 70.44, "low52": 31.97, "target_low": None, "target_mean": None, "target_high": None, "dividend_rate": None}
    fields["enterprise_value"] = fields["market_cap"] + fields["total_debt"] - fields["total_cash"]
    sources = {k: src for k in fields}
    sources["ebitda_ttm"] = "Sample: segment EBIT 3,850 + D&A 747 (FY2026 10-K)"
    sources["total_debt"] = "Sample: senior notes only (excludes leases)"
    sources["total_cash"] = "Sample: cash and short-term investments"
    sources["price"] = sources["beta"] = sources["high52"] = sources["low52"] = "Sample: Excel Stocks data type snapshot, 9 Oct 2026"
    history = {"revenue": {2023: 51217.0, 2024: 51362.0, 2025: 46309.0, 2026: 46398.0},
               "net_income": {2023: 5070.0, 2024: 5700.0, 2025: 3219.0, 2026: 3108.0},
               "equity": {2023: 14004.0, 2024: 14430.0, 2025: 13213.0, 2026: 14865.0},
               "dividends": {2024: 2169.0, 2025: 2300.0, 2026: 2407.0},
               "buybacks": {2024: 4250.0, 2025: 2985.0, 2026: 146.0},
               "diluted_eps": {2023: 3.23, 2024: 3.73, 2025: 2.16, 2026: 2.10}}
    return Company("NKE", "NIKE, Inc. (sample data)", "USD", "USD", "Consumer Cyclical", "Footwear & Accessories", fields, sources, history,
                   dt.date(2026, 5, 31), mode="sample", missing=["target_low", "target_mean", "target_high"],
                   notes=["Sample mode: figures are a fixed FY2026 snapshot, not live Yahoo data. Trading comps need live data."],
                   retrieved_at="Snapshot 9 Oct 2026")
MK = _namespace("evs_market", _before)


# ============================================================================
# 3. AI ANALYST — optional Claude features (Anthropic API)  (formerly ai.py)
# ============================================================================
_before = set(globals())
import json
import re
from typing import Dict, List, Optional

MODELS = {"Claude Sonnet 5.5 (faster)": "claude-sonnet-5-5", "Claude Opus 5.5 (deeper)": "claude-opus-5-5"}

SYSTEM = ("You are an equity research associate helping a student valuation team. Be concise, specific and numerate. "
          "Use only the figures provided; do not invent data. When you are unsure, say so. "
          "This is educational analysis, not investment advice.")


def available(api_key: Optional[str]) -> bool:
    if not api_key:
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def ask(api_key: str, model: str, prompt: str, max_tokens: int = 1200, system: str = SYSTEM) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(model=model, max_tokens=max_tokens, system=system, messages=[{"role": "user", "content": prompt}])
    return "".join(getattr(b, "text", "") for b in msg.content).strip()


def parse_tickers(text: str) -> List[str]:
    """Pull a JSON list of tickers out of a model reply; fall back to scanning for ticker-like tokens."""
    m = re.search(r"\[.*?\]", text, flags=re.S)
    if m:
        try:
            vals = json.loads(m.group(0))
            return [str(v).strip().upper() for v in vals if str(v).strip()][:10]
        except json.JSONDecodeError:
            pass
    return re.findall(r"\b[A-Z]{1,5}(?:\.[A-Z]{1,3})?\b", text)[:10]


def suggest_peers(api_key: str, model: str, company: Dict[str, str]) -> List[str]:
    prompt = (f"Suggest 6 listed comparable companies for trading-multiple valuation of {company['name']} ({company['ticker']}), "
              f"sector '{company['sector']}', industry '{company['industry']}'. Prefer companies with similar business models, "
              f"listed on major exchanges with Yahoo Finance tickers (use exchange suffixes such as .DE or .L where needed). "
              f"Reply with only a JSON array of ticker strings, nothing else.")
    return [t for t in parse_tickers(ask(api_key, model, prompt, max_tokens=200)) if t != company["ticker"]]


def review_assumptions(api_key: str, model: str, context: Dict[str, object]) -> str:
    prompt = ("Review these residual income valuation assumptions against the company's own history. For each assumption "
              "(revenue growth, net margin, payout, cost of equity, terminal growth, forecast period) say in one line whether it looks "
              "conservative, reasonable or aggressive and why, citing the numbers. Then give the two assumptions that matter most for the "
              "value and a suggested range for each. Use short markdown bullets.\n\nDATA (JSON):\n" + json.dumps(context, default=str, indent=1))
    return ask(api_key, model, prompt, max_tokens=900)


def analyst_note(api_key: str, model: str, context: Dict[str, object]) -> str:
    prompt = ("Write a one-page equity research note (about 300 words, markdown) on this company from the valuation results below. "
              "Sections: **Verdict** (value vs price, one sentence), **What drives the value**, **Comparables view**, **Key risks to the "
              "assumptions**, **What would change our mind**. Quote the figures given; do not add outside facts or data. End with: "
              "'Educational analysis, not investment advice.'\n\nRESULTS (JSON):\n" + json.dumps(context, default=str, indent=1))
    return ask(api_key, model, prompt, max_tokens=1100)


def explain_question(api_key: str, model: str, question: str, context: Dict[str, object]) -> str:
    prompt = ("Answer the user's question about this valuation using only the data below; show the arithmetic when it helps. "
              "Keep it under 200 words.\n\nQUESTION: " + question + "\n\nDATA (JSON):\n" + json.dumps(context, default=str, indent=1))
    return ask(api_key, model, prompt, max_tokens=700)
ai = _namespace("evs_ai", _before)

del _before

# ============================================================================
# 4. STREAMLIT INTERFACE  (formerly app.py)
# ============================================================================
import html
import io
import os
from dataclasses import replace

import altair as alt
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Equity Valuation Studio", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# ------------------------------------------------------------------ design tokens
INK, SLATE, RULE, PAPER = "#0E1A2B", "#5A6878", "#D8DEE5", "#F3F5F7"
ACCENT, ACCENT_SOFT, MARKET = "#1D4E89", "#7D9CC0", "#A7B2BF"
PIN, UP, DOWN = "#B7791F", "#1E7B4F", "#B3261E"

DEFAULT_PEERS = {"NKE": "ADDYY, DECK, LULU, ONON, VFC", "AAPL": "MSFT, GOOGL, SONY, DELL, HPQ", "MSFT": "AAPL, GOOGL, ORCL, CRM, ADBE",
                 "ADS.DE": "PUM.DE, NKE, DECK, ONON, ASC.L"}
ss = st.session_state

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@75..125,400..800&family=Source+Serif+4:opsz,wght@8..60,500..700&display=swap');
:root {{ --ink:{INK}; --slate:{SLATE}; --rule:{RULE}; --paper:{PAPER}; --accent:{ACCENT}; --pin:{PIN}; --up:{UP}; --down:{DOWN}; }}
html, body, .stApp, .stMarkdown, [data-testid="stSidebar"], button, input, textarea, select, label {{
  font-family: 'Archivo', system-ui, sans-serif; font-variant-numeric: tabular-nums lining-nums; }}
.stApp {{ background: var(--paper); color: var(--ink); }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 4rem; max-width: 1380px; }}
h1, h2, h3, h4 {{ color: var(--ink); font-family: 'Archivo', sans-serif; letter-spacing: -0.01em; }}
h3 {{ font-size: 1.08rem !important; font-weight: 650 !important; padding-top: .6rem !important; }}
[data-testid="stCaptionContainer"], .stCaption {{ color: var(--slate); }}

/* sidebar: the control panel */
[data-testid="stSidebar"] {{ background: #E7EBF0; border-right: 1px solid var(--rule); }}
[data-testid="stSidebar"] h3 {{ font-size: .95rem !important; margin-top: .4rem; }}
.evs-brand {{ display:flex; align-items:center; gap:.6rem; margin: -.4rem 0 1rem; }}
.evs-brand svg {{ flex: none; }}
.evs-brand b {{ font-size: 1.02rem; font-weight: 750; font-stretch: 112%; color: var(--ink); line-height: 1.1; }}
.evs-brand span {{ display:block; font-size: .78rem; color: var(--slate); font-weight: 450; }}
.evs-status {{ display:flex; gap:.5rem; align-items:flex-start; font-size:.82rem; color: var(--ink); line-height:1.35;
  padding:.55rem .7rem; border-radius:6px; background:#fff; border:1px solid var(--rule); margin:.2rem 0 .5rem; }}
.evs-dot {{ width:.55rem; height:.55rem; border-radius:50%; margin-top:.3rem; flex:none; }}
.evs-dot.live {{ background: var(--up); box-shadow: 0 0 0 3px rgba(30,123,79,.15); }}
.evs-dot.stale {{ background: var(--pin); box-shadow: 0 0 0 3px rgba(183,121,31,.18); }}
.evs-dot.sample {{ background: var(--slate); }}

/* buttons */
.stButton > button, [data-testid="stFormSubmitButton"] > button, [data-testid="stDownloadButton"] > button {{
  border-radius: 6px; font-weight: 600; border: 1px solid var(--rule); }}
[data-testid="stFormSubmitButton"] > button, .stButton > button[kind="primary"] {{
  background: var(--accent) !important; color: #fff !important; border-color: var(--accent) !important; width: 100%; }}

/* tabs as a quiet text rail */
.stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid var(--rule); }}
.stTabs [data-baseweb="tab"] {{ padding: .7rem 0 .6rem; color: var(--slate); font-weight: 560; background: transparent; }}
.stTabs [aria-selected="true"] {{ color: var(--ink); }}
.stTabs [data-baseweb="tab-highlight"] {{ background: var(--accent); height: 3px; }}
.stTabs [data-baseweb="tab-border"] {{ display: none; }}
[data-testid="stDataFrame"] {{ border: 1px solid var(--rule); border-radius: 6px; background: #fff; }}
[data-testid="stAlert"] {{ border-radius: 6px; }}

/* hero */
.evs-hero {{ background:#fff; border:1px solid var(--rule); border-radius:10px; padding: 1.5rem 1.75rem 1.25rem; margin-bottom: 1.1rem; }}
.evs-top {{ display:flex; flex-wrap:wrap; justify-content:space-between; gap:.75rem 2rem; align-items:baseline; }}
.evs-name {{ font-size: 1.85rem; font-weight: 780; font-stretch: 108%; letter-spacing:-.02em; margin:0; line-height:1.1; color: var(--ink); }}
.evs-name small {{ font-size: 1rem; font-weight: 600; color: var(--slate); font-stretch: 100%; margin-left:.5rem; letter-spacing:0; }}
.evs-sub {{ color: var(--slate); font-size: .9rem; margin-top:.3rem; }}
.evs-pill {{ display:inline-flex; gap:.45rem; align-items:center; font-size:.8rem; color: var(--ink); padding:.3rem .65rem;
  border:1px solid var(--rule); border-radius: 999px; white-space: nowrap; }}
.evs-verdict {{ font-family:'Source Serif 4', Georgia, serif; font-size: 1.42rem; line-height: 1.38; font-weight: 520; color: var(--ink);
  margin: 1.1rem 0 1.2rem; max-width: 62ch; }}
.evs-verdict b {{ font-weight: 680; }}
.evs-figs {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); border-top:1px solid var(--rule); border-bottom:1px solid var(--rule); }}
.evs-fig {{ padding: .8rem 1rem .75rem 0; }}
.evs-fig + .evs-fig {{ border-left: 1px solid var(--rule); padding-left: 1rem; }}
.evs-fig .l {{ font-size:.78rem; color: var(--slate); }}
.evs-fig .v {{ font-size: 1.55rem; font-weight: 720; font-stretch: 95%; letter-spacing:-.01em; color: var(--ink); line-height:1.25; }}
.evs-fig .v i {{ font-style: normal; font-size: .8rem; font-weight: 550; color: var(--slate); margin-left:.25rem; }}
.evs-fig .d {{ font-size:.8rem; font-weight: 600; }}
.up {{ color: var(--up); }} .down {{ color: var(--down); }} .flat {{ color: var(--slate); }}

/* the valuation ruler (football field) */
.evs-ff {{ margin-top: 1.2rem; }}
.evs-ff-row {{ display:grid; grid-template-columns: 210px 1fr; align-items:center; gap: 1rem; min-height: 40px; }}
.evs-ff-row .rng {{ display:none; }}
.evs-ff-row .m {{ font-size:.82rem; color: var(--ink); line-height:1.2; }}
.evs-ff-row .m span {{ display:block; color: var(--slate); font-size:.74rem; }}
.evs-track {{ position:relative; height: 30px; }}
.evs-track::before {{ content:""; position:absolute; left:0; right:0; top:50%; border-top:1px dashed #E1E6EC; }}
.evs-band {{ position:absolute; top: 8px; height: 14px; border-radius: 3px; }}
.evs-band.Intrinsic {{ background: var(--accent); }}
.evs-band.Trading {{ background: {ACCENT_SOFT}; }}
.evs-band.Market {{ background: {MARKET}; }}
.evs-mid {{ position:absolute; top: 5px; height: 20px; width: 3px; margin-left:-1.5px; background: #fff; border:1px solid var(--ink); border-radius:1px; }}
.evs-num {{ position:absolute; top: 7px; font-size: .72rem; color: var(--slate); white-space:nowrap; }}
.evs-pin {{ position:absolute; top:-6px; bottom: -4px; width: 0; border-left: 2px solid var(--pin); z-index: 2; pointer-events:none; }}
.evs-pin-label {{ position:absolute; top:-1.45rem; transform: translateX(-50%); font-size:.75rem; font-weight: 700; color: var(--pin); white-space: nowrap; }}
.evs-ff-body {{ position: relative; padding-top: 1.4rem; }}
.evs-pinwrap {{ position:absolute; top: 1.4rem; bottom: 0; left: calc(210px + 1rem); right: 0; pointer-events:none; }}
.evs-axis {{ display:grid; grid-template-columns: 210px 1fr; gap:1rem; }}
.evs-axis .t {{ position:relative; height: 1.2rem; border-top: 1px solid var(--rule); }}
.evs-axis .t span {{ position:absolute; top: .15rem; transform: translateX(-50%); font-size:.72rem; color: var(--slate); }}
.evs-legend {{ display:flex; flex-wrap:wrap; gap: 1.1rem; font-size:.76rem; color: var(--slate); margin-top:.6rem; }}
.evs-legend i {{ display:inline-block; width: .8rem; height: .55rem; border-radius:2px; margin-right:.35rem; vertical-align: 0; }}

@media (max-width: 760px) {{
  .evs-hero {{ padding: 1.1rem 1rem; }}
  .evs-name {{ font-size: 1.45rem; }}
  .evs-verdict {{ font-size: 1.15rem; }}
  .evs-ff-row, .evs-axis {{ grid-template-columns: 1fr; gap: 0; }}
  .evs-ff-row {{ margin-bottom: .35rem; }}
  .evs-ff-row .m {{ position:relative; z-index:3; background:#fff; width:fit-content; padding-right:.35rem; }}
  .evs-axis > div:first-child {{ display:none; }}
  .evs-pinwrap {{ left: 0; }}
  .evs-num {{ display:none; }}
  .evs-ff-row .rng {{ display:inline; color: var(--ink); font-size:.74rem; margin-left:.35rem; }}
  .evs-figs {{ grid-template-columns: 1fr 1fr; }}
  .evs-fig, .evs-fig + .evs-fig {{ border-left: 0; padding: .7rem .5rem .6rem 0; }}
  .evs-fig .v {{ font-size: 1.3rem; }}
}}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ formatting
def pct(x, d=1):
    return "—" if x is None else f"{x * 100:.{d}f}%"


def money(x, d=0, cur=""):
    if x is None:
        return "—"
    s = f"{abs(x):,.{d}f}"
    return f"({cur}{s})" if x < 0 else f"{cur}{s}"


def big(x_m, cur=""):
    """Millions to a short label: 34,100 -> 34.1bn."""
    if x_m is None:
        return "—"
    a_ = abs(x_m)
    s = f"{a_ / 1e6:,.2f}tn" if a_ >= 1e6 else f"{a_ / 1e3:,.1f}bn" if a_ >= 1e3 else f"{a_:,.0f}m"
    return ("−" if x_m < 0 else "") + s + (f" {cur}" if cur else "")


def mult(x):
    return "n/m" if x is None else f"{x:.1f}×"


def esc(x):
    return html.escape(str(x))


def themed(ch):
    return (ch.configure(font="Archivo", background="#FFFFFF", padding={"left": 8, "right": 8, "top": 10, "bottom": 6})
            .configure_axis(labelColor=SLATE, titleColor=SLATE, gridColor="#E9EDF2", domainColor=RULE, tickColor=RULE,
                            labelFontSize=11, titleFontSize=11, titleFontWeight=500, labelFont="Archivo", titleFont="Archivo")
            .configure_view(stroke=None)
            .configure_legend(labelColor=INK, titleColor=SLATE, labelFontSize=11, labelFont="Archivo", symbolType="square")
            .configure_title(font="Archivo", fontSize=13, color=INK, anchor="start", fontWeight=600, offset=10))


def show_chart(target, ch):
    target.altair_chart(themed(ch), theme=None)


# ------------------------------------------------------------------ live data (cached; failures are never cached)
@st.cache_data(ttl=900, show_spinner=False)
def load_company(ticker: str) -> MK.Company:
    return MK.fetch(ticker)


def load_peers(tickers: tuple):
    rows, errors = [], []
    for t in tickers:
        try:
            rows.append(load_company(t).comps_row())
        except Exception as exc:
            errors.append(f"{t}: {exc}")
    return rows, errors


@st.cache_data(ttl=3600, show_spinner=False)
def load_rf():
    return MK.fetch_risk_free()


@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(ticker: str):
    return MK.fetch_price_history(ticker)


# ================================================================== sidebar: company
st.sidebar.markdown(
    "<div class='evs-brand'><svg width='30' height='30' viewBox='0 0 30 30' aria-hidden='true'>"
    f"<rect width='30' height='30' rx='6' fill='{INK}'/><rect x='6' y='17' width='4' height='7' fill='{ACCENT_SOFT}'/>"
    f"<rect x='13' y='11' width='4' height='13' fill='#fff'/><rect x='20' y='6' width='4' height='18' fill='{PIN}'/></svg>"
    "<div><b>Equity Valuation Studio</b><span>Residual income and trading comps</span></div></div>", unsafe_allow_html=True)
with st.sidebar.form("load"):
    ticker_in = st.text_input("Stock ticker", value=ss.get("ticker", "NKE"), help="Yahoo Finance symbol, e.g. NKE, AAPL, MSFT, ADS.DE, 7203.T")
    peers_in = st.text_input("Comparable companies", value=ss.get("peers_txt", DEFAULT_PEERS.get(ss.get("ticker", "NKE"), "")),
                             help="Comma-separated Yahoo symbols. Leave empty to skip comps, or let the AI analyst suggest peers.")
    mode = st.radio("Data", ["Live Yahoo Finance", "Offline demo (Nike FY2026)"], index=0 if ss.get("mode", "live") == "live" else 1,
                    help="Live is the default. The offline demo is a fixed snapshot for when Yahoo can't be reached.")
    if st.form_submit_button("Value company", type="primary"):
        new_ticker = ticker_in.strip().upper() or "NKE"
        if new_ticker != ss.get("ticker") and peers_in == ss.get("peers_txt", DEFAULT_PEERS.get(ss.get("ticker", "NKE"), "")):
            peers_in = DEFAULT_PEERS.get(new_ticker, "")  # don't carry the old company's peers over
        ss["ticker"] = new_ticker
        ss["peers_txt"] = peers_in
        ss["mode"] = "live" if mode.startswith("Live") else "sample"

ticker = ss.get("ticker", "NKE")
peer_list = [p.strip().upper() for p in ss.get("peers_txt", DEFAULT_PEERS.get(ticker, "")).split(",") if p.strip() and p.strip().upper() != ticker]

# ------------------------------------------------------------------ load, with a last-good fallback
co, load_error, stale = None, None, False
if ss.get("mode", "live") == "sample":
    co = MK.sample_nike()
else:
    with st.spinner(f"Getting live Yahoo Finance data for {ticker}…"):
        try:
            co = load_company(ticker)
            ss.setdefault("last_good", {})[ticker] = co
        except Exception as exc:
            load_error = str(exc)
            co = (ss.get("last_good") or {}).get(ticker)
            stale = co is not None

status_box = st.sidebar.container()
if co is None:
    status_box.markdown(f"<div class='evs-status'><span class='evs-dot stale'></span><div>Yahoo Finance didn't return data for <b>{esc(ticker)}</b>.</div></div>",
                        unsafe_allow_html=True)
    if status_box.button("Refresh live data", key="refresh_err"):
        load_company.clear(); load_rf.clear(); load_prices.clear()
        st.rerun()
    st.title("Equity Valuation Studio")
    st.error(f"Couldn't load **{ticker}** from Yahoo Finance. {load_error or ''}")
    st.markdown("**What to try**\n\n- Press **Refresh live data** in the sidebar. Yahoo sometimes refuses a request from cloud servers and accepts the next one.\n"
                "- Check the symbol on finance.yahoo.com. Non-US listings need a suffix, e.g. `ADS.DE`, `7203.T`, `NESN.SW`.\n"
                "- To look around while Yahoo is unavailable, switch **Data** to the offline demo.")
    st.stop()

if co.mode == "sample":
    status_box.markdown("<div class='evs-status'><span class='evs-dot sample'></span><div><b>Offline demo.</b> Fixed Nike FY2026 snapshot, "
                        "not live prices.</div></div>", unsafe_allow_html=True)
elif stale:
    status_box.markdown(f"<div class='evs-status'><span class='evs-dot stale'></span><div><b>Yahoo didn't respond.</b> Showing the last good data, "
                        f"retrieved {esc(co.retrieved_at)}.</div></div>", unsafe_allow_html=True)
else:
    status_box.markdown(f"<div class='evs-status'><span class='evs-dot live'></span><div><b>Live Yahoo Finance data</b><br>Retrieved {esc(co.retrieved_at)}. "
                        "Refreshes every 15 minutes.</div></div>", unsafe_allow_html=True)
if co.mode == "live" and status_box.button("Refresh live data", key="refresh"):
    load_company.clear(); load_rf.clear(); load_prices.clear()
    st.rerun()

cur = co.currency
key = f"{co.ticker}_{co.mode}"  # widget keys reset when the company changes

# ================================================================== defaults from history
rev_hist, ni_hist, div_hist = co.series("revenue"), co.series("net_income"), co.series("dividends")
hd = E.historical_defaults(rev_hist, ni_hist, [d if d is not None else (0.0 if "dividends" not in co.history else None) for d in div_hist])
def_g = min(max(hd["revenue_growth"] if hd["revenue_growth"] is not None else 0.04, -0.10), 0.20)
def_m = hd["net_margin"] if hd["net_margin"] is not None else 0.08
def_p = hd["payout"] if hd["payout"] is not None else 0.0
rf_live = load_rf() if co.mode == "live" else None
def_rf = rf_live if rf_live else 0.0444
def_beta = co.f("beta") if co.f("beta") else 1.0

# ================================================================== sidebar: assumptions
st.sidebar.subheader("Assumptions")
scen = st.sidebar.radio("Scenario", ["Base", "Bull", "Bear"], horizontal=True, key=f"scen_{key}",
                        help="Bull and bear shift growth, margin and terminal growth from your base inputs (editable under Advanced).")
g_in = st.sidebar.number_input("Revenue growth, % a year", value=round(def_g * 100, 2), step=0.5, format="%.2f", key=f"g_{key}",
                               help=f"Default: {hd['years_of_revenue']}-year historical CAGR" if hd["revenue_growth"] is not None else "No history; default 4%") / 100
m_in = st.sidebar.number_input("Net income margin, %", value=round(def_m * 100, 2), step=0.5, format="%.2f", key=f"m_{key}",
                               help=f"Default: average of the last {hd['years_of_margin']} years" if hd["net_margin"] is not None else "No history; default 8%") / 100
p_in = st.sidebar.number_input("Payout ratio, % of net income", value=round(def_p * 100, 2), step=1.0, min_value=0.0, max_value=150.0, format="%.2f",
                               key=f"p_{key}", help="Default: average dividends paid ÷ net income") / 100
ke_mode = st.sidebar.radio("Cost of equity", ["CAPM", "Enter directly"], horizontal=True, key=f"kem_{key}")
if ke_mode == "CAPM":
    c1, c2, c3 = st.sidebar.columns(3)
    rf = c1.number_input("Rf %", value=round(def_rf * 100, 2), step=0.05, format="%.2f", key=f"rf_{key}") / 100
    beta = c2.number_input("Beta", value=round(def_beta, 2), step=0.05, format="%.2f", key=f"b_{key}")
    erp = c3.number_input("ERP %", value=4.18, step=0.1, format="%.2f", key=f"erp_{key}") / 100
    ke = E.capm(rf, beta, erp)
    st.sidebar.caption(f"kₑ = {rf:.2%} + {beta:.2f} × {erp:.2%} = **{ke:.2%}**. Rf: {'live ^TNX 10-year Treasury yield' if rf_live else 'default 4.44%'}; "
                       f"beta: {'Yahoo' if co.f('beta') else 'default 1.0'}.")
else:
    ke = st.sidebar.number_input("Cost of equity, %", value=9.0, step=0.1, format="%.2f", key=f"ke_{key}") / 100
tg = st.sidebar.number_input("Terminal growth, %", value=2.15, step=0.1, format="%.2f", key=f"tg_{key}") / 100
years = st.sidebar.slider("Forecast period, years", 3, 10, 5, key=f"n_{key}")
with st.sidebar.expander("Advanced"):
    fade = st.checkbox("Fade growth to terminal growth by the final year", value=False, key=f"fade_{key}")
    use_tm = st.checkbox("Glide net margin to a target", value=False, key=f"tmo_{key}")
    tm = st.number_input("Target net margin in final year, %", value=round(def_m * 100 + 2, 2), step=0.5, format="%.2f", key=f"tm_{key}") / 100 if use_tm else None
    st.markdown("**Scenario shifts** (percentage points)")
    s1, s2, s3 = st.columns(3)
    sg = s1.number_input("Growth ±", value=3.0, step=0.5, key=f"sg_{key}") / 100
    sm = s2.number_input("Margin ±", value=2.0, step=0.5, key=f"sm_{key}") / 100
    st_ = s3.number_input("Terminal ±", value=0.5, step=0.25, key=f"st_{key}") / 100
    ri_w = st.slider("Weight on residual income in blended value", 0, 100, 50, 5, key=f"w_{key}") / 100
shifts = {"Bear": {"revenue_growth": -sg, "net_margin": -sm, "terminal_growth": -st_}, "Base": {},
          "Bull": {"revenue_growth": sg, "net_margin": sm, "terminal_growth": st_}}

base_a = E.RIAssumptions(revenue_growth=g_in, net_margin=m_in, payout=p_in, cost_of_equity=ke, terminal_growth=tg, years=years, fade_growth=fade, target_margin=tm)
a = E.scenario(base_a, scen, shifts)

# ================================================================== base year
revenue0 = co.latest("revenue") or co.f("revenue_ttm")
book0 = co.latest("equity") or ((co.f("bvps") or 0) * (co.f("shares") or 0) or None)
shares = co.f("shares")
price = co.f("price")
base_year = co.fiscal_year_end.year if co.fiscal_year_end else (max(co.years()) if co.years() else 0)
missing_core = [n for n, v in (("revenue", revenue0), ("book value", book0), ("shares outstanding", shares)) if not v]
if missing_core:
    st.title(f"{co.name} ({co.ticker})")
    st.error(f"Yahoo Finance did not return {', '.join(missing_core)} for {co.ticker}, so the residual income model can't run. "
             "Press **Refresh live data**, try another ticker, or use the offline demo.")
    st.stop()

try:
    ri = E.residual_income_value(revenue0, book0, shares, a, base_year)
    ri_err = None
except ValueError as exc:
    ri, ri_err = None, str(exc)
ri_bear = ri_bull = None
if ri:
    try:
        ri_bear = E.residual_income_value(revenue0, book0, shares, E.scenario(base_a, "Bear", shifts))["per_share"]
        ri_bull = E.residual_income_value(revenue0, book0, shares, E.scenario(base_a, "Bull", shifts))["per_share"]
    except ValueError:
        ri_bear = ri_bull = None

# ================================================================== comps
peer_rows, peer_errors = ([], [])
if peer_list and co.mode == "live":
    with st.spinner(f"Getting live data for {len(peer_list)} comparable companies…"):
        peer_rows, peer_errors = load_peers(tuple(peer_list))
target_row = co.comps_row()
comps = E.comps_valuation(target_row, peer_rows) if peer_rows else {}
ff = E.football_field(ri_bear if ri_bear is not None else (ri["per_share"] if ri else None), ri["per_share"] if ri else None,
                      ri_bull if ri_bull is not None else (ri["per_share"] if ri else None), comps,
                      co.f("low52"), co.f("high52"), co.f("target_low"), co.f("target_mean"), co.f("target_high"))
blend = E.blended_value(ri["per_share"] if ri else None, comps, ri_w)


# ================================================================== hero
def delta_html(v, ref):
    if v is None or not ref:
        return "<span class='d flat'>&nbsp;</span>"
    g = v / ref - 1
    cls = "up" if g > 0.005 else "down" if g < -0.005 else "flat"
    return f"<span class='d {cls}'>{g:+.1%} vs price</span>"


def verdict_html():
    if not (ri and price):
        return "Enter assumptions in the sidebar to value the company."
    up = ri["per_share"] / price - 1
    word = "upside" if up >= 0 else "downside"
    rng = f" Bear to bull spans {min(ri_bear, ri_bull):,.2f} to {max(ri_bear, ri_bull):,.2f}." if ri_bear is not None else ""
    return (f"On the {scen.lower()} case, residual income values {esc(co.name)} at <b>{ri['per_share']:,.2f} {esc(cur)}</b> a share, "
            f"<b>{abs(up):.0%} {word}</b> to the current {price:,.2f}.{rng}")


def ruler_html(rows, price_now):
    vals = [v for r in rows for v in (r["low"], r["high"]) if v is not None] + ([price_now] if price_now else [])
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.08 or abs(hi) * 0.1 or 1
    lo, hi = lo - pad, hi + pad
    pos = lambda v: (v - lo) / (hi - lo) * 100
    sub = {"Intrinsic": "Your assumptions", "Trading comps": "Peer multiples", "Market": "Yahoo Finance"}
    out = ["<div class='evs-ff'><div class='evs-ff-body'>"]
    for r in rows:
        name = r["method"].split(" (")[0]
        detail = r["method"][len(name):].strip(" ()") or sub.get(r["group"], "")
        l_, h_ = pos(r["low"]), pos(r["high"])
        band = (f"<div class='evs-band {r['group'].split()[0]}' style='left:{l_:.2f}%;width:{max(h_ - l_, .6):.2f}%' "
                f"title='{esc(r['method'])}: {r['low']:,.2f} to {r['high']:,.2f}'></div>")
        mid = f"<div class='evs-mid' style='left:{pos(r['mid']):.2f}%'></div>" if r.get("mid") is not None else ""
        nums = (f"<span class='evs-num' style='right:{100 - l_ + 0.8:.2f}%'>{r['low']:,.2f}</span>"
                f"<span class='evs-num' style='left:{h_ + 0.8:.2f}%'>{r['high']:,.2f}</span>")
        out.append(f"<div class='evs-ff-row'><div class='m'>{esc(name)}<b class='rng'>{r['low']:,.2f} to {r['high']:,.2f}</b><span>{esc(detail)}</span></div><div class='evs-track'>{band}{mid}{nums}</div></div>")
    if price_now:
        out.append(f"<div class='evs-pinwrap'><div class='evs-pin' style='left:{pos(price_now):.2f}%'>"
                   f"<span class='evs-pin-label'>Price {price_now:,.2f}</span></div></div>")
    step = (hi - lo) / 5
    ticks = "".join(f"<span style='left:{pos(lo + step * i):.2f}%'>{lo + step * i:,.0f}</span>" for i in range(1, 5))
    out.append(f"</div><div class='evs-axis'><div></div><div class='t'>{ticks}</div></div>")
    out.append(f"<div class='evs-legend'><span><i style='background:{ACCENT}'></i>Residual income</span>"
               f"<span><i style='background:{ACCENT_SOFT}'></i>Trading comps, 25th to 75th percentile</span>"
               f"<span><i style='background:{MARKET}'></i>Market</span><span><i style='background:#fff;border:1px solid {INK}'></i>Midpoint</span>"
               f"<span><i style='background:{PIN};height:2px;vertical-align:3px'></i>Current price ({esc(cur)})</span></div></div>")
    return "".join(out)


exch = f", {esc(co.sector)}" if co.sector not in ("—", "") else ""
pill_cls = "live" if (co.mode == "live" and not stale) else "stale" if stale else "sample"
pill_txt = {"live": "Live Yahoo Finance", "stale": "Last good Yahoo data", "sample": "Offline demo data"}[pill_cls]
pe = (price / co.f("eps_ttm")) if (price and co.f("eps_ttm") and co.f("eps_ttm") > 0) else None
pb = (price / co.f("bvps")) if (price and co.f("bvps") and co.f("bvps") > 0) else None
figs = [
    ("Share price", f"{price:,.2f}<i>{esc(cur)}</i>" if price else "—", f"<span class='d flat'>52 weeks: {money(co.f('low52'), 2)} to {money(co.f('high52'), 2)}</span>"),
    (f"Residual income value, {scen.lower()}", f"{ri['per_share']:,.2f}<i>{esc(cur)}</i>" if ri else "—", delta_html(ri["per_share"] if ri else None, price)),
    ("Blended value", f"{blend:,.2f}<i>{esc(cur)}</i>" if blend else "—", delta_html(blend, price)),
    ("Market cap", big(co.f("market_cap")), f"<span class='d flat'>EV {big(E.enterprise_value(target_row))}</span>"),
    ("P/E and P/B", f"{mult(pe)} <i>/</i> {mult(pb)}", "<span class='d flat'>Trailing EPS, book value per share</span>"),
    ("Cost of equity", pct(ke, 2), f"<span class='d flat'>Terminal growth {pct(a.terminal_growth, 2)}</span>"),
]
fig_html = "".join(f"<div class='evs-fig'><div class='l'>{l_}</div><div class='v'>{v_}</div>{d_}</div>" for l_, v_, d_ in figs)
st.markdown(
    f"<section class='evs-hero'><div class='evs-top'><div><h1 class='evs-name'>{esc(co.name)}<small>{esc(co.ticker)}</small></h1>"
    f"<div class='evs-sub'>{esc(co.industry)}{exch}. Fiscal year ended {co.fiscal_year_end or '—'}; amounts in {esc(co.financial_currency)}, "
    f"per-share figures in {esc(cur)}.</div></div>"
    f"<span class='evs-pill'><span class='evs-dot {pill_cls}'></span>{pill_txt}, {esc(co.retrieved_at)}</span></div>"
    f"<p class='evs-verdict'>{verdict_html()}</p><div class='evs-figs'>{fig_html}</div>{ruler_html(ff, price)}</section>",
    unsafe_allow_html=True)

if stale:
    st.warning(f"Yahoo Finance didn't respond just now ({load_error}). The figures are from the last successful download.")
for n in co.notes:
    st.info(n)
if ri_err:
    st.error(f"Residual income model: {ri_err}")

tabs = st.tabs(["Overview", "Residual income", "Trading comps", "History", "Data and checks", "AI analyst", "Method"])


# ================================================================== charts
def forecast_chart(rows):
    df = pd.DataFrame([{"Year": f"FY{str(r['year'])[-2:]}E" if r['year'] > 100 else f"Y{r['t']}", "Net income": r["net_income"],
                        "Capital charge": r["capital_charge"], "Residual income": r["residual_income"]} for r in rows])
    long = df.melt("Year", ["Net income", "Capital charge"], var_name="Series", value_name="Value")
    bars = alt.Chart(long).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=alt.X("Year:N", title=None), xOffset="Series:N", y=alt.Y("Value:Q", title=f"{co.financial_currency} m"),
        color=alt.Color("Series:N", scale=alt.Scale(domain=["Net income", "Capital charge"], range=[ACCENT, MARKET]), legend=alt.Legend(orient="top", title=None)),
        tooltip=["Year", "Series", alt.Tooltip("Value:Q", format=",.0f")])
    line = alt.Chart(df).mark_line(color=PIN, point=alt.OverlayMarkDef(color=PIN, size=50), strokeWidth=2.5).encode(
        x="Year:N", y="Residual income:Q", tooltip=[alt.Tooltip("Residual income:Q", format=",.0f")])
    return alt.layer(bars, line).properties(height=300)


def heat(grid, xs, ys, xlab, ylab, price_now):
    recs = [{"x": xl, "y": yl, "v": v, "lab": "n/m" if v is None else f"{v:,.2f}", "vs": None if (v is None or not price_now) else v / price_now - 1}
            for yl, row in zip(ys, grid) for xl, v in zip(xs, row)]
    df = pd.DataFrame(recs)
    base = alt.Chart(df).encode(x=alt.X("x:O", title=xlab, sort=xs, axis=alt.Axis(labelAngle=0)), y=alt.Y("y:O", title=ylab, sort=ys))
    rect = base.mark_rect(stroke="#fff", strokeWidth=2).encode(
        color=alt.Color("vs:Q", scale=alt.Scale(domain=[-0.5, 0, 0.5], range=[DOWN, "#F4F5F2", UP], clamp=True),
                        legend=alt.Legend(title="vs price", format="+%", orient="right")))
    txt = base.mark_text(fontSize=12, fontWeight=500, color=INK).encode(text="lab:N")
    return alt.layer(rect, txt).properties(height=240)


# ================================================================== overview
with tabs[0]:
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Valuation summary")
        tbl = pd.DataFrame([{"Method": r["method"], "Low": r["low"], "Mid": r["mid"], "High": r["high"],
                             "Mid vs price": (r["mid"] / price - 1) if (r.get("mid") is not None and price) else None} for r in ff])
        if not tbl.empty:
            st.dataframe(tbl.style.format({"Low": "{:,.2f}", "Mid": lambda v: "—" if pd.isna(v) else f"{v:,.2f}", "High": "{:,.2f}",
                                           "Mid vs price": lambda v: "—" if v is None or pd.isna(v) else f"{v:+.1%}"}), hide_index=True)
        st.caption("Per share, in the trading currency. Comps apply the 25th, 50th and 75th percentile peer multiples to the company's own figures.")
    with right:
        st.subheader("What the numbers say")
        if comps:
            meds = {m_: c["implied"]["median"] for m_, c in comps.items() if c.get("implied")}
            if meds:
                st.markdown("Peer medians imply " + ", ".join(f"**{v:,.2f}** on {m_}" for m_, v in meds.items()) + ".")
        elif co.mode == "live":
            st.info("Add comparable companies in the sidebar to see trading-comps values.")
        if ri:
            tvs = ri["pv_tv"] / ri["equity_value"]
            st.markdown(f"Book value today is **{ri['parts']['Book value today']:,.2f}** a share. Forecast residual income adds "
                        f"**{ri['parts']['PV of residual income']:,.2f}** and the terminal value **{ri['parts']['PV of terminal value']:,.2f}** "
                        f"({tvs:.0%} of the value).")
            final_roe = ri["rows"][-1]["roe_begin"]
            st.markdown(f"Return on beginning equity reaches **{final_roe:.1%}** in the final forecast year against a cost of equity of **{ke:.2%}**, "
                        f"so residual income is {'positive' if final_roe > ke else 'negative'} at the horizon.")
    if co.mode == "live":
        ph = load_prices(co.ticker)
        if ph is not None and ri:
            st.subheader("Share price over two years, against the valuation")
            ph = ph.rename(columns={ph.columns[0]: "Date"})
            ref = pd.DataFrame({"y": [ri["per_share"]] + ([blend] if blend else []), "label": ["Residual income value"] + (["Blended value"] if blend else [])})
            line = alt.Chart(ph).mark_line(color=INK, strokeWidth=1.6).encode(x=alt.X("Date:T", title=None), y=alt.Y("Close:Q", title=cur, scale=alt.Scale(zero=False)),
                                                                              tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Close:Q", format=",.2f")])
            rules = alt.Chart(ref).mark_rule(strokeDash=[6, 3], strokeWidth=2).encode(
                y="y:Q", color=alt.Color("label:N", scale=alt.Scale(range=[ACCENT, PIN]), legend=alt.Legend(orient="top", title=None)))
            show_chart(st, alt.layer(line, rules).properties(height=280))

# ================================================================== residual income
with tabs[1]:
    if not ri:
        st.error(ri_err)
    else:
        c1, c2 = st.columns([2, 3])
        with c1:
            st.subheader("Value build")
            parts = pd.DataFrame({"Component": list(ri["parts"]), "Per share": list(ri["parts"].values()),
                                  f"{co.financial_currency} m": [ri["book0"], ri["pv_ri"], ri["pv_tv"]]})
            parts["Share"] = parts["Per share"] / ri["per_share"]
            st.dataframe(parts.style.format({"Per share": "{:,.2f}", f"{co.financial_currency} m": "{:,.0f}", "Share": "{:.0%}"}), hide_index=True)
            st.markdown(f"**Equity value** {ri['equity_value']:,.0f}m ÷ **{shares:,.1f}m** shares = **{ri['per_share']:,.2f} {cur}**")
            if ri["pv_tv"] / ri["equity_value"] > 0.75:
                st.warning(f"The terminal value is {ri['pv_tv'] / ri['equity_value']:.0%} of equity value; the result is very sensitive to terminal growth and the cost of equity.")
            st.subheader("Assumptions in use")
            gp, mp = E.growth_path(a), E.margin_path(a)
            st.dataframe(pd.DataFrame({"Assumption": ["Scenario", "Revenue growth (yr 1 → final)", "Net income margin (yr 1 → final)", "Payout ratio",
                                                      "Cost of equity", "Terminal growth", "Forecast period"],
                                       "Value": [scen, f"{gp[0]:.2%} → {gp[-1]:.2%}", f"{mp[0]:.2%} → {mp[-1]:.2%}", pct(a.payout, 1), pct(a.cost_of_equity, 2),
                                                 pct(a.terminal_growth, 2), f"{a.years} years"],
                                       "History": [ "—", pct(hd["revenue_growth"]) + " CAGR" if hd["revenue_growth"] is not None else "n/a",
                                                    pct(hd["net_margin"]) + " avg" if hd["net_margin"] is not None else "n/a",
                                                    pct(hd["payout"]) + " avg" if hd["payout"] is not None else "n/a", "—", "—", "—"]}),
                         hide_index=True)
        with c2:
            st.subheader("Net income vs capital charge")
            show_chart(st, forecast_chart(ri["rows"]))
        st.subheader("Forecast")
        fc = pd.DataFrame([{"Year": r["year"], "Revenue": r["revenue"], "Growth": r["growth"], "Net margin": r["net_margin"], "Net income": r["net_income"],
                            "Dividends": r["dividends"], "Beginning book value": r["bv_begin"], "Ending book value": r["bv_end"], "ROE (beginning)": r["roe_begin"],
                            "Capital charge": r["capital_charge"], "Residual income": r["residual_income"], "Discount factor": r["discount_factor"], "PV of RI": r["pv_ri"]}
                           for r in ri["rows"]]).set_index("Year").T
        fmt_rows = {"Growth": "{:.1%}", "Net margin": "{:.1%}", "ROE (beginning)": "{:.1%}", "Discount factor": "{:.4f}"}
        st.dataframe(fc.apply(lambda row: row.map((lambda v, f=fmt_rows.get(row.name, "{:,.0f}"): "—" if v is None or pd.isna(v) else f.format(v))), axis=1))
        st.caption(f"{co.financial_currency} millions. Base year: fiscal {base_year} revenue {revenue0:,.0f}m and book value {book0:,.0f}m. "
                   "Terminal value = final-year residual income × (1 + g) ÷ (kₑ − g).")
        st.subheader("Sensitivity")
        s1c, s2c = st.columns(2)
        kes = [a.cost_of_equity + d for d in (-0.01, -0.005, 0, 0.005, 0.01)]
        gs = [a.terminal_growth + d for d in (-0.01, -0.005, 0, 0.005, 0.01)]
        g1 = E.sensitivity_grid(revenue0, book0, shares, a, "terminal_growth", gs, "cost_of_equity", kes)
        show_chart(s1c, heat(g1, [f"{x:.2%}" for x in gs], [f"{y:.2%}" for y in kes], "Terminal growth", "Cost of equity", price))
        grs = [a.revenue_growth + d for d in (-0.04, -0.02, 0, 0.02, 0.04)]
        mgs = [a.net_margin + d for d in (-0.02, -0.01, 0, 0.01, 0.02)]
        g2 = E.sensitivity_grid(revenue0, book0, shares, replace(a, target_margin=None), "revenue_growth", grs, "net_margin", mgs)
        show_chart(s2c, heat(g2, [f"{x:.1%}" for x in grs], [f"{y:.1%}" for y in mgs], "Revenue growth", "Net margin", price))
        st.caption("Cells are value per share; colour shows the gap to the current price (blue above, red below).")

# ================================================================== comps
with tabs[2]:
    if co.mode != "live":
        st.info("Trading comparables need live data. Switch **Data** in the sidebar to **Live Yahoo Finance**.")
    elif not peer_list:
        st.info("Enter comparable tickers in the sidebar (comma-separated), or use **Suggest peers** in the AI analyst tab.")
    else:
        for e_ in peer_errors:
            st.warning(f"Skipped {e_}")
        rows = []
        for p in [target_row] + peer_rows:
            mm = E.multiples(p)
            rows.append({"Ticker": p["ticker"], "Company": p.get("name", ""), "Currency": p.get("currency", ""), "Price": p.get("price"),
                         "Market cap (m)": p.get("market_cap"), "EV (m)": E.enterprise_value(p), **mm, "Role": "Target" if p is target_row else "Peer"})
        cdf = pd.DataFrame(rows)
        st.subheader("Peer multiples")
        st.dataframe(cdf.style.format({"Price": lambda v: "—" if pd.isna(v) else f"{v:,.2f}", "Market cap (m)": lambda v: "—" if pd.isna(v) else f"{v:,.0f}",
                                       "EV (m)": lambda v: "—" if pd.isna(v) else f"{v:,.0f}", **{m_: (lambda v: "n/m" if v is None or pd.isna(v) else f"{v:.1f}×") for m_ in E.MULTIPLES}}),
                     hide_index=True)
        mixed = {p.get("currency") for p in peer_rows} - {cur}
        if mixed:
            st.warning(f"Some peers trade in {', '.join(sorted(mixed))}. Multiples are currency-neutral ratios, but check that each peer's price and statements share a currency.")
        st.caption("TTM figures from Yahoo. EV = enterprise value (Yahoo, or market cap + debt − cash). n/m = not meaningful (negative or missing denominator); excluded from statistics. "
                   "Yahoo's total debt can include lease liabilities.")
        st.subheader("Implied value for " + co.ticker)
        rows2 = []
        for m_ in E.MULTIPLES:
            c = comps.get(m_, {})
            s_ = c.get("stats")
            imp = c.get("implied") or {}
            rows2.append({"Multiple": m_, "Peers used": s_["n"] if s_ else 0, "25th pct": s_["low"] if s_ else None, "Median": s_["median"] if s_ else None,
                          "75th pct": s_["high"] if s_ else None, "Target metric": {"EV/Revenue": target_row["revenue"], "EV/EBITDA": target_row["ebitda"],
                                                                                    "P/E": target_row["eps"], "P/Book": target_row["bvps"]}[m_],
                          "Implied low": imp.get("low"), "Implied median": imp.get("median"), "Implied high": imp.get("high"),
                          "vs price": (imp["median"] / price - 1) if (imp.get("median") and price) else None, "Note": c.get("reason", "")})
        idf = pd.DataFrame(rows2)
        f2 = lambda v: "—" if v is None or pd.isna(v) else f"{v:,.2f}"
        st.dataframe(idf.style.format({"25th pct": lambda v: mult(None if pd.isna(v) else v), "Median": lambda v: mult(None if pd.isna(v) else v),
                                       "75th pct": lambda v: mult(None if pd.isna(v) else v), "Target metric": f2, "Implied low": f2, "Implied median": f2,
                                       "Implied high": f2, "vs price": lambda v: "—" if v is None or pd.isna(v) else f"{v:+.1%}"}),
                     hide_index=True)
        st.caption(f"EV multiples: implied EV − net debt ({money((co.f('total_debt') or 0) - (co.f('total_cash') or 0))}m) = equity, ÷ {shares:,.1f}m shares. "
                   "P/E uses trailing EPS; P/Book uses book value per share.")

# ================================================================== history
with tabs[3]:
    ys = co.years()
    if not ys:
        st.info("No annual statements returned.")
    else:
        hdf = pd.DataFrame({"Fiscal year": ys, "Revenue": co.series("revenue"), "Net income": co.series("net_income"),
                            "Shareholders' equity": co.series("equity"), "Dividends paid": co.series("dividends"), "Buybacks": co.series("buybacks"),
                            "EBITDA": co.series("ebitda"), "Diluted EPS": co.series("diluted_eps")})
        hdf["Revenue growth"] = hdf["Revenue"].pct_change()
        hdf["Net margin"] = hdf["Net income"] / hdf["Revenue"]
        hdf["ROE (ending equity)"] = hdf["Net income"] / hdf["Shareholders' equity"]
        hdf["Payout"] = hdf["Dividends paid"] / hdf["Net income"]
        c1, c2 = st.columns(2)
        rb = alt.Chart(hdf).mark_bar(color=ACCENT, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(x=alt.X("Fiscal year:O"), y=alt.Y("Revenue:Q", title=None),
                                                         tooltip=[alt.Tooltip("Revenue:Q", format=",.0f")])
        show_chart(c1, rb.properties(height=260, title=f"Revenue, {co.financial_currency} m"))
        ml = hdf.melt("Fiscal year", ["Net margin", "ROE (ending equity)"], var_name="Ratio", value_name="Value")
        show_chart(c2, alt.Chart(ml).mark_line(point=True, strokeWidth=2).encode(
            x="Fiscal year:O", y=alt.Y("Value:Q", title=None, axis=alt.Axis(format="%")),
            color=alt.Color("Ratio:N", scale=alt.Scale(range=[ACCENT, PIN]), legend=alt.Legend(orient="top", title=None)),
            tooltip=["Fiscal year:O", "Ratio:N", alt.Tooltip("Value:Q", format=".1%")]).properties(height=260, title="Net margin and ROE"))
        pf = {"Revenue growth": "{:.1%}", "Net margin": "{:.1%}", "ROE (ending equity)": "{:.1%}", "Payout": "{:.1%}", "Diluted EPS": "{:.2f}"}
        show = hdf.set_index("Fiscal year").T
        st.dataframe(show.apply(lambda row: row.map((lambda v, f=pf.get(row.name, "{:,.0f}"): "—" if v is None or pd.isna(v) else f.format(v))), axis=1))
        st.caption(f"{co.financial_currency} millions except EPS. Source: Yahoo Finance annual statements (usually the last four fiscal years).")

# ================================================================== data & quality
with tabs[4]:
    checks = []
    def chk(name, ok, detail, testable=True):
        checks.append({"Check": name, "Status": ("PASS" if ok else "FLAG") if testable else "NOT TESTABLE", "Detail": detail})
    chk("Core inputs available", not missing_core, "Revenue, book value and shares present" if not missing_core else ", ".join(missing_core))
    chk("Cost of equity above terminal growth", ke > a.terminal_growth, f"{ke:.2%} vs {a.terminal_growth:.2%}")
    chk("Positive book value", (book0 or 0) > 0, f"{book0:,.0f}m" if book0 else "missing")
    chk("Currency consistent", co.currency == co.financial_currency, f"Trading {co.currency}, statements {co.financial_currency}")
    if hd["net_margin"] is not None:
        chk("Margin within 5 points of history", abs(a.net_margin - hd["net_margin"]) <= 0.05, f"{a.net_margin:.1%} vs {hd['net_margin']:.1%} average")
    if hd["revenue_growth"] is not None:
        chk("Growth within 10 points of history", abs(a.revenue_growth - hd["revenue_growth"]) <= 0.10, f"{a.revenue_growth:.1%} vs {hd['revenue_growth']:.1%} CAGR")
    if ri:
        tvs = ri["pv_tv"] / ri["equity_value"]
        chk("Terminal value below 75% of value", tvs <= 0.75, f"{tvs:.0%} of equity value")
    if co.mode == "live" and peer_list:
        n_ok = len(peer_rows)
        chk("At least three peers loaded", n_ok >= 3, f"{n_ok} of {len(peer_list)} loaded")
    chk("Live data", co.mode == "live" and not stale, ("Yahoo Finance, retrieved " + co.retrieved_at) if co.mode == "live" else "Offline demo snapshot")
    st.subheader("Checks")
    st.dataframe(pd.DataFrame(checks), hide_index=True)
    st.subheader("Where each input comes from")
    labels = {"price": "Share price", "shares": "Shares outstanding (m)", "market_cap": "Market cap (m)", "enterprise_value": "Enterprise value (m)",
              "total_debt": "Total debt (m)", "total_cash": "Total cash (m)", "revenue_ttm": "Revenue, TTM (m)", "ebitda_ttm": "EBITDA, TTM (m)",
              "eps_ttm": "EPS, trailing", "bvps": "Book value per share", "beta": "Beta", "high52": "52-week high", "low52": "52-week low",
              "target_low": "Analyst target, low", "target_mean": "Analyst target, mean", "target_high": "Analyst target, high"}
    st.dataframe(pd.DataFrame([{"Item": lab, "Value": "—" if co.f(k_) is None else f"{co.f(k_):,.2f}", "Source": co.sources.get(k_, "")} for k_, lab in labels.items()]),
                 hide_index=True)
    if co.missing:
        st.caption("Not returned by Yahoo: " + ", ".join(co.missing) + ". Missing values are left blank, never set to zero.")

# ================================================================== AI analyst
with tabs[5]:
    st.subheader("AI analyst (Claude)")
    st.caption("Optional. Uses the Anthropic API with your own key; the app sends only the figures shown here. Output is AI-generated — check it before use.")
    env_key = os.environ.get("ANTHROPIC_API_KEY")
    try:
        env_key = env_key or st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        pass
    api_key = st.text_input("Anthropic API key", type="password", value="", help="Or set ANTHROPIC_API_KEY in the environment / .streamlit/secrets.toml") or env_key
    model = ai.MODELS[st.selectbox("Model", list(ai.MODELS))]
    context = {
        "company": {"ticker": co.ticker, "name": co.name, "sector": co.sector, "industry": co.industry, "currency": cur, "price": price,
                    "market_cap_m": co.f("market_cap"), "52w_low": co.f("low52"), "52w_high": co.f("high52")},
        "history": {"years": co.years(), "revenue_m": co.series("revenue"), "net_income_m": co.series("net_income"), "equity_m": co.series("equity"),
                    "dividends_m": co.series("dividends"), "hist_defaults": hd},
        "assumptions": {"scenario": scen, "revenue_growth": a.revenue_growth, "net_margin": a.net_margin, "payout": a.payout, "cost_of_equity": a.cost_of_equity,
                        "terminal_growth": a.terminal_growth, "years": a.years, "fade_growth": a.fade_growth, "target_margin": a.target_margin},
        "residual_income": None if not ri else {"value_per_share": ri["per_share"], "book_value_per_share": ri["parts"]["Book value today"],
                                                 "pv_ri_per_share": ri["parts"]["PV of residual income"], "pv_tv_per_share": ri["parts"]["PV of terminal value"],
                                                 "bear": ri_bear, "bull": ri_bull, "final_year_roe": ri["rows"][-1]["roe_begin"]},
        "comps": {m_: {"median_multiple": c["stats"]["median"] if c.get("stats") else None, "implied_median": c["implied"]["median"] if c.get("implied") else None}
                  for m_, c in comps.items()},
        "blended_value": blend,
    }
    if not ai.available(api_key):
        st.info("Add an Anthropic API key (and `pip install anthropic`) to enable: peer suggestions, an assumption review, an analyst note and Q&A.")
    else:
        b1, b2, b3 = st.columns(3)
        if b1.button("Suggest peers"):
            with st.spinner("Asking Claude for comparables…"):
                try:
                    sug = ai.suggest_peers(api_key, model, {"ticker": co.ticker, "name": co.name, "sector": co.sector, "industry": co.industry})
                    ss["ai_peers"] = sug
                except Exception as exc:
                    st.error(f"Claude request failed: {exc}")
        if b2.button("Review my assumptions"):
            with st.spinner("Reviewing…"):
                try:
                    ss["ai_review"] = ai.review_assumptions(api_key, model, context)
                except Exception as exc:
                    st.error(f"Claude request failed: {exc}")
        if b3.button("Write analyst note"):
            with st.spinner("Drafting…"):
                try:
                    ss["ai_note"] = ai.analyst_note(api_key, model, context)
                except Exception as exc:
                    st.error(f"Claude request failed: {exc}")
        if ss.get("ai_peers"):
            st.markdown("**Suggested peers:** " + ", ".join(ss["ai_peers"]))
            if st.button("Use these peers"):
                ss["peers_txt"] = ", ".join(ss["ai_peers"])
                st.rerun()
        if ss.get("ai_review"):
            st.markdown("#### Assumption review")
            st.markdown(ss["ai_review"])
        if ss.get("ai_note"):
            st.markdown("#### Analyst note")
            st.markdown(ss["ai_note"])
        q = st.text_input("Ask a question about this valuation", placeholder="Why is the terminal value so large? What margin justifies today's price?")
        if q and st.button("Ask"):
            with st.spinner("Thinking…"):
                try:
                    st.markdown(ai.explain_question(api_key, model, q, context))
                except Exception as exc:
                    st.error(f"Claude request failed: {exc}")

# ================================================================== method
with tabs[6]:
    st.markdown(f"""
**Residual income.** Revenue grows at the chosen rate (optionally fading to terminal growth); net income = revenue × net margin (optionally gliding to a target);
dividends = net income × payout; book value rolls forward on net income less dividends. Residual income = net income − cost of equity × *beginning* book value.
Value = book value today + present value of residual income over the forecast + present value of a terminal value that grows residual income at the terminal rate.
Discounting is end-of-year from the latest fiscal year end; there is no roll-forward to today's date.

**Cost of equity.** CAPM: risk-free rate (Yahoo ^TNX 10-year Treasury yield when available) + beta (Yahoo) × equity risk premium (your input), or enter it directly.

**Trading comparables.** For each peer, EV/Revenue, EV/EBITDA, P/E and P/Book from Yahoo's trailing figures. Negative or missing denominators are "n/m" and excluded.
The 25th, 50th and 75th percentiles are applied to {co.ticker}'s own revenue, EBITDA (EV multiples, then less net debt and divided by shares), trailing EPS and book value per share.

**Football field.** Residual income bear–bull range, each comps multiple's interquartile range, the 52-week trading range and Yahoo's analyst price targets when available,
against the current price. The blended value weights residual income against the average of the comps medians.

**Limits.** Yahoo data can lag filings and its definitions (debt, EBITDA) differ from company reports. Clean-surplus book value ignores buybacks and other equity movements.
Educational tool, not investment advice.
""")

# ================================================================== export
def excel_bytes() -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame([{"Ticker": co.ticker, "Company": co.name, "Data": co.mode, "Retrieved": co.retrieved_at, "Price": price, "Currency": cur,
                       "Scenario": scen, "Residual income value": ri["per_share"] if ri else None, "Bear": ri_bear, "Bull": ri_bull, "Blended value": blend,
                       "Cost of equity": ke, "Terminal growth": a.terminal_growth, "Revenue growth": a.revenue_growth, "Net margin": a.net_margin,
                       "Payout": a.payout, "Forecast years": a.years}]).T.to_excel(xw, sheet_name="Summary", header=False)
        if ri:
            pd.DataFrame(ri["rows"]).to_excel(xw, sheet_name="Forecast", index=False)
        pd.DataFrame(ff).to_excel(xw, sheet_name="Football field", index=False)
        if peer_rows:
            pd.DataFrame([{**p, **E.multiples(p)} for p in [target_row] + peer_rows]).to_excel(xw, sheet_name="Comps", index=False)
        if co.years():
            pd.DataFrame({"Fiscal year": co.years(), **{k_: co.series(k_) for k_ in co.history}}).to_excel(xw, sheet_name="History", index=False)
        pd.DataFrame([{"Field": k_, "Value": v, "Source": co.sources.get(k_, "")} for k_, v in co.fields.items()]).to_excel(xw, sheet_name="Data sources", index=False)
    return buf.getvalue()


st.sidebar.divider()
try:
    st.sidebar.download_button("Download Excel workbook", excel_bytes(), file_name=f"{co.ticker}_valuation.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
except Exception as exc:  # openpyxl missing
    st.sidebar.caption(f"Excel export unavailable: {exc}")
st.sidebar.caption("Data from Yahoo Finance via yfinance. Educational tool, not investment advice.")
