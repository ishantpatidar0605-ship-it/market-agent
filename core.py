"""
Shared logic: global market snapshot + news + auto-discovered companies + Gemini analysis.
agent.py isi folder me rehni chahiye (news/company data functions wahin se aate hain).
"""
import datetime as dt
import os
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import yfinance as yf
from google.genai import types

from agent import company_snapshot, fetch_news

# Model name Google badalta rehta hai. Error aaye to: $env:GEMINI_MODEL="naya-model"
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

SYSTEM = (
    "Tum ek senior macro + equity research analyst ho. Sirf diye gaye data (aur web search, agar on ho) "
    "par based raho; koi number khud se mat banao. Facts aur apne inference me clearly farq rakho. "
    "Agar data kam ya purana lage to bolo. Reply Hinglish (Roman Hindi + English terms) me do. "
    "Ye investment advice nahi hai."
)

GROUPS = {
    "Equity indices": {
        "S&P 500": "^GSPC", "Nasdaq": "^IXIC", "Dow Jones": "^DJI", "FTSE 100": "^FTSE",
        "DAX": "^GDAXI", "CAC 40": "^FCHI", "Nikkei 225": "^N225", "Hang Seng": "^HSI",
        "Shanghai": "000001.SS", "Kospi": "^KS11", "Sensex": "^BSESN", "Nifty 50": "^NSEI",
        "ASX 200": "^AXJO",
    },
    "Commodities": {
        "Brent crude": "BZ=F", "WTI crude": "CL=F", "Gold": "GC=F", "Silver": "SI=F",
        "Copper": "HG=F", "Natural gas": "NG=F",
    },
    "FX, rates, risk": {
        "Dollar index": "DX-Y.NYB", "USD/INR": "INR=X", "EUR/USD": "EURUSD=X",
        "USD/JPY": "JPY=X", "US 10Y yield (%)": "^TNX", "VIX": "^VIX",
    },
}

MARKET_PROMPT = """Aaj ki date: {date}.

NEECHE DO DATA HAIN: (A) global market snapshot (yfinance, last close vs previous close, kuch delayed ho sakta hai)
aur (B) world economy / market news headlines.

Report in sections me banao:
1. Top 10 key developments (1-line each)
2. Global economy: positive effects | negative effects
3. Major economies (US, China, Europe, India, Japan) ka outlook
4. Share market impact: global indices (snapshot ke numbers use karo), sectors winners/losers, India (Nifty/Sensex) pe asar
5. Commodities, currency, bond yields (oil, gold, USD, INR, US 10Y) ka impact
6. Aaj ki badi corporate news: earnings, M&A, buyback, dividend, IPO, guidance
7. Companies/sectors jo positively ya negatively affected honge aur kyun
8. Key risks aur upcoming events (central bank meetings, data releases)
9. Short-term (1-2 hafte) vs medium-term (3-6 mahine) view

Sabse aakhir me, report ke baad, ek alag last line EXACTLY is format me likho (kuch aur uske baad nahi):
TICKERS: SYM1, SYM2, SYM3
Isme news me sabse zyada important {n} companies ke Yahoo Finance tickers do (alag-alag desh/sector mix,
jaise AAPL, NVDA, RELIANCE.NS, 7203.T, SAP.DE). Sirf wahi jo tumhe pakka pata hon.

(A) SNAPSHOT:
{snapshot}

(B) NEWS:
{news}
"""

COMPANY_PROMPT = """Aaj ki date: {date}. Macro context: {macro}

Neeche companies ka financial data, recent corporate actions aur news hai. Inme se kuch companies aaj ki
news se auto-select hui hain, kuch user ne diye hain.

Har company ke liye:
- Financial health summary (revenue/profit trend, margins, debt, valuation)
- Latest corporate actions (dividend, split, buyback, M&A, results dates)
- Positive factors | Negative factors
- Share price pe likely impact (bullish / neutral / bearish + reasoning + key risk)
Last me ek comparison table do. Jis company ka data missing ho, wahan clearly likho "data unavailable".

DATA:
{data}
"""


def _quote(sym):
    h = yf.Ticker(sym).history(period="7d").dropna(subset=["Close"])
    if len(h) < 2:
        return None
    last, prev = float(h["Close"].iloc[-1]), float(h["Close"].iloc[-2])
    return last, (last / prev - 1) * 100, h.index[-1].strftime("%Y-%m-%d")


def global_snapshot():
    """Returns markdown table text of global indices, commodities, FX/rates."""
    lines = []
    for group, items in GROUPS.items():
        lines.append(f"\n**{group}**\n")
        lines.append("| Name | Last | Change % | As of |")
        lines.append("|---|---|---|---|")
        for name, sym in items.items():
            try:
                q = _quote(sym)
            except Exception:
                q = None
            if q:
                lines.append(f"| {name} | {q[0]:,.2f} | {q[1]:+.2f}% | {q[2]} |")
            else:
                lines.append(f"| {name} | n/a | n/a | n/a |")
    return "\n".join(lines)


def ask_gemini(client, prompt, use_web=False, log=print):
    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM,
        max_output_tokens=8000,
        tools=[types.Tool(google_search=types.GoogleSearch())] if use_web else None,
    )
    for attempt in range(4):  # free tier me 429/503 aam hai
        try:
            resp = client.models.generate_content(model=MODEL, contents=prompt, config=cfg)
            return resp.text or ""
        except Exception as ex:
            msg = str(ex)
            if ("429" in msg or "RESOURCE_EXHAUSTED" in msg or "503" in msg) and attempt < 3:
                wait = 30 * (attempt + 1)
                log(f"limit/busy, {wait}s ruk kar retry...")
                time.sleep(wait)
                continue
            raise
    return ""


def split_tickers(text):
    """Report text se last 'TICKERS: ...' line alag karta hai."""
    m = re.search(r"^\s*TICKERS:\s*(.+?)\s*$", text, flags=re.MULTILINE | re.IGNORECASE)
    if not m:
        return text, []
    clean = (text[: m.start()] + text[m.end():]).rstrip()
    tickers = [t.strip().strip("`*'\"") for t in re.split(r"[,\s]+", m.group(1)) if t.strip()]
    return clean, [t for t in tickers if re.fullmatch(r"[\^A-Za-z0-9.\-=]{1,15}", t)]


def valid_ticker(t):
    try:
        return not yf.Ticker(t).history(period="5d").empty
    except Exception:
        return False


def build_report(client, extra_tickers=None, use_web=False, auto=True, max_auto=8, log=print):
    today = dt.date.today().isoformat()
    extra = [t.strip() for t in (extra_tickers or []) if t.strip() and not t.startswith("#")]

    log("1/4 Global market snapshot (indices, oil, gold, FX, yields)...")
    snapshot = global_snapshot()

    log("2/4 World news fetch ho rahi hai...")
    news = fetch_news(max_per_feed=15)

    log("3/4 Market analysis (Gemini)...")
    raw = ask_gemini(
        client,
        MARKET_PROMPT.format(date=today, snapshot=snapshot, news=news[:35000], n=max_auto if auto else 0),
        use_web,
        log,
    )
    market, found = split_tickers(raw)

    tickers = list(dict.fromkeys(extra + (found[:max_auto] if auto else [])))
    tickers = [t for t in tickers if valid_ticker(t)][: max_auto + len(extra)]

    company = ""
    if tickers:
        log(f"4/4 Companies ka data + analysis: {', '.join(tickers)}")
        data = "\n\n".join(company_snapshot(t) for t in tickers)
        company = ask_gemini(
            client,
            COMPANY_PROMPT.format(date=today, macro=market[:3000], data=data[:35000]),
            use_web,
            log,
        )
    else:
        log("4/4 Koi valid company ticker nahi mila, company section skip.")

    return {"date": today, "snapshot": snapshot, "market": market, "company": company, "tickers": tickers}


def render_markdown(r):
    out = f"# Economy & Markets Report - {r['date']}\n\n"
    out += f"## Global Snapshot\n{r['snapshot']}\n\n"
    out += f"## Part 1: World Economy & Markets\n\n{r['market']}\n"
    if r["company"]:
        out += f"\n\n## Part 2: Companies ({', '.join(r['tickers'])})\n\n{r['company']}\n"
    out += "\n\n---\n*Automated analysis, investment advice nahi hai. Decision se pehle khud verify karo.*\n"
    return out
