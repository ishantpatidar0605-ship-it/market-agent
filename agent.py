"""
Economy & Markets Analyst Agent
--------------------------------
1. World economy / market news RSS feeds se padhta hai
2. Watchlist ki companies ka financial data + corporate actions (yfinance) laata hai
3. Claude se analysis karwata hai: positive/negative effects on economy, sectors,
   share market, global markets, aur individual stocks
4. Markdown report save karta hai: reports/report_YYYY-MM-DD.md

Usage:
    export ANTHROPIC_API_KEY=your_key
    python agent.py                 # watchlist.txt use karega
    python agent.py --web           # Claude ka live web search bhi on (latest corporate actions ke liye)
    python agent.py --tickers RELIANCE.NS TCS.NS AAPL
"""
import argparse
import datetime as dt
import os
import sys

import anthropic
import feedparser
import yfinance as yf

MODEL = "claude-sonnet-5-5"

FEEDS = {
    "Reuters Business": "https://feeds.reuters.com/reuters/businessNews",
    "BBC Business": "http://feeds.bbci.co.uk/news/business/rss.xml",
    "CNBC Economy": "https://www.cnbc.com/id/20910258/device/rss/rss.html",
    "CNBC Markets": "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "Economic Times Markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "Economic Times Economy": "https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms",
    "Livemint Markets": "https://www.livemint.com/rss/markets",
    "Moneycontrol Business": "https://www.moneycontrol.com/rss/business.xml",
    "Google News - World Economy": "https://news.google.com/rss/search?q=world+economy+OR+inflation+OR+central+bank+OR+oil+prices&hl=en-IN&gl=IN&ceid=IN:en",
    "Google News - Corporate Actions": "https://news.google.com/rss/search?q=dividend+OR+buyback+OR+merger+OR+acquisition+OR+stock+split+OR+earnings&hl=en-IN&gl=IN&ceid=IN:en",
}


# ----------------------------- data collection -----------------------------
def fetch_news(max_per_feed: int = 20) -> str:
    lines = []
    for name, url in FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:max_per_feed]:
                title = e.get("title", "").strip()
                summary = e.get("summary", "").strip()[:300]
                date = e.get("published", "")
                lines.append(f"[{name}] {date} | {title} | {summary}")
        except Exception as ex:  # ek feed fail ho to baaki chalein
            print(f"  ! feed failed: {name}: {ex}", file=sys.stderr)
    return "\n".join(lines)


def _safe(v):
    try:
        return None if v is None else (float(v) if isinstance(v, (int, float)) else str(v))
    except Exception:
        return str(v)


def company_snapshot(ticker: str) -> str:
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        keys = [
            "longName", "sector", "industry", "currency", "currentPrice", "marketCap",
            "trailingPE", "forwardPE", "priceToBook", "dividendYield", "debtToEquity",
            "returnOnEquity", "profitMargins", "revenueGrowth", "earningsGrowth",
            "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "targetMeanPrice", "recommendationKey",
        ]
        out = [f"=== {ticker} ==="]
        out += [f"{k}: {_safe(info.get(k))}" for k in keys if info.get(k) is not None]

        try:
            fin = t.quarterly_financials
            if fin is not None and not fin.empty:
                rows = [r for r in ("Total Revenue", "Gross Profit", "Operating Income", "Net Income", "EBITDA") if r in fin.index]
                out.append("Quarterly financials (latest 4 quarters):")
                out.append(fin.loc[rows].iloc[:, :4].to_string())
        except Exception:
            pass

        try:
            actions = t.actions
            if actions is not None and not actions.empty:
                out.append("Recent corporate actions (dividends/splits):")
                out.append(actions.tail(6).to_string())
        except Exception:
            pass

        try:
            cal = t.calendar
            if cal:
                out.append(f"Upcoming calendar (earnings/dividend dates): {cal}")
        except Exception:
            pass

        try:
            for n in (t.news or [])[:6]:
                c = n.get("content", n)
                out.append(f"News: {c.get('title', '')}")
        except Exception:
            pass
        return "\n".join(out)
    except Exception as ex:
        return f"=== {ticker} === data unavailable: {ex}"


# --------------------------------- analysis --------------------------------
SYSTEM = (
    "Tum ek senior macro + equity research analyst ho. Sirf diye gaye data (aur web search, agar on ho) "
    "par based raho. Facts aur apne inference me clearly farq rakho. Agar data kam ho to bolo. "
    "Reply Hinglish (Roman Hindi + English terms) me do. Ye investment advice nahi hai."
)

MARKET_PROMPT = """Neeche aaj ({date}) ki world economy / market news headlines hain.

Report in sections me banao:
1. Top 10 key developments (1-line each)
2. Global economy: positive effects | negative effects
3. Major economies (US, China, Europe, India, Japan) ka outlook
4. Share market impact: global indices, sectors (winners / losers), India (Nifty/Sensex) pe asar
5. Commodities, currency, bond yields (oil, gold, USD, INR) ka impact
6. Companies/sectors jo positively ya negatively affected honge aur kyun
7. Key risks aur upcoming events to watch
8. Short-term (1-2 hafte) vs medium-term (3-6 mahine) view

NEWS:
{news}
"""

COMPANY_PROMPT = """Neeche watchlist companies ka financial data, recent corporate actions aur news hai.
Aaj ki date: {date}. Macro context: {macro}

Har company ke liye:
- Financial health summary (revenue/profit trend, margins, debt, valuation)
- Latest corporate actions (dividend, split, buyback, M&A, results dates)
- Positive factors | Negative factors
- Share price pe likely impact (bullish / neutral / bearish + reasoning + key risk)
Last me ek comparison table do.

DATA:
{data}
"""


def ask_claude(client, prompt: str, use_web: bool) -> str:
    kwargs = dict(model=MODEL, max_tokens=8000, system=SYSTEM,
                  messages=[{"role": "user", "content": prompt}])
    if use_web:
        kwargs["tools"] = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 10}]
    resp = client.messages.create(**kwargs)
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def load_watchlist(path="watchlist.txt"):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("#")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", help="e.g. RELIANCE.NS TCS.NS AAPL")
    ap.add_argument("--web", action="store_true", help="Claude web search on karo")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY set karo.")

    client = anthropic.Anthropic()
    today = dt.date.today().isoformat()
    tickers = args.tickers or load_watchlist()

    print("1/3 News fetch ho rahi hai...")
    news = fetch_news()
    print("2/3 Market analysis...")
    market_report = ask_claude(client, MARKET_PROMPT.format(date=today, news=news[:60000]), args.web)

    company_report = ""
    if tickers:
        print(f"3/3 {len(tickers)} companies ka data + analysis...")
        data = "\n\n".join(company_snapshot(t) for t in tickers)
        company_report = ask_claude(
            client,
            COMPANY_PROMPT.format(date=today, macro=market_report[:3000], data=data[:60000]),
            args.web,
        )

    os.makedirs("reports", exist_ok=True)
    path = f"reports/report_{today}.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Economy & Markets Report - {today}\n\n## Part 1: World Economy & Markets\n\n{market_report}\n")
        if company_report:
            f.write(f"\n\n## Part 2: Companies - Financials & Corporate Actions\n\n{company_report}\n")
        f.write("\n\n---\n*Automated analysis, investment advice nahi hai. Decision se pehle khud verify karo.*\n")
    print(f"Done -> {path}")


if __name__ == "__main__":
    main()
