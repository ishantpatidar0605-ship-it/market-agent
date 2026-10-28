"""
World economy & markets agent (free Gemini API).
Watchlist ZAROORI NAHI: news se khud important companies dhundhta hai.

Usage (PowerShell):
    $env:GEMINI_API_KEY="your_key"
    python agent_gemini.py                      # world snapshot + news + auto companies
    python agent_gemini.py --web                # + live Google Search
    python agent_gemini.py --tickers TCS.NS AAPL   # apni extra companies bhi
    python agent_gemini.py --no-auto            # sirf world analysis, koi company nahi
    python agent_gemini.py --max-auto 5         # auto companies ki sankhya
"""
import argparse
import os
import sys

from google import genai

from core import build_report, render_markdown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=[], help="optional extra tickers")
    ap.add_argument("--web", action="store_true", help="Google Search grounding on")
    ap.add_argument("--no-auto", action="store_true", help="news se companies auto-select mat karo")
    ap.add_argument("--max-auto", type=int, default=8)
    args = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY set karo (aistudio.google.com/apikey se free key).")

    client = genai.Client(api_key=key)
    result = build_report(
        client,
        extra_tickers=args.tickers,
        use_web=args.web,
        auto=not args.no_auto,
        max_auto=args.max_auto,
        log=lambda m: print(m, file=sys.stderr if "retry" in m else sys.stdout),
    )

    os.makedirs("reports", exist_ok=True)
    path = f"reports/report_{result['date']}.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_markdown(result))
    print(f"Done -> {path}")


if __name__ == "__main__":
    main()
