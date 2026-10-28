"""
Browser UI (Streamlit).  Run:  streamlit run app.py
agent.py, core.py, agent_gemini.py isi folder me hone chahiye.
"""
import os
import warnings

warnings.filterwarnings("ignore")

import streamlit as st
from google import genai

from core import build_report, render_markdown

st.set_page_config(page_title="Market Agent", page_icon="📈", layout="wide")
st.title("📈 World Economy & Markets Agent")
st.caption("Global snapshot + world news + auto-selected companies ka AI analysis. Investment advice nahi hai.")

with st.sidebar:
    st.header("Settings")
    key = st.text_input(
        "Gemini API key",
        value=os.environ.get("GEMINI_API_KEY", ""),
        type="password",
        help="aistudio.google.com/apikey se free key",
    )
    auto = st.checkbox("News se companies khud dhundo", value=True)
    max_auto = st.slider("Auto companies (max)", 0, 10, 6, disabled=not auto)
    use_web = st.checkbox("Live web search (Google grounding)", value=False,
                          help="Free tier me quota jaldi khatam ho sakta hai")
    extra_text = st.text_area("Extra tickers (optional, ek line me ek)", value="", height=100,
                              help="Jaise TCS.NS, RELIANCE.NS, AAPL")
    run = st.button("Report banao", type="primary", use_container_width=True)

if run:
    if not key:
        st.error("Sidebar me Gemini API key daalo.")
        st.stop()
    client = genai.Client(api_key=key)
    extra = [t.strip() for t in extra_text.splitlines() if t.strip()]
    try:
        with st.status("Report ban rahi hai (2-5 minute)...", expanded=True) as status:
            result = build_report(client, extra, use_web, auto, max_auto, log=lambda m: st.write(m))
            status.update(label="Report ready", state="complete")
        st.session_state["result"] = result
    except Exception as ex:
        st.error(f"Error: {ex}")
        st.info("429 / quota error aaye to 'Live web search' band karo, auto companies kam karo, ya thodi der baad try karo.")

r = st.session_state.get("result")
if r:
    st.download_button("Report download (.md)", render_markdown(r), file_name=f"report_{r['date']}.md")
    t1, t2, t3 = st.tabs(["📊 Global Snapshot", "🌍 World & Markets", "🏢 Companies"])
    with t1:
        st.markdown(r["snapshot"])
    with t2:
        st.markdown(r["market"])
    with t3:
        if r["tickers"]:
            st.caption("Companies: " + ", ".join(r["tickers"]))
        st.markdown(r["company"] or "Koi company analyse nahi hui.")
else:
    st.info("Left sidebar me key daalke **Report banao** dabao. Watchlist ki zaroorat nahi.")
