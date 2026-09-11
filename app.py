import io
import time
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from google import genai
from google.genai import types
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# -----------------------------------------
# 1. LIVE MARKET DATA ENGINE (yfinance)
# -----------------------------------------
@st.cache_data(ttl=60)
def fetch_market_pulse():
    pulse_data = {}
    indices = {
        "Nifty 50": "^NSEI",
        "Bank Nifty": "^NSEBANK",
        "IT": "^CNXIT",
        "Auto": "^CNXAUTO",
        "FMCG": "^CNXFMCG",
        "Metal": "^CNXMETAL",
        "Pharma": "^CNXPHARMA"
    }
    
    try:
        tickers = " ".join(indices.values())
        data = yf.download(tickers, period="2d", group_by="ticker", progress=False)
        
        for name, ticker in indices.items():
            if ticker in data:
                ticker_data = data[ticker].dropna()
                if len(ticker_data) >= 2:
                    current_price = float(ticker_data['Close'].iloc[-1])
                    prev_close = float(ticker_data['Close'].iloc[-2])
                    pct_change = ((current_price - prev_close) / prev_close) * 100
                    
                    pulse_data[name] = {
                        "price": current_price,
                        "change": pct_change,
                        "bias": "Bullish Uptrend 🟢" if pct_change > 0 else "Bearish Downtrend 🔴"
                    }
    except Exception as e:
        pulse_data["error"] = str(e)
        
    return pulse_data

# -----------------------------------------
# 2. DATA EXTRACTION ENGINE (Screener.in)
# -----------------------------------------
@st.cache_data(ttl=3600)
def fetch_screener_data(ticker: str) -> dict:
    clean_ticker = ticker.upper().strip().replace(" ", "")
    
    urls = [
        f"https://www.screener.in/company/{clean_ticker}/consolidated/",
        f"https://www.screener.in/company/{clean_ticker}/"
    ]
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    resp = None
    for url in urls:
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            break
            
    if not resp or resp.status_code != 200:
        return {"error": f"Ticker {clean_ticker} not found on Screener.in."}

    soup = BeautifulSoup(resp.content, "html.parser")
    data = {"ticker": clean_ticker, "tables": {}}

    # Scrape Point 1: Quick Ratios snapshot bar
    ratio_items = soup.select("#top-ratios li")
    for li in ratio_items:
        name_elem = li.select_one(".name")
        val_elem = li.select_one(".nowrap .number")
        if name_elem and val_elem:
            name = name_elem.text.strip().lower()
            val_clean = val_elem.text.strip().replace(",", "")
            data[name] = val_clean # We store raw string, parse safely later

    # Scrape Points 2, 4, 5, 6, 7: Deep HTML Tables
    table_ids = ["quarters", "profit-loss", "balance-sheet", "cash-flow", "shareholding"]
    for tid in table_ids:
        div = soup.find(id=tid)
        if div:
            table = div.find("table")
            if table:
                df = pd.read_html(io.StringIO(str(table)))[0]
                data["tables"][tid] = df

    return data

# -----------------------------------------
# HELPER: PARSE HISTORICAL ROWS FROM TABLES
# -----------------------------------------
def extract_trend(df, keyword: str):
    if df is None or df.empty:
        return []
    try:
        row = df[df.iloc[:, 0].astype(str).str.contains(keyword, case=False, na=False)]
        if row.empty:
            return []
        vals = []
        for val in row.iloc[0, 1:].values:
            clean = str(val).replace(',', '').replace('%', '').strip()
            if clean not in ['-', '', 'nan', 'NaN', 'None']:
                try:
                    vals.append(float(clean))
                except ValueError:
                    continue
        return vals
    except Exception:
        return []

# -----------------------------------------
# 3. 10-POINT QUANTITATIVE SCORING ENGINE
# -----------------------------------------
def evaluate_fundamentals(data: dict) -> dict:
    if "error" in data:
        return {"verdict": "ERROR", "reasons": [data["error"]]}

    reasons, red_flags = [], []
    score = 0
    tables = data.get("tables", {})

    # SAFELY CONVERT TO FLOAT TO PREVENT CRASHES ON MISSING DATA
    def safe_float(val, default=0.0):
        if val is None:
            return default
        try:
            return float(str(val).replace(',', '').strip())
        except (ValueError, TypeError):
            return default

    # POINT 1: Top Quick Ratios
    book_value = safe_float(data.get("book value"), 1.0)
    current_price = safe_float(data.get("current price"), 0.0)
    market_cap = safe_float(data.get("market cap"), 0.0)
    pe_ratio = safe_float(data.get("stock p/e"), 0.0)
    roce = safe_float(data.get("roce"), 0.0)
    roe = safe_float(data.get("roe"), 0.0)
    debt_equity = safe_float(data.get("debt to equity"), 0.0)
    pledge = safe_float(data.get("pledged percentage"), 0.0)
    
    pb = (current_price / book_value) if book_value > 0 else 999.0

    if book_value <= 0:
        red_flags.append("🚨 Point 6 (Balance Sheet): Negative Net Worth / Book Value is negative.")
    if pledge > 5.0:
        red_flags.append(f"🚨 Point 2 (Shareholding): Significant Promoter Pledge at {pledge}%.")
    if debt_equity > 1.5:
        red_flags.append(f"🚨 Point 6 (Solvency): High Debt-to-Equity at {debt_equity}x.")
    
    if roce >= 15.0 and roe >= 15.0:
        score += 15
        reasons.append(f"✅ Point 1 (Capital Efficiency): Strong consistency — ROCE {roce}%, ROE {roe}%.")
    elif roce > 0 or roe > 0:
        reasons.append(f"⚠️ Point 1 (Capital Efficiency): Sub-par ROCE ({roce}%) or ROE ({roe}%).")

    # POINT 4: Quarterly Results (OPM Expansion Check)
    q_df = tables.get("quarters")
    opm_trend = extract_trend(q_df, "OPM")
    if opm_trend and len(opm_trend) >= 4:
        recent_opm = opm_trend[-4:]
        if recent_opm[-1] > recent_opm[0]:
            score += 15
            reasons.append(f"✅ Point 4 (Quarterly Inflection): OPM expanded from {recent_opm[0]}% to {recent_opm[-1]}% over recent quarters.")
        else:
            reasons.append(f"⚠️ Point 4 (Quarterly Results): OPM margin trend is flat or contracting ({recent_opm[-1]}%).")

    # POINT 5: Profit & Loss (10-Year Consistency & Chronic Losses)
    pnl_df = tables.get("profit-loss")
    net_profit = extract_trend(pnl_df, "Net Profit")
    if net_profit:
        profitable_years = sum(1 for p in net_profit if p > 0)
        total_years = len(net_profit)
        if total_years > 0:
            if profitable_years <= (total_years / 2):
                red_flags.append(f"🚨 Point 5 (P&L): Chronic losses detected — profitable in only {profitable_years} of {total_years} years.")
            elif profitable_years >= total_years - 1:
                score += 15
                reasons.append(f"✅ Point 5 (P&L Consistency): Profitable in {profitable_years} of {total_years} recorded years.")

    # POINT 6: Balance Sheet (Reserves Trend)
    bs_df = tables.get("balance-sheet")
    reserves = extract_trend(bs_df, "Reserves")
    if reserves and len(reserves) >= 2:
        if reserves[-1] < reserves[-2]:
            red_flags.append("🚨 Point 6 (Balance Sheet): Shrinking accumulated reserves.")
        else:
            score += 10
            reasons.append("✅ Point 6 (Balance Sheet): Reserves are expanding.")

    # POINT 7: Cash Flow Statement (Consistent Positive CFO)
    cf_df = tables.get("cash-flow")
    cfo = extract_trend(cf_df, "Operating Activity")
    if cfo:
        positive_cfo = sum(1 for c in cfo if c > 0)
        if positive_cfo < len(cfo) / 2:
            red_flags.append("🚨 Point 7 (Cash Flow): Operating cash flow is negative in most reported periods.")
        else:
            score += 10
            reasons.append(f"✅ Point 7 (Cash Generation): Positive CFO in {positive_cfo} of {len(cfo)} recorded years.")

    # POINT 2: Shareholding Pattern (Promoter Trend)
    sh_df = tables.get("shareholding")
    promoter = extract_trend(sh_df, "Promoters")
    if promoter and len(promoter) >= 4:
        if promoter[-1] < promoter[-4]:
            red_flags.append(f"🚨 Point 2 (Shareholding): Promoters trimming stake ({promoter[-4]}% down to {promoter[-1]}%).")
        elif promoter[-1] >= 45.0:
            score += 10
            reasons.append(f"✅ Point 2 (Shareholding): Stable promoter backing at {promoter[-1]}%.")

    # Final Verdict Computation
    verdict = "WATCH"
    color = "orange"
    if red_flags:
        verdict = "AVOID / SELL"
        color = "red"
    elif score >= 60 and pb <= 6.0:
        verdict = "MULTIBAGGER CANDIDATE / BUY"
        color = "green"
    elif score >= 40:
        verdict = "BUY (Steady Compounder)"
        color = "blue"

    return {
        "verdict": verdict,
        "color": color,
        "score": score,
        "flags": red_flags,
        "observations": reasons,
        "clean_metrics": {
            "Market Cap (Cr)": market_cap,
            "P/E": pe_ratio,
            "P/B": round(pb, 2),
            "ROCE %": roce,
            "ROE %": roe,
            "D/E": debt_equity
        }
    }

# -----------------------------------------
# 4. WEB SEARCH & AI LAYER (Reinforced Retry Loop)
# -----------------------------------------
@st.cache_data(ttl=3600)
def fetch_live_news(ticker: str) -> str:
    try:
        results = DDGS().text(f"{ticker} stock news India latest business update", max_results=5)
        if not results:
            return "No recent news headlines found."
        return "Recent Business & Market News:\n" + "\n".join([f"- {r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"Web news bypassed: {str(e)}"

@st.cache_data(ttl=3600)
def get_ai_verdict(ticker: str, metrics: dict, flags: list, observations: list) -> str:
    api_key = st.secrets.get("GEMINI_API_KEY", None)
    if not api_key:
        return "⚠️ Gemini API key not found. Please add `GEMINI_API_KEY` to `.streamlit/secrets.toml` or Streamlit Cloud Secrets."
        
    try:
        live_news = fetch_live_news(ticker)
        client = genai.Client(api_key=api_key)
        
        prompt = f"""
You are a senior institutional equity analyst. Evaluate this Indian stock: {ticker}.

Quantitative Scorecard (Points 1, 2, 4, 5, 6, 7):
- Metrics: {metrics}
- Detected Red Flags: {flags}
- Trend Observations: {observations}

Live Market & Concall Context (Points 3, 9, 10):
{live_news}

Task:
1. Peer & Industry Standing (Point 3): Compare its competitive moat against sector peers based on your knowledge.
2. Governance & Catalysts (Point 9): Highlight any corporate governance warnings, credit rating notes, or management concall updates from recent events.
3. Multibagger Inflection Thesis (Point 10): Is there an active catalyst (e.g., massive capex, margin inflection, new order books) or is it a steady compounder?
4. Conclude with a strict 3-bullet summary justifying BUY, SELL, AVOID, WATCH, or MULTIBAGGER.
"""
        config = types.GenerateContentConfig(temperature=0.2)
        
        # 5 retries with backoff to handle 503 UNAVAILABLE or 429 rate limit spikes
        max_retries = 5
        delay = 4
        
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model='gemini-3.6-flash',
                    contents=prompt,
                    config=config
                )
                return response.text
            except Exception as api_err:
                err_str = str(api_err)
                if ("503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str or "RESOURCE_EXHAUSTED" in err_str) and attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                else:
                    return f"⚠️ **API Temporary Constraint ({err_str[:40]}...)**: Google's servers are under high load. Please wait a minute and re-run."
                    
    except Exception as e:
        return f"Execution error: {str(e)}"

# -----------------------------------------
# 5. STREAMLIT UI DASHBOARD
# -----------------------------------------
st.set_page_config(page_title="10-Point Multibagger Screener", layout="wide")

# --- LIVE MARKET HEADER ---
st.markdown("### 📊 Live Market Pulse")
market_data = fetch_market_pulse()

if market_data and "error" not in market_data:
    col1, col2, col3, col4 = st.columns(4)
    if "Nifty 50" in market_data:
        nifty = market_data["Nifty 50"]
        col1.metric("Nifty 50 Index", f"₹{nifty['price']:.2f}", f"{nifty['change']:.2f}%")
        col2.markdown(f"**Nifty Trend**<br>{nifty['bias']}", unsafe_allow_html=True)
    if "Bank Nifty" in market_data:
        bank = market_data["Bank Nifty"]
        col3.metric("Bank Nifty Index", f"₹{bank['price']:.2f}", f"{bank['change']:.2f}%")
        col4.markdown(f"**Bank Nifty Trend**<br>{bank['bias']}", unsafe_allow_html=True)

    sectors = {k: v for k, v in market_data.items() if k not in ["Nifty 50", "Bank Nifty", "error"]}
    if sectors:
        sorted_sectors = sorted(sectors.items(), key=lambda x: x[1]['change'], reverse=True)
        top_2 = sorted_sectors[:2]
        worst_2 = sorted_sectors[-2:]
        
        st.markdown("---")
        sec_col1, sec_col2 = st.columns(2)
        with sec_col1:
            st.markdown("**🏆 Top Performing Sectors**")
            for sec_name, sec_data in top_2:
                st.markdown(f"- **{sec_name}**: {sec_data['change']:.2f}% 🟢")
        with sec_col2:
            st.markdown("**📉 Worst Performing Sectors**")
            for sec_name, sec_data in worst_2:
                st.markdown(f"- **{sec_name}**: {sec_data['change']:.2f}% 🔴")

st.markdown("---")
st.title("📈 10-Point Multibagger & Fundamental Screener")

ticker_input = st.text_input("🔍 Enter NSE/BSE Symbol or Name (e.g., HFCL, TATA STEEL, ITC, RENUKA):", "")

if st.button("Run Analysis") and ticker_input:
    with st.spinner(f"Scraping Screener.in tables and metrics for {ticker_input.upper()}..."):
        raw_data = fetch_screener_data(ticker_input)
        
    if "error" in raw_data:
        st.error(raw_data["error"])
    else:
        eval_results = evaluate_fundamentals(raw_data)
        metrics = eval_results["clean_metrics"]
        
        st.markdown(f"<h2 style='text-align: center; color: {eval_results['color']};'>VERDICT: {eval_results['verdict']}</h2>", unsafe_allow_html=True)
        st.progress(eval_results['score'] / 100)
        st.caption(f"Quantitative Health Score: {eval_results['score']}/100")
        
        st.subheader("Point 1: Top Quick Ratios")
        cols = st.columns(6)
        cols[0].metric("Market Cap (Cr)", metrics["Market Cap (Cr)"])
        cols[1].metric("P/E", metrics["P/E"])
        cols[2].metric("P/B", metrics["P/B"])
        cols[3].metric("ROCE", f"{metrics['ROCE %']}%")
        cols[4].metric("ROE", f"{metrics['ROE %']}%")
        cols[5].metric("Debt/Equity", f"{metrics['D/E']}x")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Points 2, 4, 5, 6, 7: Table & Trend Audit")
            for obs in eval_results["observations"]:
                st.write(obs)
            if eval_results["flags"]:
                st.error("🚨 RED FLAGS TRIGGERED")
                for flag in eval_results["flags"]:
                    st.write(flag)
        
        with col2:
            st.subheader("Points 3, 9, 10: AI Peer, Catalyst & Inflection Thesis")
            with st.spinner("Analyzing news and synthesizing final investment thesis..."):
                ai_insight = get_ai_verdict(
                    ticker=raw_data["ticker"],
                    metrics=metrics,
                    flags=eval_results["flags"],
                    observations=eval_results["observations"]
                )
            st.info(ai_insight)
