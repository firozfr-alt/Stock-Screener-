import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf
from google import genai
from google.genai import types
from duckduckgo_search import DDGS
import io

# -----------------------------------------
# 1. LIVE MARKET DATA ENGINE (yfinance)
# -----------------------------------------
@st.cache_data(ttl=60)
def fetch_market_pulse():
    pulse_data = {}
    indices = {
        "Nifty 50": "^NSEI", "Bank Nifty": "^NSEBANK", "IT": "^CNXIT",
        "Auto": "^CNXAUTO", "FMCG": "^CNXFMCG", "Metal": "^CNXMETAL", "Pharma": "^CNXPHARMA"
    }
    try:
        tickers = " ".join(indices.values())
        data = yf.download(tickers, period="2d", group_by="ticker", progress=False)
        for name, ticker in indices.items():
            if ticker in data:
                ticker_data = data[ticker]
                if len(ticker_data) >= 2:
                    current_price = ticker_data['Close'].iloc[-1]
                    prev_close = ticker_data['Close'].iloc[-2]
                    pct_change = ((current_price - prev_close) / prev_close) * 100
                    pulse_data[name] = {
                        "price": float(current_price),
                        "change": float(pct_change),
                        "bias": "Bullish 🟢" if pct_change > 0 else "Bearish 🔴"
                    }
    except Exception as e:
        pulse_data["error"] = str(e)
    return pulse_data

# -----------------------------------------
# 2. DATA EXTRACTION ENGINE (Screener.in)
# -----------------------------------------
@st.cache_data(ttl=3600)
def fetch_screener_data(ticker: str) -> dict:
    ticker = ticker.upper().strip().replace(" ", "")
    urls = [
        f"https://www.screener.in/company/{ticker}/consolidated/",
        f"https://www.screener.in/company/{ticker}/"
    ]
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    resp = None
    for url in urls:
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200: break
            
    if not resp or resp.status_code != 200:
        return {"error": f"Ticker {ticker} not found."}

    soup = BeautifulSoup(resp.content, "html.parser")
    data = {"ticker": ticker, "tables": {}}

    # Scrape Point 1: Quick Ratios
    ratio_items = soup.select("#top-ratios li")
    for li in ratio_items:
        name_elem = li.select_one(".name")
        val_elem = li.select_one(".nowrap .number")
        if name_elem and val_elem:
            name = name_elem.text.strip().lower()
            try:
                data[name] = float(val_elem.text.strip().replace(",", ""))
            except ValueError:
                data[name] = val_elem.text.strip().replace(",", "")

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
# Helper function to parse table rows
# -----------------------------------------
def extract_trend(df, keyword):
    if df is None or df.empty: return []
    try:
        row = df[df.iloc[:, 0].astype(str).str.contains(keyword, case=False, na=False)]
        if row.empty: return []
        vals = []
        for val in row.iloc[0, 1:].values:
            clean = str(val).replace(',', '').replace('%', '').strip()
            if clean not in ['-', '', 'nan', 'NaN']:
                vals.append(float(clean))
        return vals
    except:
        return []

# -----------------------------------------
# 3. 10-POINT SCORING ENGINE
# -----------------------------------------
def evaluate_fundamentals(data: dict) -> dict:
    if "error" in data: return {"verdict": "ERROR", "reasons": [data["error"]]}
    
    reasons, red_flags = [], []
    score = 0
    tables = data.get("tables", {})

    # POINT 1: Top Quick Ratios
    book_value = data.get("book value", 1.0)
    pb = (data.get("current price", 0.0) / book_value) if book_value > 0 else 999.0
    roce, roe = data.get("roce", 0.0), data.get("roe", 0.0)
    debt_equity, pledge = data.get("debt to equity", 0.0), data.get("pledged percentage", 0.0)

    if book_value <= 0: red_flags.append("🚨 Negative Net Worth (Point 6).")
    if pledge > 5.0: red_flags.append(f"🚨 High Pledge (Point 2): {pledge}%.")
    if debt_equity > 1.5: red_flags.append(f"🚨 High Debt/Equity (Point 6): {debt_equity}x.")
    
    if roce >= 15.0 and roe >= 15.0:
        score += 15
        reasons.append(f"✅ Strong Capital Efficiency (Point 1): ROCE {roce}%, ROE {roe}%.")

    # POINT 4: Quarterly Results (Margin Expansion)
    q_df = tables.get("quarters")
    opm_trend = extract_trend(q_df, "OPM")
    if opm_trend and len(opm_trend) >= 4:
        recent_opm = opm_trend[-4:]
        if recent_opm[-1] > recent_opm[0]:
            score += 15
            reasons.append(f"✅ Margin Expansion (Point 4): OPM % expanded from {recent_opm[0]}% to {recent_opm[-1]}% over 4 quarters.")
        else:
            reasons.append(f"⚠️ Flat/Declining Margins (Point 4): Recent OPM % is {recent_opm[-1]}%.")

    # POINT 5: P&L (Chronic Losses & Consistency)
    pnl_df = tables.get("profit-loss")
    net_profit = extract_trend(pnl_df, "Net Profit")
    if net_profit:
        profitable_years = sum(1 for p in net_profit if p > 0)
        total_years = len(net_profit)
        if total_years > 0:
            if profitable_years <= (total_years / 2):
                red_flags.append(f"🚨 Chronic Losses (Point 5): Profitable in only {profitable_years} out of {total_years} years.")
            elif profitable_years >= total_years - 1:
                score += 15
                reasons.append(f"✅ Profit Consistency (Point 5): Profitable in {profitable_years} of {total_years} years.")

    # POINT 6: Balance Sheet (Reserves Trend)
    bs_df = tables.get("balance-sheet")
    reserves = extract_trend(bs_df, "Reserves")
    if reserves and len(reserves) >= 2:
        if reserves[-1] < reserves[-2]:
            red_flags.append(f"🚨 Shrinking Reserves (Point 6): Reserves declined recently.")
        else:
            score += 10
            reasons.append("✅ Growing Reserves (Point 6): Accumulated capital is increasing.")

    # POINT 7: Cash Flow Statement (Consistent CFO)
    cf_df = tables.get("cash-flow")
    cfo = extract_trend(cf_df, "Operating Activity")
    if cfo:
        positive_cfo = sum(1 for c in cfo if c > 0)
        if positive_cfo < len(cfo) / 2:
            red_flags.append("🚨 Weak Cash Generation (Point 7): Operating cash flow negative in most years.")
        else:
            score += 10
            reasons.append(f"✅ Cash Flow (Point 7): Positive operating cash flow in {positive_cfo} of {len(cfo)} years.")

    # POINT 2: Shareholding Pattern (Promoter Trend)
    sh_df = tables.get("shareholding")
    promoter = extract_trend(sh_df, "Promoters")
    if promoter and len(promoter) >= 4:
        if promoter[-1] < promoter[-4]:
            red_flags.append(f"🚨 Promoter Selling (Point 2): Holding declined from {promoter[-4]}% to {promoter[-1]}% recently.")
        elif promoter[-1] > 40:
            score += 10
            reasons.append(f"✅ Strong Promoters (Point 2): Holding stable at {promoter[-1]}%.")

    # Verdict Generation
    verdict = "WATCH"
    color = "orange"
    if red_flags:
        verdict = "AVOID / SELL"
        color = "red"
    elif score >= 65 and pb <= 6.0:
        verdict = "MULTIBAGGER CANDIDATE / BUY"
        color = "green"
    elif score >= 45:
        verdict = "BUY (Steady Compounder)"
        color = "blue"

    return {
        "verdict": verdict, "color": color, "score": score,
        "flags": red_flags, "observations": reasons,
        "clean_metrics": {
            "Market Cap (Cr)": data.get("market cap", 0.0),
            "P/E": data.get("stock p/e", 0.0),
            "P/B": round(pb, 2),
            "ROCE %": roce, "ROE %": roe, "D/E": debt_equity
        }
    }

# -----------------------------------------
# 4. FREE WEB SEARCH & AI REASONING
# -----------------------------------------
@st.cache_data(ttl=3600)
def fetch_live_news(ticker: str) -> str:
    try:
        results = DDGS().text(f"{ticker} stock news India latest update", max_results=5) 
        if not results: return "No recent news found."
        return "Latest Web Headlines:\n" + "\n".join([f"- {r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"Web search bypassed due to error: {str(e)}"

@st.cache_data(ttl=3600)
def get_ai_verdict(ticker: str, metrics: dict, flags: list, observations: list) -> str:
    api_key = st.secrets.get("GEMINI_API_KEY", None)
    if not api_key: return "⚠️ Gemini API key not found in secrets."
        
    try:
        live_news = fetch_live_news(ticker)
        client = genai.Client(api_key=api_key)
        
        prompt = f"""
        You are an expert equity analyst. Evaluate this Indian stock: {ticker}.
        Quantitative Data (Points 1, 2, 4, 5, 6, 7):
        - Metrics: {metrics}
        - Red Flags: {flags}
        - Observations: {observations}
        
        Live Market Context (Points 3, 9, 10):
        {live_news}
        
        Task:
        1. Evaluate Peer Position and News Catalyst (Points 3 & 9).
        2. Identify Multibagger Inflection Points based on news/margins (Point 10).
        3. Provide a concise 3-bullet thesis on whether this stock is a MULTIBAGGER, BUY, AVOID, or WATCH. 
        """
        config = types.GenerateContentConfig(temperature=0.2)
        response = client.models.generate_content(model='gemini-3.6-flash', contents=prompt, config=config)
        return response.text
    except Exception as e:
        return f"AI Analysis failed: {str(e)}"

# -----------------------------------------
# 5. STREAMLIT UI DASHBOARD
# -----------------------------------------
st.set_page_config(page_title="10-Point Screener", layout="wide")

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

st.markdown("---")
st.title("📈 10-Point Multibagger Screener")

ticker_input = st.text_input("🔍 Enter NSE/BSE Ticker (e.g., HFCL, TATA STEEL):", "")

if st.button("Run Analysis") and ticker_input:
    with st.spinner(f"Extracting all 10-Year Tables from Screener.in for {ticker_input.upper()}..."):
        raw_data = fetch_screener_data(ticker_input)
        
    if "error" in raw_data:
        st.error(raw_data["error"])
    else:
        eval_results = evaluate_fundamentals(raw_data)
        metrics = eval_results["clean_metrics"]
        
        st.markdown(f"<h2 style='text-align: center; color: {eval_results['color']};'>VERDICT: {eval_results['verdict']}</h2>", unsafe_allow_html=True)
        st.progress(eval_results['score'] / 100)
        
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
            st.subheader("Points 2, 4, 5, 6, 7: Deep Table Observations")
            for obs in eval_results["observations"]:
                st.write(obs)
            if eval_results["flags"]:
                st.error("🚨 CRITICAL RED FLAGS DETECTED")
                for flag in eval_results["flags"]:
                    st.write(flag)
        
        with col2:
            st.subheader("Points 3, 9, 10: AI Catalyst & Peer Check")
            with st.spinner("Searching DuckDuckGo and asking Gemini for final synthesis..."):
                ai_insight = get_ai_verdict(
                    ticker=ticker_input.upper(),
                    metrics=metrics,
                    flags=eval_results["flags"],
                    observations=eval_results["observations"]
                )
            st.info(ai_insight)
