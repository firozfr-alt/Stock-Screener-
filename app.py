import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf
from google import genai
from google.genai import types

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
                ticker_data = data[ticker]
                if len(ticker_data) >= 2:
                    current_price = ticker_data['Close'].iloc[-1]
                    prev_close = ticker_data['Close'].iloc[-2]
                    pct_change = ((current_price - prev_close) / prev_close) * 100
                    
                    pulse_data[name] = {
                        "price": float(current_price),
                        "change": float(pct_change),
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
    ticker = ticker.upper().strip()
    urls = [
        f"https://www.screener.in/company/{ticker}/consolidated/",
        f"https://www.screener.in/company/{ticker}/"
    ]
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    resp = None
    for url in urls:
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            break
            
    if not resp or resp.status_code != 200:
        return {"error": f"Ticker {ticker} not found. Ensure it is a valid Indian stock symbol."}

    soup = BeautifulSoup(resp.content, "html.parser")
    data = {"ticker": ticker}

    ratio_items = soup.select("#top-ratios li")
    for li in ratio_items:
        name_elem = li.select_one(".name")
        val_elem = li.select_one(".nowrap .number")
        if name_elem and val_elem:
            name = name_elem.text.strip().lower()
            val = val_elem.text.strip().replace(",", "")
            try:
                data[name] = float(val)
            except ValueError:
                data[name] = val

    return data

# -----------------------------------------
# 3. QUANTITATIVE SCORING ENGINE
# -----------------------------------------
def evaluate_fundamentals(data: dict) -> dict:
    if "error" in data:
        return {"verdict": "ERROR", "reasons": [data["error"]]}

    reasons = []
    red_flags = []
    score = 0

    roce = data.get("roce", 0.0)
    roe = data.get("roe", 0.0)
    pe = data.get("stock p/e", 0.0)
    market_cap = data.get("market cap", 0.0)
    debt_equity = data.get("debt to equity", 0.0)
    pledge = data.get("pledged percentage", 0.0)
    book_value = data.get("book value", 1.0)
    current_price = data.get("current price", 0.0)
    pb = current_price / book_value if book_value > 0 else 999.0

    if book_value <= 0:
        red_flags.append("🚨 Negative Net Worth: Automatic Disqualification.")
    if pledge > 5.0:
        red_flags.append(f"🚨 High Promoter Pledge: {pledge}% (Red Flag).")
    if debt_equity > 1.5:
        red_flags.append(f"🚨 Excessive Debt to Equity: {debt_equity}x.")

    if roce >= 15.0 and roe >= 15.0:
        score += 30
        reasons.append(f"✅ Strong capital efficiency: ROCE ({roce}%) and ROE ({roe}%) > 15%.")
    else:
        reasons.append(f"⚠️ Weak capital efficiency: ROCE {roce}%, ROE {roe}%.")

    if debt_equity < 0.5:
        score += 20
        reasons.append(f"✅ Healthy Balance Sheet: Debt to Equity at {debt_equity}x.")
        
    if pb <= 4.0:
        score += 15
        reasons.append(f"✅ Reasonable Valuation: P/B is {pb:.1f}x.")
    else:
        reasons.append(f"⚠️ High Valuation: P/B is {pb:.1f}x.")

    if pledge == 0.0:
        score += 10
        reasons.append("✅ Zero promoter pledge.")

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
        "verdict": verdict,
        "color": color,
        "score": score,
        "flags": red_flags,
        "observations": reasons,
        "clean_metrics": {
            "Market Cap (Cr)": market_cap,
            "P/E": pe,
            "P/B": round(pb, 2),
            "ROCE %": roce,
            "ROE %": roe,
            "D/E": debt_equity
        }
    }

# -----------------------------------------
# 4. AI REASONING LAYER (Using st.secrets)
# -----------------------------------------
def get_ai_verdict(ticker: str, metrics: dict, flags: list, observations: list) -> str:
    # Safely retrieve the key from Streamlit secrets
    api_key = st.secrets.get("GEMINI_API_KEY", None)
    
    if not api_key:
        return "⚠️ Gemini API key not found. Please set `GEMINI_API_KEY` inside `.streamlit/secrets.toml` or Streamlit Cloud Secrets."
        
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"""
        You are a strict, fundamental equity analyst. Evaluate this Indian stock: {ticker}.
        Quantitative Data: {metrics}
        Red Flags Detected: {flags}
        Rule-Based Observations: {observations}
        Task:
        1. Identify potential 'inflection points' (margin expansion, capacity additions, new order books).
        2. Provide a final 3-bullet point thesis on whether this is a MULTIBAGGER, BUY, AVOID, or WATCH.
        """
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return response.text
    except Exception as e:
        return f"AI Analysis failed: {str(e)}"

# -----------------------------------------
# 5. STREAMLIT UI DASHBOARD
# -----------------------------------------
st.set_page_config(page_title="AI Fundamental Screener", layout="wide")

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

# --- MAIN SCREENER ---
st.title("📈 AI-Powered Multibagger Screener")

ticker_input = st.text_input("🔍 Enter NSE/BSE Ticker (e.g., HFCL, ATHER, ITC):", "")

if st.button("Run Analysis") and ticker_input:
    with st.spinner(f"Scraping Screener.in for {ticker_input.upper()}..."):
        raw_data = fetch_screener_data(ticker_input)
        
    if "error" in raw_data:
        st.error(raw_data["error"])
    else:
        st.success("Data extracted successfully!")
        
        eval_results = evaluate_fundamentals(raw_data)
        metrics = eval_results["clean_metrics"]
        
        st.markdown(f"<h2 style='text-align: center; color: {eval_results['color']};'>VERDICT: {eval_results['verdict']}</h2>", unsafe_allow_html=True)
        st.progress(eval_results['score'] / 100)
        
        st.subheader("1. Top Quick Ratios")
        cols = st.columns(6)
        cols[0].metric("Market Cap (Cr)", metrics["Market Cap (Cr)"])
        cols[1].metric("P/E", metrics["P/E"])
        cols[2].metric("P/B", metrics["P/B"])
        cols[3].metric("ROCE", f"{metrics['ROCE %']}%")
        cols[4].metric("ROE", f"{metrics['ROE %']}%")
        cols[5].metric("Debt/Equity", f"{metrics['D/E']}x")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("2. Checklist Observations")
            for obs in eval_results["observations"]:
                st.write(obs)
            if eval_results["flags"]:
                st.error("🚨 RED FLAGS DETECTED")
                for flag in eval_results["flags"]:
                    st.write(flag)
        
        with col2:
            st.subheader("3. AI Reasoning & Inflection Check")
            with st.spinner("Analyzing fundamentals with Gemini..."):
                ai_insight = get_ai_verdict(
                    ticker=ticker_input.upper(),
                    metrics=metrics,
                    flags=eval_results["flags"],
                    observations=eval_results["observations"]
                )
            st.info(ai_insight)
