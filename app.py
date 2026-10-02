import io
import time
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from google import genai
from google.genai import types
import pandas as pd
import requests
import streamlit as st
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception

# ==========================================
# 1. DATA EXTRACTION ENGINE (Screener.in)
# ==========================================
@st.cache_data(ttl=3600)
def fetch_screener_data(ticker: str) -> dict:
    clean_ticker = ticker.upper().strip().replace(" ", "")
    urls = [
        f"https://www.screener.in/company/{clean_ticker}/consolidated/",
        f"https://www.screener.in/company/{clean_ticker}/"
    ]
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    resp = None
    for url in urls:
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            break
            
    if not resp or resp.status_code != 200:
        return {"error": f"Ticker {clean_ticker} not found on Screener.in."}

    soup = BeautifulSoup(resp.content, "html.parser")
    data = {"ticker": clean_ticker, "tables": {}}

    ratio_items = soup.select("#top-ratios li")
    for li in ratio_items:
        name_elem = li.select_one(".name")
        val_elem = li.select_one(".nowrap .number")
        if name_elem and val_elem:
            name = name_elem.text.strip().lower()
            val_clean = val_elem.text.strip().replace(",", "")
            data[name] = val_clean 

    table_ids = ["quarters", "profit-loss", "balance-sheet", "cash-flow", "shareholding"]
    for tid in table_ids:
        div = soup.find(id=tid)
        if div:
            table = div.find("table")
            if table:
                df = pd.read_html(io.StringIO(str(table)))[0]
                data["tables"][tid] = df

    return data

def extract_trend(df, keyword: str):
    if df is None or df.empty: return []
    try:
        row = df[df.iloc[:, 0].astype(str).str.contains(keyword, case=False, na=False)]
        if row.empty: return []
        vals = []
        for val in row.iloc[0, 1:].values:
            clean = str(val).replace(',', '').replace('%', '').strip()
            if clean not in ['-', '', 'nan', 'NaN', 'None']:
                try: vals.append(float(clean))
                except ValueError: continue
        return vals
    except Exception: return []

# ==========================================
# 2. QUANTITATIVE SCORING WITH CIRCUIT BREAKERS
# ==========================================
def evaluate_fundamentals(data: dict) -> dict:
    if "error" in data:
        return {"verdict": "ERROR", "flags": [data["error"]], "observations": []}

    pos_flags, neg_flags = [], []
    tables = data.get("tables", {})

    def safe_float(val, default=0.0):
        if val is None: return default
        try: return float(str(val).replace(',', '').strip())
        except (ValueError, TypeError): return default

    book_value = safe_float(data.get("book value"), 1.0)
    current_price = safe_float(data.get("current price"), 0.0)
    market_cap = safe_float(data.get("market cap"), 0.0)
    pe_ratio = safe_float(data.get("stock p/e"), 0.0)
    roce = safe_float(data.get("roce"), 0.0)
    roe = safe_float(data.get("roe"), 0.0)
    debt_equity = safe_float(data.get("debt to equity"), 0.0)
    pledge = safe_float(data.get("pledged percentage"), 0.0)
    pb = (current_price / book_value) if book_value > 0 else 999.0

    # 1. Solvency & Balance Sheet
    if book_value <= 0: neg_flags.append("🚨 Balance Sheet: Negative Net Worth.")
    if pledge > 5.0: neg_flags.append(f"🚨 Shareholding: Promoter Pledge at {pledge}%.")
    if debt_equity > 1.5: neg_flags.append(f"🚨 Solvency: High Debt-to-Equity at {debt_equity}x.")
    elif debt_equity < 0.5: pos_flags.append(f"✅ Healthy Balance Sheet: D/E is {debt_equity}x.")
    
    # 2. Capital Efficiency
    if roce >= 15.0 and roe >= 15.0:
        pos_flags.append(f"✅ Capital Efficiency: Strong consistency — ROCE {roce}%, ROE {roe}%.")
    else:
        neg_flags.append(f"⚠️ Capital Efficiency: Sub-par Return Ratios — ROCE {roce}%, ROE {roe}%.")

    # 3. Growth Trends
    pnl_df = tables.get("profit-loss")
    sales = extract_trend(pnl_df, "Sales")
    sales_yoy = 0
    if sales and len(sales) >= 2:
        sales_yoy = ((sales[-1] - sales[-2]) / abs(sales[-2])) * 100 if sales[-2] != 0 else 0
        if sales_yoy > 15: pos_flags.append(f"✅ Topline: Excellent YoY Sales Growth of {sales_yoy:.1f}%.")
        elif sales_yoy < 0: neg_flags.append(f"🚨 Topline: Sales declined YoY by {sales_yoy:.1f}%.")

    # 4. Cash Flow
    cf_df = tables.get("cash-flow")
    cfo = extract_trend(cf_df, "Operating Activity")
    if cfo:
        positive_cfo = sum(1 for c in cfo if c > 0)
        if positive_cfo < len(cfo) / 2: neg_flags.append("🚨 Cash Flow: Operating cash flow is negative in most reported periods.")
        else: pos_flags.append(f"✅ Cash Generation: Positive CFO in {positive_cfo} of {len(cfo)} recorded years.")

    # 5. Shareholding & Regulatory Overhang
    sh_df = tables.get("shareholding")
    promoter = extract_trend(sh_df, "Promoters")
    current_promoter_holding = 0
    if promoter and len(promoter) >= 1:
        current_promoter_holding = promoter[-1]
        if current_promoter_holding > 75.0:
            neg_flags.append(f"🚨 Regulatory Overhang: Promoter holding ({current_promoter_holding}%) breaches SEBI 75% limit.")
        else:
            pos_flags.append(f"✅ Stable promoter backing at {current_promoter_holding}%.")

    # --- THE CIRCUIT BREAKERS (Fixing the false "BUY" ratings) ---
    if pe_ratio > 80 and roce < 15:
        verdict = "HIGH RISK (Overvalued Momentum)"
        color = "red"
    elif pe_ratio > 50:
        verdict = "WATCH (Valuation Stretch)"
        color = "orange"
    elif roce >= 18 and sales_yoy >= 12 and pe_ratio <= 50 and debt_equity < 1.0:
        verdict = "BUY (Steady Compounder)"
        color = "blue"
    else:
        verdict = "HOLD (Mixed Signals)"
        color = "gray"

    # Edge Case Overrides
    if len(neg_flags) >= 4:
        verdict = "AVOID / SELL"
        color = "red"

    return {
        "verdict": verdict,
        "color": color,
        "flags": neg_flags,
        "observations": pos_flags,
        "promoter_holding": current_promoter_holding,
        "clean_metrics": {
            "Market Cap (Cr)": market_cap, "P/E": pe_ratio, "P/B": round(pb, 2),
            "ROCE %": roce, "ROE %": roe, "D/E": debt_equity, "Sales YoY %": round(sales_yoy, 2)
        }
    }

# ==========================================
# 3. WEB SEARCH & AI LAYER (SYNTHESIZED)
# ==========================================
@st.cache_data(ttl=3600)
def fetch_live_news(ticker: str) -> str:
    try:
        news = DDGS().text(f"{ticker} stock news India latest", max_results=3)
        deals = DDGS().text(f"{ticker} bulk deal block deal NSE BSE", max_results=2)
        return "News:\n" + "\n".join([r['body'] for r in news]) + "\nDeals:\n" + "\n".join([r['body'] for r in deals])
    except: return "Web news bypassed."

def is_rate_limit_error(exception):
    return any(err in str(exception) for err in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"])

@retry(wait=wait_random_exponential(multiplier=2, max=10), stop=stop_after_attempt(3), retry=retry_if_exception(is_rate_limit_error), reraise=True)
def generate_content_with_backoff(client, prompt, config, model_name):
    return client.models.generate_content(model=model_name, contents=prompt, config=config)

@st.cache_data(ttl=3600)
def get_ai_verdict(ticker: str, metrics: dict, flags: list, observations: list, quant_verdict: str, promoter_holding: float) -> str:
    api_key = st.secrets.get("GEMINI_API_KEY", None)
    if not api_key: return "⚠️ Gemini API key not found."
        
    try:
        live_news = fetch_live_news(ticker)
        client = genai.Client(api_key=api_key)
        
        # Calculate Overhang data for the prompt
        overhang_pct = max(0.0, promoter_holding - 75.0)
        overhang_cr = (overhang_pct / 100) * metrics.get("Market Cap (Cr)", 0)
        
        prompt = f"""
        You are a senior institutional equity analyst evaluating {ticker}. 
        You MUST reconcile your qualitative thesis with the Quantitative Screener's exact findings below:

        Quantitative Verdict: {quant_verdict}
        Metrics: P/E {metrics['P/E']}x | ROCE {metrics['ROCE %']}% | Sales YoY {metrics['Sales YoY %']}%
        Strengths: {observations}
        Risks: {flags}

        Live Market Context: {live_news}

        TASKS:
        1. Implied Growth Reality Check: At {metrics['P/E']}x P/E, is the market pricing in hyper-growth? Reconcile this multiple against the historical {metrics['Sales YoY %']}% sales growth and {metrics['ROCE %']}% ROCE.
        2. Supply Overhang: The promoters own {promoter_holding}%. If this is over 75%, explicitly state that an excess {overhang_pct:.2f}% stake (approx ₹{overhang_cr:.2f} Cr) must be liquidated for SEBI compliance, creating a supply barrier.
        3. Final Stance: Deliver a 3-bullet thesis concluding with a final stance (BUY, HOLD, WATCH, or AVOID) that aligns with the Quantitative verdict.
        
        FORMATTING: Use :green[text] for strengths/bull arguments and :red[text] for risks/bear arguments.
        """
        config = types.GenerateContentConfig(temperature=0.2)
        
        try:
            return generate_content_with_backoff(client, prompt, config, 'gemini-3.5-flash').text
        except Exception as e:
            if is_rate_limit_error(e):
                return generate_content_with_backoff(client, prompt, config, 'gemini-3.1-flash-lite').text
            raise e
    except Exception as e:
        return f"⚠️ AI analysis failed: {str(e)}"

# ==========================================
# 4. STREAMLIT UI DASHBOARD
# ==========================================
st.set_page_config(page_title="Fundamental Screener", page_icon="📈", layout="wide")
st.title("Fundamental Screener")

col_search, _ = st.columns([1, 2])
with col_search:
    ticker_input = st.text_input("🔍 Enter NSE/BSE Symbol (e.g., TATASTEEL, ITC):", "")

if st.button("Run Quantitative Analysis") and ticker_input:
    with st.spinner(f"Fetching math and fundamentals for {ticker_input.upper()}..."):
        raw_data = fetch_screener_data(ticker_input)
        if "error" in raw_data:
            st.error(raw_data["error"])
        else:
            st.session_state['ticker'] = ticker_input
            st.session_state['raw_data'] = raw_data
            st.session_state['eval_results'] = evaluate_fundamentals(raw_data)
            st.session_state['ai_insight_generated'] = False

if 'eval_results' in st.session_state:
    eval_results = st.session_state['eval_results']
    metrics = eval_results["clean_metrics"]
    raw_data = st.session_state['raw_data']
    ticker = st.session_state['ticker']
    
    st.markdown(f"<h3 style='color: {eval_results['color']};'>Verdict: {eval_results['verdict']}</h3>", unsafe_allow_html=True)
    
    cols = st.columns(6)
    cols[0].metric("Market Cap (Cr)", f"{metrics['Market Cap (Cr)']:,.0f}")
    cols[1].metric("P/E", metrics["P/E"])
    cols[2].metric("P/B", metrics["P/B"])
    cols[3].metric("ROCE", f"{metrics['ROCE %']}%")
    cols[4].metric("ROE", f"{metrics['ROE %']}%")
    cols[5].metric("Debt/Equity", f"{metrics['D/E']}x")
    st.divider()

    tab1, tab2 = st.tabs(["📊 Trend Audit", "🧠 AI Analyst Thesis (On-Demand)"])
    
    with tab1:
        # FIX: STRICT UI SEGREGATION
        ui_col1, ui_col2 = st.columns(2)
        
        with ui_col1:
            st.success("✅ Positive Observations (Tailwinds & Strengths)")
            if not eval_results["observations"]: st.write("None found.")
            for obs in eval_results["observations"]:
                st.write(obs)
                
        with ui_col2:
            st.error("⚠️️ Risk Factors (Friction & Valuation)")
            if not eval_results["flags"]: st.write("None found.")
            for flag in eval_results["flags"]:
                st.write(flag)
            
    with tab2:
        st.write("Generating an AI thesis consumes API quota. Ensure the fundamental metrics look promising before proceeding.")
        if "AVOID" in eval_results["verdict"] or "HIGH RISK" in eval_results["verdict"]:
            st.warning("⚠️ This stock failed the quantitative screen. Running AI analysis is not recommended, but you can override below.")
            
        if st.button("Generate Deep AI Thesis 🧠"):
            with st.spinner("Fetching live news, bulk deals, and synthesizing final investment thesis..."):
                ai_insight = get_ai_verdict(
                    ticker=ticker,
                    metrics=metrics,
                    flags=eval_results["flags"],
                    observations=eval_results["observations"],
                    quant_verdict=eval_results["verdict"],
                    promoter_holding=eval_results.get("promoter_holding", 0)
                )
                st.session_state['ai_insight_generated'] = ai_insight
                
        if st.session_state.get('ai_insight_generated'):
            # Constrain text width for better readability on desktop
            st.markdown(
                f"""<div style="max-width: 900px; line-height: 1.6;">
                {st.session_state['ai_insight_generated']}
                </div>""", 
                unsafe_allow_html=True
            )
