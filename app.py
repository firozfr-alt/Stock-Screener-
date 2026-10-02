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

    # Capturing financial statements and ratios (Debtor Days / Working Capital)
    table_ids = ["quarters", "profit-loss", "balance-sheet", "cash-flow", "shareholding", "ratios"]
    for tid in table_ids:
        div = soup.find(id=tid)
        if div:
            table = div.find("table")
            if table:
                df = pd.read_html(io.StringIO(str(table)))[0]
                data["tables"][tid] = df

    return data

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

# ==========================================
# 2. ADVANCED QUANTITATIVE SCORING ENGINE
# ==========================================
def evaluate_fundamentals(data: dict) -> dict:
    if "error" in data:
        return {"verdict": "ERROR", "flags": [data["error"]], "observations": []}

    pos_flags, neg_flags = [], []
    tables = data.get("tables", {})

    def safe_float(val, default=0.0):
        if val is None: 
            return default
        try: 
            return float(str(val).replace(',', '').strip())
        except (ValueError, TypeError): 
            return default

    book_value = safe_float(data.get("book value"), 1.0)
    current_price = safe_float(data.get("current price"), 0.0)
    market_cap = safe_float(data.get("market cap"), 0.0)
    pe_ratio = safe_float(data.get("stock p/e"), 0.0)
    roce = safe_float(data.get("roce"), 0.0)
    roe = safe_float(data.get("roe"), 0.0)
    debt_equity = safe_float(data.get("debt to equity"), 0.0)
    pledge = safe_float(data.get("pledged percentage"), 0.0)
    pb = (current_price / book_value) if book_value > 0 else 999.0

    # 1. Solvency & Balance Sheet Health
    if book_value <= 0: 
        neg_flags.append("🚨 Balance Sheet: Negative Net Worth / Book Value.")
    if pledge > 5.0: 
        neg_flags.append(f"🚨 Shareholding: High Promoter Pledge at {pledge}%.")
    if debt_equity > 1.5: 
        neg_flags.append(f"🚨 Solvency: High Debt-to-Equity at {debt_equity}x.")
    elif debt_equity < 0.5: 
        pos_flags.append(f"✅ Balance Sheet: Conservative leverage with D/E of {debt_equity}x.")
    
    # 2. Capital Efficiency
    if roce >= 15.0 and roe >= 15.0:
        pos_flags.append(f"✅ Capital Efficiency: High compounding returns — ROCE {roce}%, ROE {roe}%.")
    else:
        neg_flags.append(f"⚠️ Capital Efficiency: Sub-par Return Ratios — ROCE {roce}%, ROE {roe}%.")

    # 3. Growth & Operating Leverage
    pnl_df = tables.get("profit-loss")
    sales = extract_trend(pnl_df, "Sales")
    sales_yoy = 0.0
    if sales and len(sales) >= 2:
        sales_yoy = ((sales[-1] - sales[-2]) / abs(sales[-2])) * 100 if sales[-2] != 0 else 0.0
        if sales_yoy > 15: 
            pos_flags.append(f"✅ Topline: Solid YoY Sales Growth of {sales_yoy:.1f}%.")
        elif sales_yoy < 0: 
            neg_flags.append(f"🚨 Topline: Sales contracted YoY by {sales_yoy:.1f}%.")

    q_df = tables.get("quarters")
    opm_trend = extract_trend(q_df, "OPM")
    if opm_trend and len(opm_trend) >= 4:
        recent_opm = opm_trend[-4:]
        if recent_opm[-1] > recent_opm[0]:
            pos_flags.append(f"✅ Margin Expansion: OPM improved from {recent_opm[0]}% to {recent_opm[-1]}%.")
        elif recent_opm[-1] < recent_opm[0] - 2:
            neg_flags.append(f"⚠️ Margins: OPM contracting from {recent_opm[0]}% down to {recent_opm[-1]}%.")

    # 4. Forensic Checks: Earnings Quality (CFO vs PAT) & Free Cash Flow (FCF)
    cf_df = tables.get("cash-flow")
    net_profit_trend = extract_trend(pnl_df, "Net Profit")
    cfo_trend = extract_trend(cf_df, "Operating Activity")
    capex_trend = extract_trend(cf_df, "Fixed assets purchased")

    if net_profit_trend and cfo_trend:
        min_len = min(len(net_profit_trend), len(cfo_trend), 3)
        if min_len >= 3:
            sum_pat = sum(net_profit_trend[-min_len:])
            sum_cfo = sum(cfo_trend[-min_len:])
            if sum_pat > 0:
                cfo_pat_ratio = sum_cfo / sum_pat
                if cfo_pat_ratio < 0.5:
                    neg_flags.append(f"🚨 Earnings Quality: Poor cash conversion. 3-Yr CFO is only {cfo_pat_ratio*100:.0f}% of Net Profit (Paper Profits).")
                elif cfo_pat_ratio >= 0.8:
                    pos_flags.append(f"✅ Earnings Quality: Reliable conversion (3-Yr CFO is {cfo_pat_ratio*100:.0f}% of Net Profit).")

        if capex_trend:
            recent_cfo = cfo_trend[-1]
            recent_capex = capex_trend[-1]
            fcf = recent_cfo - abs(recent_capex)
            if fcf < 0:
                neg_flags.append(f"⚠️ Free Cash Flow: Negative FCF (₹{fcf:.0f} Cr). Capex exceeds operating cash flow.")
            else:
                pos_flags.append(f"✅ Free Cash Flow: Positive FCF generation (₹{fcf:.0f} Cr) after growth capex.")

    # 5. Working Capital Integrity (Debtor Days)
    ratios_df = tables.get("ratios")
    debtor_days = extract_trend(ratios_df, "Debtor Days")
    if debtor_days and len(debtor_days) >= 2:
        if debtor_days[-2] > 0 and (debtor_days[-1] / debtor_days[-2]) > 1.3:
            neg_flags.append(f"🚨 Working Capital: Debtor days spiked from {debtor_days[-2]} to {debtor_days[-1]} days (High collection risk).")
        elif debtor_days[-1] < 45:
            pos_flags.append(f"✅ Working Capital: Rapid cash conversion cycle ({debtor_days[-1]} Debtor Days).")

    # 6. Shareholding & Regulatory Overhang
    sh_df = tables.get("shareholding")
    promoter = extract_trend(sh_df, "Promoters")
    current_promoter_holding = 0.0
    if promoter and len(promoter) >= 1:
        current_promoter_holding = promoter[-1]
        if current_promoter_holding > 75.0:
            neg_flags.append(f"🚨 Regulatory Overhang: Promoter stake at {current_promoter_holding}% breaches SEBI 75% limit.")
        else:
            pos_flags.append(f"✅ Stable promoter holding at {current_promoter_holding}%.")

    # --- VALUATION CIRCUIT BREAKERS ---
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

    # Emergency Override for Accumulated Red Flags
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
# 3. WEB SEARCH & MULTIBAGGER CATALYST LAYER
# ==========================================
@st.cache_data(ttl=3600)
def fetch_market_context(ticker: str) -> str:
    try:
        news = DDGS().text(f"{ticker} stock news India latest", max_results=2)
        deals = DDGS().text(f"{ticker} bulk deal block deal NSE BSE", max_results=1)
        concall = DDGS().text(f"{ticker} earnings call transcript summary management guidance capex order book", max_results=3)
        
        context = "Live Market News:\n" + "\n".join([r['body'] for r in news]) if news else ""
        context += "\n\nBlock/Bulk Deals:\n" + "\n".join([r['body'] for r in deals]) if deals else ""
        context += "\n\nManagement Guidance & Concall Transcripts:\n" + "\n".join([r['body'] for r in concall]) if concall else ""
        return context
    except Exception as e:
        return f"Market context search bypassed: {str(e)}"

def is_rate_limit_error(exception):
    err_str = str(exception)
    return any(err in err_str for err in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"])

@retry(wait=wait_random_exponential(multiplier=2, max=10), stop=stop_after_attempt(3), retry=retry_if_exception(is_rate_limit_error), reraise=True)
def generate_content_with_backoff(client, prompt, config, model_name):
    return client.models.generate_content(model=model_name, contents=prompt, config=config)

@st.cache_data(ttl=3600)
def get_ai_verdict(ticker: str, metrics: dict, flags: list, observations: list, quant_verdict: str, promoter_holding: float) -> str:
    api_key = st.secrets.get("GEMINI_API_KEY", None)
    if not api_key: 
        return "⚠️ Gemini API key not found in Streamlit Secrets."
        
    try:
        market_context = fetch_market_context(ticker)
        client = genai.Client(api_key=api_key)
        
        # Calculate exact SEBI Minimum Public Shareholding (MPS) overhang
        overhang_pct = max(0.0, promoter_holding - 75.0)
        overhang_cr = (overhang_pct / 100) * metrics.get("Market Cap (Cr)", 0)
        
        prompt = f"""
You are a senior institutional equity research analyst evaluating the Indian listed stock: {ticker}.
You MUST reconcile your investment thesis directly with the Quantitative Screener's findings below:

--- QUANTITATIVE SCREENER FINDINGS ---
Quantitative Verdict: {quant_verdict}
Financial Ratios: P/E {metrics['P/E']}x | P/B {metrics['P/B']}x | ROCE {metrics['ROCE %']}% | ROE {metrics['ROE %']}% | D/E {metrics['D/E']}x | Sales YoY {metrics['Sales YoY %']}%
Strengths Identified:
{chr(10).join(['- ' + str(item) for item in observations])}
Risks Identified:
{chr(10).join(['- ' + str(item) for item in flags])}

--- LIVE MARKET CONTEXT & CONCALL TRANSCRIPT NOTES ---
{market_context}

--- TASKS FOR YOUR THESIS ---
1. Implied Growth Reality Check: At {metrics['P/E']}x P/E, what level of multi-year EPS growth is the market discounting? Reconcile this against historical sales growth ({metrics['Sales YoY %']}%) and capital efficiency ({metrics['ROCE %']}% ROCE).
2. Regulatory Supply Barrier: The promoters hold {promoter_holding}%. If this exceeds 75%, explicitly calculate the supply overhang of {overhang_pct:.2f}% (approx ₹{overhang_cr:.2f} Cr) that must be liquidated to meet SEBI compliance.
3. Multibagger Inflection & Management Guidance: Review the concall and management commentary above. Is there an active catalyst (capacity expansion, major order book, operating leverage) that justifies a multibagger re-rating, or is growth already priced in?
4. Final Institutional Stance: Conclude with a clear recommendation (BUY, HOLD, WATCH, or AVOID) that directly aligns with the quantitative screening verdict or provides institutional justification for an override.

CRITICAL FORMATTING INSTRUCTION:
Throughout your output, you MUST wrap positive drivers and tailwinds in :green[text] and risk factors or valuation concerns in :red[text].
"""
        config = types.GenerateContentConfig(temperature=0.2)
        
        try:
            return generate_content_with_backoff(client, prompt, config, 'gemini-3.5-flash').text
        except Exception as primary_error:
            if is_rate_limit_error(primary_error):
                return generate_content_with_backoff(client, prompt, config, 'gemini-3.1-flash-lite').text
            raise primary_error
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
        ui_col1, ui_col2 = st.columns(2)
        
        with ui_col1:
            st.success("✅ Positive Observations (Tailwinds & Strengths)")
            if not eval_results["observations"]: 
                st.write("None found.")
            for obs in eval_results["observations"]:
                st.write(obs)
                
        with ui_col2:
            st.error("⚠️ Risk Factors (Friction & Valuation)")
            if not eval_results["flags"]: 
                st.write("None found.")
            for flag in eval_results["flags"]:
                st.write(flag)
            
    with tab2:
        st.write("Generating an AI thesis consumes API quota. Review the quantitative metrics before proceeding.")
        if "AVOID" in eval_results["verdict"] or "HIGH RISK" in eval_results["verdict"]:
            st.warning("⚠️ This stock tripped quantitative risk breakers. Review the flagged risks before generating a thesis.")
            
        if st.button("Generate Deep AI Thesis 🧠"):
            with st.spinner("Extracting concall guidance, live news, and synthesizing multibagger thesis..."):
                ai_insight = get_ai_verdict(
                    ticker=ticker,
                    metrics=metrics,
                    flags=eval_results["flags"],
                    observations=eval_results["observations"],
                    quant_verdict=eval_results["verdict"],
                    promoter_holding=eval_results.get("promoter_holding", 0.0)
                )
                st.session_state['ai_insight_generated'] = ai_insight
                
        if st.session_state.get('ai_insight_generated'):
            st.markdown(
                f"""<div style="max-width: 900px; line-height: 1.6;">
                {st.session_state['ai_insight_generated']}
                </div>""", 
                unsafe_allow_html=True
            )
