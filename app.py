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

    # Scrape top-ratios container, preserving multi-value tags like High / Low
    ratio_items = soup.select("#top-ratios li")
    for li in ratio_items:
        name_elem = li.select_one(".name")
        val_elem = li.select_one(".value") or li.select_one(".nowrap")
        if name_elem and val_elem:
            name = name_elem.text.strip().lower()
            val_clean = val_elem.text.strip().replace(",", "").replace("₹", "").strip()
            data[name] = val_clean 

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
# 2. ADVANCED INFLECTION & BREAKOUT ENGINE
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

    # Core Ratios
    book_value = safe_float(data.get("book value"), 1.0)
    current_price = safe_float(data.get("current price"), 0.0)
    market_cap = safe_float(data.get("market cap"), 0.0)
    pe_ratio = safe_float(data.get("stock p/e"), 0.0)
    roce = safe_float(data.get("roce"), 0.0)
    roe = safe_float(data.get("roe"), 0.0)
    debt_equity = safe_float(data.get("debt to equity"), 0.0)
    pledge = safe_float(data.get("pledged percentage"), 0.0)
    pb = (current_price / book_value) if book_value > 0 else 999.0

    # 1. 52-Week High / Low & Breakout Proximity Analysis
    high_52w, low_52w = 0.0, 0.0
    drawdown_pct = 0.0
    breakout_proximity_pct = 0.0
    high_low_str = data.get("high / low", "")
    if "/" in high_low_str:
        parts = high_low_str.split("/")
        high_52w = safe_float(parts[0].strip(), 0.0)
        low_52w = safe_float(parts[1].strip(), 0.0)
        if high_52w > 0 and current_price > 0:
            drawdown_pct = ((high_52w - current_price) / high_52w) * 100.0
            breakout_proximity_pct = (current_price / high_52w) * 100.0

    # 2. Solvency & Balance Sheet Integrity
    if book_value <= 0: 
        neg_flags.append("🚨 Balance Sheet: Negative Net Worth / Book Value.")
    if pledge > 5.0: 
        neg_flags.append(f"🚨 Shareholding: High Promoter Pledge at {pledge}%.")
    if debt_equity > 1.5: 
        neg_flags.append(f"🚨 Solvency: Elevated Debt-to-Equity at {debt_equity}x.")
    elif debt_equity < 0.3: 
        pos_flags.append(f"✅ Balance Sheet Fortress: Ultra-low D/E of {debt_equity}x provides strong downside protection.")

    # 3. Capital Efficiency (Current & Multi-Year Resilience)
    if roce >= 15.0 and roe >= 15.0:
        pos_flags.append(f"✅ Capital Efficiency: High compounding returns — ROCE {roce}%, ROE {roe}%.")
    elif roce < 10.0:
        neg_flags.append(f"⚠️ Capital Efficiency: Trough or sub-par return ratios — ROCE {roce}%, ROE {roe}%.")

    # 4. Operating Leverage & Growth Inflection (Sales vs PAT Acceleration)
    pnl_df = tables.get("profit-loss")
    sales = extract_trend(pnl_df, "Sales")
    net_profit = extract_trend(pnl_df, "Net Profit")
    sales_yoy, pat_yoy = 0.0, 0.0
    
    if sales and len(sales) >= 2:
        sales_yoy = ((sales[-1] - sales[-2]) / abs(sales[-2])) * 100 if sales[-2] != 0 else 0.0
    if net_profit and len(net_profit) >= 2:
        pat_yoy = ((net_profit[-1] - net_profit[-2]) / abs(net_profit[-2])) * 100 if net_profit[-2] != 0 else 0.0

    # Operating Leverage Detection: Profits accelerating significantly faster than sales
    if pat_yoy > 25.0 and pat_yoy > (sales_yoy * 1.5):
        pos_flags.append(f"🚀 Operating Leverage Active: PAT grew {pat_yoy:.1f}% vs Sales growth of {sales_yoy:.1f}%.")
    elif sales_yoy > 15.0:
        pos_flags.append(f"✅ Topline Momentum: Solid YoY Sales Growth of {sales_yoy:.1f}%.")
    elif sales_yoy < -5.0:
        neg_flags.append(f"🚨 Topline Drag: Sales contracted YoY by {sales_yoy:.1f}%.")

    # 5. J-Curve Catalyst: CWIP to Fixed Assets Shift (Upcoming Capex Monetization)
    bs_df = tables.get("balance-sheet")
    cwip_trend = extract_trend(bs_df, "CWIP") or extract_trend(bs_df, "Capital Work in Progress")
    fixed_assets_trend = extract_trend(bs_df, "Fixed assets")

    cwip_ratio = 0.0
    if cwip_trend and fixed_assets_trend and fixed_assets_trend[-1] > 0:
        cwip_ratio = (cwip_trend[-1] / fixed_assets_trend[-1]) * 100.0
        if cwip_ratio >= 15.0:
            pos_flags.append(f"🏭 Capex Inflection (J-Curve): CWIP stands at {cwip_ratio:.1f}% of Fixed Assets (Capacity expansion nearing completion).")

    # 6. Forensic Checks: Earnings Quality & Free Cash Flow
    cf_df = tables.get("cash-flow")
    cfo_trend = extract_trend(cf_df, "Operating Activity")
    capex_trend = extract_trend(cf_df, "Fixed assets purchased")

    cfo_pat_ratio = 1.0
    if net_profit and cfo_trend:
        min_len = min(len(net_profit), len(cfo_trend), 3)
        if min_len >= 3:
            sum_pat = sum(net_profit[-min_len:])
            sum_cfo = sum(cfo_trend[-min_len:])
            if sum_pat > 0:
                cfo_pat_ratio = sum_cfo / sum_pat
                if cfo_pat_ratio < 0.5:
                    neg_flags.append(f"🚨 Earnings Quality: 3-Yr cumulative CFO is only {cfo_pat_ratio*100:.0f}% of Net Profit (Paper profits).")
                elif cfo_pat_ratio >= 0.8:
                    pos_flags.append(f"✅ Cash Conversion: Strong cash generation (3-Yr CFO is {cfo_pat_ratio*100:.0f}% of Net Profit).")

    # 7. Institutional Stealth Accumulation vs Price Position
    sh_df = tables.get("shareholding")
    fii = extract_trend(sh_df, "FIIs")
    dii = extract_trend(sh_df, "DIIs")
    inst_accumulating = False
    if fii and dii and len(fii) >= 2 and len(dii) >= 2:
        inst_latest = fii[-1] + dii[-1]
        inst_prev = fii[-2] + dii[-2]
        if inst_latest > inst_prev:
            inst_accumulating = True
            pos_flags.append(f"🐋 Institutional Accumulation: Smart money increased stake from {inst_prev:.2f}% to {inst_latest:.2f}%.")
        elif (inst_prev - inst_latest) > 2.0:
            neg_flags.append(f"⚠️️ Institutional Outflow: Smart money trimmed stake by {inst_prev - inst_latest:.2f}%.")

    # 8. Shareholding & Regulatory Overhang
    promoter = extract_trend(sh_df, "Promoters")
    current_promoter_holding = 0.0
    if promoter and len(promoter) >= 1:
        current_promoter_holding = promoter[-1]
        if current_promoter_holding > 75.0:
            neg_flags.append(f"🚨 Regulatory Overhang: Promoter stake ({current_promoter_holding}%) breaches SEBI 75% limit.")
        else:
            pos_flags.append(f"✅ Stable promoter holding at {current_promoter_holding}%.")

    # --- ADVANCED DUAL-ENGINE VERDICT CLASSIFICATION ---
    verdict = "HOLD (Mixed Signals)"
    color = "gray"

    # Circuit Breakers (Overvaluation & Extreme Risk)
    if pe_ratio > 80 and roce < 15:
        verdict = "HIGH RISK (Overvalued Momentum)"
        color = "red"
    elif len(neg_flags) >= 4 or book_value <= 0:
        verdict = "AVOID / SELL"
        color = "red"
    # Setup A: Breakout Inflection (Trading in high zone + Institutional backing + Operating Leverage)
    elif breakout_proximity_pct >= 90.0 and inst_accumulating and (pat_yoy > 20.0 or sales_yoy > 15.0) and pe_ratio <= 65:
        verdict = "BUY (Breakout Inflection / High Momentum)"
        color = "#00c853"
    # Setup B: Contrarian Deep Value / Cyclical Turnaround (Downtrend + Strong Balance Sheet + Low D/E)
    elif drawdown_pct >= 25.0 and debt_equity < 0.4 and cfo_pat_ratio >= 0.75 and (pe_ratio < 25.0 or cwip_ratio >= 15.0):
        verdict = "CONTRARIAN BUY (Cyclical Turnaround / Deep Value)"
        color = "#00bcd4"
    # Setup C: Classic Steady Compounder
    elif roce >= 18.0 and sales_yoy >= 12.0 and pe_ratio <= 45 and debt_equity < 0.8:
        verdict = "BUY (Steady Compounder)"
        color = "blue"
    elif pe_ratio > 50:
        verdict = "WATCH (Valuation Stretch)"
        color = "orange"

    return {
        "verdict": verdict,
        "color": color,
        "flags": neg_flags,
        "observations": pos_flags,
        "promoter_holding": current_promoter_holding,
        "clean_metrics": {
            "Market Cap (Cr)": market_cap, "P/E": pe_ratio, "P/B": round(pb, 2),
            "ROCE %": roce, "ROE %": roe, "D/E": debt_equity, "Sales YoY %": round(sales_yoy, 2),
            "PAT YoY %": round(pat_yoy, 2), "Drawdown from 52W High %": round(drawdown_pct, 1),
            "CWIP to Fixed Assets %": round(cwip_ratio, 1)
        }
    }

# ==========================================
# 3. WEB SEARCH & MULTIBAGGER CONTEXT LAYER
# ==========================================
@st.cache_data(ttl=3600)
def fetch_market_context(ticker: str) -> str:
    try:
        news = DDGS().text(f"{ticker} stock news India latest business", max_results=2)
        deals = DDGS().text(f"{ticker} bulk deal block deal NSE BSE", max_results=1)
        concall = DDGS().text(f"{ticker} earnings call transcript management guidance capex order book expansion", max_results=3)
        
        context = "Live Market News:\n" + "\n".join([r['body'] for r in news]) if news else ""
        context += "\n\nBlock/Bulk Deals:\n" + "\n".join([r['body'] for r in deals]) if deals else ""
        context += "\n\nConcall & Guidance Insights:\n" + "\n".join([r['body'] for r in concall]) if concall else ""
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
        
        overhang_pct = max(0.0, promoter_holding - 75.0)
        overhang_cr = (overhang_pct / 100) * metrics.get("Market Cap (Cr)", 0)
        
        prompt = f"""
You are a senior institutional equity portfolio manager evaluating {ticker}.
Reconcile your thesis directly with the Quantitative Engine's verdict: **{quant_verdict}**.

--- QUANTITATIVE METRICS & TECHNICAL POSITION ---
- Valuation: P/E {metrics['P/E']}x | P/B {metrics['P/B']}x | Market Cap: ₹{metrics['Market Cap (Cr)']} Cr
- Quality & Solvency: ROCE {metrics['ROCE %']}% | ROE {metrics['ROE %']}% | D/E {metrics['D/E']}x
- Inflection Indicators: Sales YoY {metrics['Sales YoY %']}% | PAT YoY {metrics['PAT YoY %']}% | CWIP/Fixed Assets: {metrics['CWIP to Fixed Assets %']}%
- Chart Position: Drawdown from 52-Week High: {metrics['Drawdown from 52W High %']}%
- Strengths Detected: {observations}
- Risks Detected: {flags}

--- MANAGEMENT GUIDANCE & CONCALL TRANSCRIPT CONTEXT ---
{market_context}

--- REQUIRED MULTIBAGGER THESIS STRUCTURE ---
1. Structural Context & Valuation: Address whether this is a Steady Compounder, a Breakout Setup, or a Beaten-down Cyclical Turnaround. Reconcile current valuation multiples against earnings velocity.
2. Inflection Catalysts (CWIP / Operating Leverage): Evaluate if upcoming capex commercialization or margin expansion justifies multi-year re-rating.
3. Supply Overhang: If promoter holding ({promoter_holding}%) exceeds 75%, explicitly state the {overhang_pct:.2f}% excess stake (approx ₹{overhang_cr:.2f} Cr) creating a supply ceiling.
4. Final Institutional Stance: Conclude with a definitive stance (BUY, CONTRARIAN BUY, WATCH, or AVOID) strictly reconciled with the quantitative verdict.

FORMATTING RULE: Wrap all strengths and catalysts in :green[text] and all risks, valuation flags, or dilution hurdles in :red[text].
"""
        config = types.GenerateContentConfig(temperature=0.2)
        
        try:
            return generate_content_with_backoff(client, prompt, config, 'gemini-3.5-flash').text
        except Exception as primary_error:
            if is_rate_limit_error(primary_error):
                return generate_content_with_backoff(client, prompt, config, 'gemini-3.1-flash-lite').text
            raise primary_error
    except Exception as e:
        return f"⚠️ AI thesis generation failed: {str(e)}"

# ==========================================
# 4. STREAMLIT UI DASHBOARD
# ==========================================
st.set_page_config(page_title="Fundamental Screener", page_icon="📈", layout="wide")
st.title("Fundamental Screener")

col_search, _ = st.columns([1, 2])
with col_search:
    ticker_input = st.text_input("🔍 Enter NSE/BSE Symbol (e.g., TATASTEEL, ITC, UNIMECH):", "")

if st.button("Run Quantitative Analysis") and ticker_input:
    with st.spinner(f"Auditing fundamentals, capex cycle, and price position for {ticker_input.upper()}..."):
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
    
    # Primary Metrics Bar
    cols = st.columns(6)
    cols[0].metric("Market Cap (Cr)", f"{metrics['Market Cap (Cr)']:,.0f}")
    cols[1].metric("P/E", metrics["P/E"])
    cols[2].metric("ROCE", f"{metrics['ROCE %']}%")
    cols[3].metric("PAT YoY", f"{metrics['PAT YoY %']}%")
    cols[4].metric("52W Drawdown", f"-{metrics['Drawdown from 52W High %']}%")
    cols[5].metric("CWIP / Fixed Assets", f"{metrics['CWIP to Fixed Assets %']}%")
    st.divider()

    tab1, tab2 = st.tabs(["📊 Trend & Inflection Audit", "🧠 Multibagger AI Thesis"])
    
    with tab1:
        ui_col1, ui_col2 = st.columns(2)
        
        with ui_col1:
            st.success("✅ Positive Observations & Growth Catalysts")
            if not eval_results["observations"]: 
                st.write("No major catalysts detected.")
            for obs in eval_results["observations"]:
                st.write(obs)
                
        with ui_col2:
            st.error("⚠️ Risk Factors, Friction & Valuation Overhang")
            if not eval_results["flags"]: 
                st.write("No major red flags detected.")
            for flag in eval_results["flags"]:
                st.write(flag)
            
    with tab2:
        st.write("The AI thesis reconciles the quantitative audit with concall transcripts, operating leverage, and capex cycles.")
        if "AVOID" in eval_results["verdict"] or "HIGH RISK" in eval_results["verdict"]:
            st.warning("⚠️ This stock tripped quantitative risk breakers. Review the risks above before requesting an AI thesis.")
            
        if st.button("Generate Deep Multibagger Thesis 🧠"):
            with st.spinner("Extracting concall commentary, verifying capex milestones, and synthesizing institutional stance..."):
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
