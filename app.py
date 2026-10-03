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
            val_clean = val_elem.text.strip().replace(",", "").replace("₹", "").replace("Rs.", "").strip()
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

def extract_trend(df, keyword: str, exclude_ttm: bool = False):
    """
    Extracts time-series rows while optionally filtering out the 'TTM' column 
    to maintain strict fiscal-year alignment with Cash Flow statements.
    """
    if df is None or df.empty: 
        return []
    try:
        row = df[df.iloc[:, 0].astype(str).str.contains(keyword, case=False, na=False)]
        if row.empty: 
            return []
        
        cols = list(df.columns[1:])
        row_vals = row.iloc[0, 1:]
        
        vals = []
        for col, val in zip(cols, row_vals):
            # Temporal fix: Ignore TTM when comparing historical fiscal years with Cash Flow
            if exclude_ttm and "ttm" in str(col).lower():
                continue
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
# 2. ADVANCED INFLECTION & AUDITING ENGINE
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

    # Core Financial Metrics
    book_value = safe_float(data.get("book value"), 1.0)
    current_price = safe_float(data.get("current price"), 0.0)
    market_cap = safe_float(data.get("market cap"), 0.0)
    pe_ratio = safe_float(data.get("stock p/e"), 0.0)
    roce = safe_float(data.get("roce"), 0.0)
    roe = safe_float(data.get("roe"), 0.0)
    debt_equity = safe_float(data.get("debt to equity"), 0.0)
    pledge = safe_float(data.get("pledged percentage"), 0.0)
    pb = (current_price / book_value) if book_value > 0 else 999.0

    # 1. 52-Week High / Low & Breakout Proximity
    high_52w, low_52w = 0.0, 0.0
    drawdown_pct = 0.0
    breakout_proximity_pct = 0.0
    high_low_str = data.get("high / low", "")
    if "/" in high_low_str:
        parts = high_low_str.split("/")
        def clean_val(s):
            clean = s.replace(",", "").replace("₹", "").replace("Rs.", "").strip()
            try: return float(clean)
            except: return 0.0
        high_52w = clean_val(parts[0])
        low_52w = clean_val(parts[1]) if len(parts) > 1 else 0.0
        if high_52w > 0 and current_price > 0:
            drawdown_pct = max(0.0, ((high_52w - current_price) / high_52w) * 100.0)
            breakout_proximity_pct = min(100.0, (current_price / high_52w) * 100.0)

    # 2. Solvency & Debt Servicing (Interest Coverage Check)
    if book_value <= 0: 
        neg_flags.append("🚨 Balance Sheet: Negative Net Worth / Book Value.")
    if pledge > 5.0: 
        neg_flags.append(f"🚨 Shareholding: High Promoter Pledge at {pledge}%.")
    if debt_equity > 1.5: 
        neg_flags.append(f"🚨 Solvency: Elevated Debt-to-Equity at {debt_equity}x.")
    elif debt_equity < 0.3: 
        pos_flags.append(f"✅ Balance Sheet Fortress: Ultra-low D/E of {debt_equity}x.")

    pnl_df = tables.get("profit-loss")
    op_profit = extract_trend(pnl_df, "Operating Profit")
    interest = extract_trend(pnl_df, "Interest")
    if op_profit and interest and interest[-1] > 0:
        interest_coverage = op_profit[-1] / interest[-1]
        if interest_coverage < 2.5:
            neg_flags.append(f"🚨 Debt Servicing: Low Interest Coverage ({interest_coverage:.1f}x < 2.5x). Vulnerable if earnings soften.")
        elif interest_coverage >= 5.0:
            pos_flags.append(f"✅ Debt Servicing: Strong Interest Coverage ({interest_coverage:.1f}x).")

    # 3. Capital Efficiency Checks
    if roce >= 15.0 and roe >= 15.0:
        pos_flags.append(f"✅ Capital Efficiency: Robust compounding returns — ROCE {roce}%, ROE {roe}%.")
    elif roce < 10.0:
        neg_flags.append(f"⚠️ Capital Efficiency: Trough or sub-par return ratios — ROCE {roce}%, ROE {roe}%.")

    # 4. Annual Operating Leverage (Sales vs PAT Acceleration)
    sales = extract_trend(pnl_df, "Sales")
    net_profit = extract_trend(pnl_df, "Net Profit")
    sales_yoy, pat_yoy = 0.0, 0.0
    
    if sales and len(sales) >= 2:
        sales_yoy = ((sales[-1] - sales[-2]) / abs(sales[-2])) * 100 if sales[-2] != 0 else 0.0
    if net_profit and len(net_profit) >= 2:
        pat_yoy = ((net_profit[-1] - net_profit[-2]) / abs(net_profit[-2])) * 100 if net_profit[-2] != 0 else 0.0

    if pat_yoy > 25.0 and pat_yoy > (sales_yoy * 1.5):
        pos_flags.append(f"🚀 Annual Operating Leverage: PAT surged {pat_yoy:.1f}% vs Sales growth of {sales_yoy:.1f}%.")
    elif sales_yoy > 15.0:
        pos_flags.append(f"✅ Annual Topline: Solid YoY Sales Growth of {sales_yoy:.1f}%.")
    elif sales_yoy < -5.0:
        neg_flags.append(f"🚨 Topline Drag: Annual sales contracted YoY by {sales_yoy:.1f}%.")

    # 5. Real-Time Quarterly Inflection Check
    q_df = tables.get("quarters")
    q_sales = extract_trend(q_df, "Sales")
    q_pat = extract_trend(q_df, "Net Profit")
    q_opm = extract_trend(q_df, "OPM")
    
    q_sales_yoy, q_pat_yoy = 0.0, 0.0
    if q_sales and len(q_sales) >= 5:
        q_sales_yoy = ((q_sales[-1] - q_sales[-5]) / abs(q_sales[-5])) * 100 if q_sales[-5] != 0 else 0.0
    if q_pat and len(q_pat) >= 5:
        q_pat_yoy = ((q_pat[-1] - q_pat[-5]) / abs(q_pat[-5])) * 100 if q_pat[-5] != 0 else 0.0
        
    if q_pat_yoy > 30.0 and q_sales_yoy > 15.0:
        pos_flags.append(f"🔥 Quarterly Inflection: Latest Qtr PAT surged {q_pat_yoy:.1f}% YoY on {q_sales_yoy:.1f}% Sales growth.")
    elif q_pat_yoy < -20.0:
        neg_flags.append(f"⚠️ Quarterly Deterioration: Latest Qtr PAT dropped {q_pat_yoy:.1f}% YoY.")

    if q_opm and len(q_opm) >= 4:
        recent_opm = q_opm[-4:]
        if recent_opm[-1] > recent_opm[0]:
            pos_flags.append(f"✅ Margin Expansion: OPM expanded from {recent_opm[0]}% to {recent_opm[-1]}% across recent quarters.")

    # 6. J-Curve Catalyst: CWIP to Fixed Assets
    bs_df = tables.get("balance-sheet")
    cwip_trend = extract_trend(bs_df, "CWIP") or extract_trend(bs_df, "Capital Work in Progress")
    fixed_assets_trend = extract_trend(bs_df, "Fixed assets")

    cwip_ratio = 0.0
    if cwip_trend and fixed_assets_trend and fixed_assets_trend[-1] > 0:
        cwip_ratio = (cwip_trend[-1] / fixed_assets_trend[-1]) * 100.0
        if cwip_ratio >= 15.0:
            pos_flags.append(f"🏭 Capex Inflection (J-Curve): CWIP is {cwip_ratio:.1f}% of Fixed Assets (Upcoming capacity expansion).")

    # 7. Forensic Checks: Earnings Quality (Strict Fiscal Year Alignment)
    cf_df = tables.get("cash-flow")
    net_profit_annual = extract_trend(pnl_df, "Net Profit", exclude_ttm=True)
    cfo_annual = extract_trend(cf_df, "Operating Activity")

    cfo_pat_ratio = 1.0
    if net_profit_annual and cfo_annual:
        min_len = min(len(net_profit_annual), len(cfo_annual), 3)
        if min_len >= 3:
            sum_pat = sum(net_profit_annual[-min_len:])
            sum_cfo = sum(cfo_annual[-min_len:])
            if sum_pat > 0:
                cfo_pat_ratio = sum_cfo / sum_pat
                if cfo_pat_ratio < 0.5:
                    neg_flags.append(f"🚨 Earnings Quality: 3-Yr cumulative CFO is only {cfo_pat_ratio*100:.0f}% of PAT (Paper profits).")
                elif cfo_pat_ratio >= 0.8:
                    pos_flags.append(f"✅ Cash Conversion: Reliable conversion (3-Yr CFO is {cfo_pat_ratio*100:.0f}% of PAT).")

    # 8. Institutional Smart Money Movement
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

    # 9. Promoter Ownership & SEBI MPS Limit
    promoter = extract_trend(sh_df, "Promoters")
    current_promoter_holding = 0.0
    if promoter and len(promoter) >= 1:
        current_promoter_holding = promoter[-1]
        if current_promoter_holding > 75.0:
            neg_flags.append(f"🚨 Regulatory Overhang: Promoter stake ({current_promoter_holding}%) breaches SEBI 75% limit.")
        else:
            pos_flags.append(f"✅ Stable promoter holding at {current_promoter_holding}%.")

    # --- CLASSIFICATION & CIRCUIT BREAKERS ---
    verdict = "HOLD (Mixed Signals)"
    color = "gray"

    if pe_ratio <= 0.0:
        neg_flags.append("🚨 Valuation / Earnings: Negative or unlisted P/E (Loss-making or nil EPS).")

    # Circuit Breakers (Overvaluation & Capital Waste)
    if pe_ratio > 80 and roce < 15:
        verdict = "HIGH RISK (Overvalued Momentum)"
        color = "red"
    elif len(neg_flags) >= 4 or book_value <= 0:
        verdict = "AVOID / SELL"
        color = "red"
    # Setup A: Breakout Inflection (Near 52W Highs + Smart Money + Earnings Velocity)
    elif breakout_proximity_pct >= 90.0 and inst_accumulating and (q_pat_yoy > 25.0 or pat_yoy > 20.0) and (0 < pe_ratio <= 65):
        verdict = "BUY (Breakout Inflection / High Momentum)"
        color = "#00c853"
    # Setup B: Contrarian Deep Value / Turnaround (Patched: Strictly requires positive P/E or Net Cash)
    elif drawdown_pct >= 25.0 and debt_equity < 0.4 and cfo_pat_ratio >= 0.75 and ((0 < pe_ratio <= 25.0) or (cwip_ratio >= 15.0 and debt_equity < 0.2)):
        verdict = "CONTRARIAN BUY (Cyclical Turnaround / Deep Value)"
        color = "#00bcd4"
    # Setup C: Steady Compounder
    elif roce >= 18.0 and sales_yoy >= 12.0 and (0 < pe_ratio <= 45) and debt_equity < 0.8:
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
            "PAT YoY %": round(pat_yoy, 2), "Quarterly PAT YoY %": round(q_pat_yoy, 1),
            "Drawdown from 52W High %": round(drawdown_pct, 1),
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
- Inflection Indicators: Annual PAT YoY {metrics['PAT YoY %']}% | Qtr PAT YoY {metrics['Quarterly PAT YoY %']}% | CWIP/Fixed Assets: {metrics['CWIP to Fixed Assets %']}%
- Chart Position: Drawdown from 52-Week High: {metrics['Drawdown from 52W High %']}%
- Strengths Detected: {observations}
- Risks Detected: {flags}

--- MANAGEMENT GUIDANCE & CONCALL TRANSCRIPT CONTEXT ---
{market_context}

--- REQUIRED MULTIBAGGER THESIS STRUCTURE ---
1. Structural Context & Valuation: Address whether this is a Steady Compounder, a Breakout Setup, or a Beaten-down Cyclical Turnaround. Reconcile current valuation multiples against earnings velocity.
2. Inflection Catalysts (CWIP / Quarterly Growth): Evaluate if recent quarterly acceleration or upcoming capex commercialization justifies a multi-year re-rating.
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
    
    cols = st.columns(6)
    cols[0].metric("Market Cap (Cr)", f"{metrics['Market Cap (Cr)']:,.0f}")
    cols[1].metric("P/E", metrics["P/E"])
    cols[2].metric("ROCE", f"{metrics['ROCE %']}%")
    cols[3].metric("Qtr PAT YoY", f"{metrics['Quarterly PAT YoY %']}%")
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
            st.error("⚠️️ Risk Factors, Friction & Valuation Overhang")
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
