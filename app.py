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

# -----------------------------------------
# 1. DATA EXTRACTION ENGINE (Screener.in)
# -----------------------------------------
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
# 2. QUANTITATIVE SCORING ENGINE
# -----------------------------------------
def evaluate_fundamentals(data: dict) -> dict:
    if "error" in data:
        return {"verdict": "ERROR", "reasons": [data["error"]]}

    reasons, red_flags = [], []
    score = 0
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

    if book_value <= 0:
        red_flags.append("🚨 Point 1 (Balance Sheet): Negative Net Worth / Book Value is negative.")
    if pledge > 5.0:
        red_flags.append(f"🚨 Point 2 (Shareholding): Significant Promoter Pledge at {pledge}%.")
    if debt_equity > 1.5:
        red_flags.append(f"🚨 Point 6 (Solvency): High Debt-to-Equity at {debt_equity}x.")
    
    if roce >= 15.0 and roe >= 15.0:
        score += 15
        reasons.append(f"✅ Point 1 (Capital Efficiency): Strong consistency — ROCE {roce}%, ROE {roe}%.")
    elif roce > 0 or roe > 0:
        reasons.append(f"⚠️ Point 1 (Capital Efficiency): Sub-par ROCE ({roce}%) or ROE ({roe}%).")

    pnl_df = tables.get("profit-loss")
    sales = extract_trend(pnl_df, "Sales")
    if sales and len(sales) >= 2:
        sales_yoy = ((sales[-1] - sales[-2]) / abs(sales[-2])) * 100 if sales[-2] != 0 else 0
        if sales_yoy > 15:
            score += 10
            reasons.append(f"✅ Point 3 (Topline): Excellent YoY Sales Growth of {sales_yoy:.1f}%.")
        elif sales_yoy > 0:
            reasons.append(f"⚠️ Point 3 (Topline): Moderate YoY Sales Growth of {sales_yoy:.1f}%.")
        else:
            red_flags.append(f"🚨 Point 3 (Topline): Sales declined YoY by {sales_yoy:.1f}%.")

    q_df = tables.get("quarters")
    opm_trend = extract_trend(q_df, "OPM")
    if opm_trend and len(opm_trend) >= 4:
        recent_opm = opm_trend[-4:]
        if recent_opm[-1] > recent_opm[0]:
            score += 15
            reasons.append(f"✅ Point 4 (Quarterly Inflection): OPM expanded from {recent_opm[0]}% to {recent_opm[-1]}% recently.")
        else:
            reasons.append(f"⚠️ Point 4 (Quarterly Results): OPM margin trend is flat or contracting ({recent_opm[-1]}%).")

    net_profit = extract_trend(pnl_df, "Net Profit")
    if net_profit:
        if len(net_profit) >= 2:
            profit_yoy = ((net_profit[-1] - net_profit[-2]) / abs(net_profit[-2])) * 100 if net_profit[-2] != 0 else 0
            if profit_yoy > 15:
                score += 15
                reasons.append(f"✅ Point 5 (Bottomline): Superb YoY Net Profit Growth of {profit_yoy:.1f}%.")
            elif profit_yoy < 0:
                red_flags.append(f"🚨 Point 5 (Bottomline): Net Profit declined YoY by {profit_yoy:.1f}%.")
        
        profitable_years = sum(1 for p in net_profit if p > 0)
        total_years = len(net_profit)
        if total_years > 0 and profitable_years <= (total_years / 2):
            red_flags.append(f"🚨 Point 5 (P&L): Chronic losses — profitable in only {profitable_years} of {total_years} years.")

    bs_df = tables.get("balance-sheet")
    reserves = extract_trend(bs_df, "Reserves")
    if reserves and len(reserves) >= 2:
        if reserves[-1] < reserves[-2]:
            red_flags.append("🚨 Point 6 (Balance Sheet): Shrinking accumulated reserves.")
        else:
            score += 10
            reasons.append("✅ Point 6 (Balance Sheet): Reserves are expanding.")

    cf_df = tables.get("cash-flow")
    cfo = extract_trend(cf_df, "Operating Activity")
    if cfo:
        positive_cfo = sum(1 for c in cfo if c > 0)
        if positive_cfo < len(cfo) / 2:
            red_flags.append("🚨 Point 7 (Cash Flow): Operating cash flow is negative in most reported periods.")
        else:
            score += 10
            reasons.append(f"✅ Point 7 (Cash Generation): Positive CFO in {positive_cfo} of {len(cfo)} recorded years.")

    sh_df = tables.get("shareholding")
    fii = extract_trend(sh_df, "FIIs")
    dii = extract_trend(sh_df, "DIIs")
    if fii and dii and len(fii) >= 2 and len(dii) >= 2:
        inst_latest = fii[-1] + dii[-1]
        inst_prev = fii[-2] + dii[-2]
        if inst_latest > inst_prev:
            score += 15
            reasons.append(f"✅ Point 8 (Smart Money): FII/DII accumulating, total stake increased to {inst_latest:.2f}%.")
        else:
            reasons.append(f"⚠️ Point 8 (Smart Money): FII/DII stake decreased or remained flat at {inst_latest:.2f}%.")

    promoter = extract_trend(sh_df, "Promoters")
    if promoter and len(promoter) >= 4:
        if promoter[-1] < promoter[-4]:
            red_flags.append(f"🚨 Point 2 (Shareholding): Promoters trimming stake ({promoter[-4]}% down to {promoter[-1]}%).")
        elif promoter[-1] >= 45.0:
            score += 10
            reasons.append(f"✅ Point 2 (Shareholding): Stable promoter backing at {promoter[-1]}%.")

    verdict = "WATCH"
    color = "orange"
    if red_flags:
        verdict = "AVOID / SELL"
        color = "red"
    elif score >= 75 and pb <= 6.0:
        verdict = "MULTIBAGGER CANDIDATE / BUY"
        color = "green"
    elif score >= 50:
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
# 3. WEB SEARCH & AI LAYER (CASCADING FAILOVER)
# -----------------------------------------
@st.cache_data(ttl=3600)
def fetch_live_news(ticker: str) -> str:
    try:
        news_results = DDGS().text(f"{ticker} stock news India latest business", max_results=3)
        deals_results = DDGS().text(f"{ticker} bulk deal block deal NSE BSE latest", max_results=2)
        news_text = "Recent News:\n" + "\n".join([f"- {r['title']}: {r['body']}" for r in news_results]) if news_results else ""
        deals_text = "\nBulk/Block Deals:\n" + "\n".join([f"- {r['title']}: {r['body']}" for r in deals_results]) if deals_results else ""
        return news_text + "\n" + deals_text
    except Exception as e:
        return f"Web news bypassed: {str(e)}"

def is_rate_limit_error(exception):
    err_str = str(exception)
    return any(err in err_str for err in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"])

@retry(wait=wait_random_exponential(multiplier=2, max=10), stop=stop_after_attempt(3), retry=retry_if_exception(is_rate_limit_error), reraise=True)
def generate_content_with_backoff(client, prompt, config, model_name):
    return client.models.generate_content(model=model_name, contents=prompt, config=config)

@st.cache_data(ttl=3600)
def get_ai_verdict(ticker: str, metrics: dict, flags: list, observations: list) -> str:
    api_key = st.secrets.get("GEMINI_API_KEY", None)
    if not api_key: return "⚠️ Gemini API key not found."
        
    try:
        live_news = fetch_live_news(ticker)
        client = genai.Client(api_key=api_key)
        
        prompt = f"""
You are a senior institutional equity analyst. Evaluate this Indian stock: {ticker}.

Quantitative Scorecard:
- Metrics: {metrics}
- Detected Red Flags: {flags}
- Trend Observations: {observations}

Live Market & Bulk Deals:
{live_news}

Task:
1. Peer & Industry Standing: Compare its competitive moat against sector peers.
2. Governance & Catalysts: Highlight management updates, credit rating notes, or recent bulk/block deals.
3. Multibagger Inflection Thesis: Is there an active catalyst (capex, margin inflection) or is it a steady compounder?
4. Conclusion: A 3-bullet summary justifying BUY, SELL, AVOID, WATCH, or MULTIBAGGER.

CRITICAL FORMATTING INSTRUCTION:
You MUST use Streamlit color markdown to highlight positives and negatives throughout your entire response:
- Wrap all positive factors, strengths, and bull arguments in :green[text].
- Wrap all negative factors, risks, red flags, and bear arguments in :red[text].
"""
        config = types.GenerateContentConfig(temperature=0.2)
        
        try:
            # 1. ALWAYS TRY THE PRIMARY HIGH-CAPACITY MODEL FIRST
            response = generate_content_with_backoff(client, prompt, config, model_name='gemini-3.5-flash')
            return response.text
        except Exception as primary_error:
            # 2. CATCH QUOTA (429) OR CAPACITY (503) ERRORS
            if any(err in str(primary_error) for err in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"]):
                st.warning("⚠️ Primary AI Quota Exhausted. Auto-routing to secondary Lite model...")
                try:
                    # 3. IMMEDIATELY FALLBACK TO FLASH-LITE
                    response_fallback = generate_content_with_backoff(client, prompt, config, model_name='gemini-3.1-flash-lite')
                    return response_fallback.text
                except Exception as fallback_error:
                    st.error("⚠️ **Total Quota Exhausted**: Both primary and secondary models have reached their Free Tier limits for today.")
                    return "AI analysis could not be completed."
            else:
                st.error(f"⚠️ **API Error**: {str(primary_error)}")
                return "AI analysis could not be completed."
    except Exception as general_error:
        st.error(f"⚠️ **System Error**: {str(general_error)}")
        return "AI analysis could not be completed."

# -----------------------------------------
# 4. STREAMLIT UI DASHBOARD
# -----------------------------------------
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
        # Store raw data in session state so we don't lose it when generating the AI report later
        st.session_state['ticker'] = ticker_input
        st.session_state['raw_data'] = raw_data
        st.session_state['eval_results'] = evaluate_fundamentals(raw_data)
        st.session_state['ai_insight_generated'] = False

# Render the dashboard if data exists in session state
if 'eval_results' in st.session_state:
    eval_results = st.session_state['eval_results']
    metrics = eval_results["clean_metrics"]
    raw_data = st.session_state['raw_data']
    ticker = st.session_state['ticker']
    
    st.markdown(f"<h3 style='color: {eval_results['color']};'>Verdict: {eval_results['verdict']}</h3>", unsafe_allow_html=True)
    st.progress(eval_results['score'] / 100)
    st.caption(f"Health Score: {eval_results['score']}/100")
    
    cols = st.columns(6)
    market_cap_formatted = f"{metrics['Market Cap (Cr)']:,.0f}"
    cols[0].metric("Market Cap (Cr)", market_cap_formatted)
    cols[1].metric("P/E", metrics["P/E"])
    cols[2].metric("P/B", metrics["P/B"])
    cols[3].metric("ROCE", f"{metrics['ROCE %']}%")
    cols[4].metric("ROE", f"{metrics['ROE %']}%")
    cols[5].metric("Debt/Equity", f"{metrics['D/E']}x")
    st.divider()

    tab1, tab2 = st.tabs(["📊 Trend Audit", "🧠 AI Analyst Thesis (On-Demand)"])
    
    with tab1:
        if eval_results["flags"]:
            st.error("🚨 Red Flags Triggered")
            for flag in eval_results["flags"]:
                st.write(flag)
        
        st.success("✅ Positive Observations")
        for obs in eval_results["observations"]:
            st.write(obs)
            
    with tab2:
        st.write("Generating an AI thesis consumes API quota. Ensure the fundamental metrics look promising before proceeding.")
        
        if eval_results["verdict"] == "AVOID / SELL":
            st.warning("⚠️ This stock failed the quantitative screen. Running AI analysis is not recommended, but you can override below.")
            
        # The user must click this button to actually hit the Gemini API
        if st.button("Generate Deep AI Thesis 🧠"):
            with st.spinner("Fetching live news, bulk deals, and synthesizing final investment thesis..."):
                ai_insight = get_ai_verdict(
                    ticker=raw_data["ticker"],
                    metrics=metrics,
                    flags=eval_results["flags"],
                    observations=eval_results["observations"]
                )
                # Store the result in session state so it doesn't vanish if the user interacts with other parts of the app
                st.session_state['ai_insight_generated'] = ai_insight
                
        # Display the AI insight if it was successfully generated
        if st.session_state.get('ai_insight_generated'):
            st.markdown(st.session_state['ai_insight_generated'])
