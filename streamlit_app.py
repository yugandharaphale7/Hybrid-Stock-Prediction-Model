import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Hybrid Nifty 50 Backtest Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- PATHS ----------
FOLDER = Path(__file__).parent
RESULTS_DIR = FOLDER / "backtest_results"
PNG_PATH = RESULTS_DIR / "equity_curve.png"
JSON_PATH = RESULTS_DIR / "results.json"
REPORT_PATH = RESULTS_DIR / "BACKTEST_REPORT.txt"

# ---------- HARD-CACHED REPORT DATA ----------
METRICS = {
    "Total return (%)":   {"FULL": 133.7,  "NO-DCF": 32.2,   "Nifty B&H": 194.4},
    "CAGR (%)":           {"FULL": 8.2,    "NO-DCF": 2.6,    "Nifty B&H": 10.5},
    "Max drawdown (%)":   {"FULL": -26.6,  "NO-DCF": -29.5,  "Nifty B&H": -38.4},
    "Sharpe (rf=0)":      {"FULL": 0.76,   "NO-DCF": 0.31,   "Nifty B&H": 0.71},
    "Win rate (%)":       {"FULL": 33.9,   "NO-DCF": 32.3,   "Nifty B&H": None},
    "Profit factor":      {"FULL": 1.17,   "NO-DCF": 1.06,   "Nifty B&H": None},
    "Trades filled":      {"FULL": 2665,   "NO-DCF": 2557,   "Nifty B&H": None},
    "Final equity (Rs.)": {"FULL": 233680, "NO-DCF": 132198, "Nifty B&H": 294400},
}

PER_YEAR = pd.DataFrame({
    "Year":   [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026],
    "FULL":   [2.2,  27.2, -1.5, -1.3, 47.3, 16.5, -7.1,  23.1, 10.4, -1.9, -13.0],
    "NO-DCF": [1.1,  31.5, -2.7,  0.6,  4.3,  1.6, -11.6, 15.6, 12.4, -3.6, -13.3],
})

def cum_equity(returns, start=100000):
    e = [start]
    for r in returns:
        e.append(e[-1] * (1 + r/100))
    return e

YEARS_X = [2015] + PER_YEAR["Year"].tolist()
CUM_FULL = cum_equity(PER_YEAR["FULL"].tolist())
CUM_NODCF = cum_equity(PER_YEAR["NO-DCF"].tolist())

TOP_WINS = pd.DataFrame([
    {"Symbol":"ADANIENT", "Side":"LONG", "Entry":"2021-03-17", "Exit":"2021-03-23", "Return %":20.99, "Exit":"TP", "Score":68},
    {"Symbol":"ETERNAL",  "Side":"LONG", "Entry":"2024-03-18", "Exit":"2024-04-08", "Return %":18.43, "Exit":"TP", "Score":62},
    {"Symbol":"ETERNAL",  "Side":"LONG", "Entry":"2024-02-02", "Exit":"2024-02-20", "Return %":14.42, "Exit":"TP", "Score":78},
    {"Symbol":"WIPRO",    "Side":"LONG", "Entry":"2024-01-17", "Exit":"2024-02-16", "Return %":14.30, "Exit":"TP", "Score":80},
    {"Symbol":"ETERNAL",  "Side":"LONG", "Entry":"2024-07-10", "Exit":"2024-08-01", "Return %":13.70, "Exit":"TP", "Score":63},
])

TOP_LOSSES = pd.DataFrame([
    {"Symbol":"ADANIPORTS","Side":"LONG","Entry":"2024-12-04","Exit":"2024-12-23","Return %":-9.85,"Exit":"SL","Score":56},
    {"Symbol":"TRENT",     "Side":"LONG","Entry":"2020-12-08","Exit":"2020-12-21","Return %":-9.39,"Exit":"SL","Score":70},
    {"Symbol":"TMPV",      "Side":"LONG","Entry":"2021-03-12","Exit":"2021-03-19","Return %":-8.30,"Exit":"SL","Score":69},
    {"Symbol":"KOTAKBANK", "Side":"LONG","Entry":"2024-04-10","Exit":"2024-04-25","Return %":-7.12,"Exit":"SL","Score":64},
    {"Symbol":"ETERNAL",   "Side":"LONG","Entry":"2024-10-07","Exit":"2024-10-07","Return %":-6.26,"Exit":"SL","Score":60},
])

EXIT_DIST = pd.DataFrame({
    "Exit Type": ["Take Profit", "Stop Loss", "Time Stop"],
    "Count":     [520, 1655, 490],
})

LONG_SHORT = pd.DataFrame({
    "Side":      ["Long",  "Short"],
    "Trades":    [2274,    391],
    "PnL (Rs.)": [185412,  -52127],
})

COLORS = {"FULL": "#1f77b4", "NO-DCF": "#ff7f0e", "Nifty B&H": "#2ca02c"}

def color_return(val):
    if isinstance(val, (int, float)):
        if val > 0: return "color: #2ca02c; font-weight: bold"
        if val < 0: return "color: #d62728; font-weight: bold"
    return ""

# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("Hybrid Nifty 50")
    st.caption("ML + DCF + Smart-Money + Regime engine")
    st.divider()
    st.subheader("Data files")
    for name, path in [("equity_curve.png", PNG_PATH),
                       ("BACKTEST_REPORT.txt", REPORT_PATH),
                       ("results.json", JSON_PATH)]:
        st.markdown(f"{'OK' if path.exists() else 'MISSING'} - `{name}`")
    st.divider()
    st.subheader("Variant")
    variant = st.radio("Select variant", ["FULL", "NO-DCF"], horizontal=True,
                       help="FULL = full signal incl. DCF; NO-DCF = DCF factor zeroed (honest core)")
    st.divider()
    st.caption("10.8-yr walk-forward  |  2,665 trades  |  Rs.100k start")

# ---------- HEADER ----------
st.title("Hybrid Nifty 50 Backtest Dashboard")
st.caption("Production-pipeline simulation: ML engine + DCF valuation + smart-money flow + regime filter, walked weekly over 2016-2026.")

# ---------- KPI STRIP ----------
st.subheader("Headline KPIs")
k1, k2, k3, k4 = st.columns(4)
v = METRICS["Total return (%)"][variant]; b = METRICS["Total return (%)"]["Nifty B&H"]
k1.metric("Total Return", f"{v:.1f}%", delta=f"{v-b:+.1f} vs Nifty")
v = METRICS["CAGR (%)"][variant]; b = METRICS["CAGR (%)"]["Nifty B&H"]
k2.metric("CAGR", f"{v:.1f}%", delta=f"{v-b:+.1f} vs Nifty")
v = METRICS["Sharpe (rf=0)"][variant]; b = METRICS["Sharpe (rf=0)"]["Nifty B&H"]
k3.metric("Sharpe", f"{v:.2f}", delta=f"{v-b:+.2f} vs Nifty")
v = METRICS["Max drawdown (%)"][variant]; b = METRICS["Max drawdown (%)"]["Nifty B&H"]
k4.metric("Max Drawdown", f"{v:.1f}%", delta=f"{abs(v)-abs(b):+.1f}pp vs Nifty", delta_color="inverse")

st.divider()

# ---------- TABS ----------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Performance", "Trades", "Methodology"])

# ===== TAB 1: OVERVIEW =====
with tab1:
    st.subheader("Equity Curve (report-generated)")
    if PNG_PATH.exists():
        st.image(str(PNG_PATH), use_container_width=True)
    else:
        st.warning("equity_curve.png not found in backtest_results/")

    st.subheader("Variant Comparison")
    comp = pd.DataFrame(METRICS).T.reset_index().rename(columns={"index": "Metric"})
    comp = comp.fillna("—")
    st.dataframe(comp, use_container_width=True, hide_index=True)

    st.subheader("Final Equity")
    c1, c2, c3 = st.columns(3)
    c1.metric("FULL",      f"Rs. {METRICS['Final equity (Rs.)']['FULL']:,}")
    c2.metric("NO-DCF",    f"Rs. {METRICS['Final equity (Rs.)']['NO-DCF']:,}")
    c3.metric("Nifty B&H", f"Rs. {METRICS['Final equity (Rs.)']['Nifty B&H']:,}")

# ===== TAB 2: PERFORMANCE =====
with tab2:
    st.subheader("Cumulative Equity Curve")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=YEARS_X, y=CUM_FULL, mode="lines+markers",
                             name="FULL", line=dict(color=COLORS["FULL"], width=3)))
    fig.add_trace(go.Scatter(x=YEARS_X, y=CUM_NODCF, mode="lines+markers",
                             name="NO-DCF", line=dict(color=COLORS["NO-DCF"], width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=[YEARS_X[0], YEARS_X[-1]], y=[100000, 294400], mode="lines",
                             name="Nifty B&H (endpoints)", line=dict(color=COLORS["Nifty B&H"], width=2, dash="dot")))
    fig.update_layout(template="plotly_white", height=450, hovermode="x unified",
                      yaxis_title="Equity (Rs.)", xaxis_title="Year",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Per-Year Returns")
    py_melt = PER_YEAR.melt(id_vars="Year", var_name="Variant", value_name="Return %")
    fig2 = px.bar(py_melt, x="Year", y="Return %", color="Variant", barmode="group",
                  color_discrete_map={"FULL": COLORS["FULL"], "NO-DCF": COLORS["NO-DCF"]},
                  template="plotly_white", height=400)
    fig2.add_hline(y=0, line_color="black", line_width=1)
    st.plotly_chart(fig2, use_container_width=True)

    col_dd, col_sh = st.columns(2)
    with col_dd:
        st.subheader("Max Drawdown")
        dd = pd.DataFrame({"Variant": ["FULL", "NO-DCF", "Nifty B&H"], "Max Drawdown (%)": [-26.6, -29.5, -38.4]})
        fig3 = px.bar(dd, x="Variant", y="Max Drawdown (%)", color="Variant",
                      color_discrete_map=COLORS, template="plotly_white", height=350, text="Max Drawdown (%)")
        st.plotly_chart(fig3, use_container_width=True)
    with col_sh:
        st.subheader("Sharpe Ratio")
        sh = pd.DataFrame({"Variant": ["FULL", "NO-DCF", "Nifty B&H"], "Sharpe": [0.76, 0.31, 0.71]})
        fig4 = px.bar(sh, x="Variant", y="Sharpe", color="Variant",
                      color_discrete_map=COLORS, template="plotly_white", height=350, text="Sharpe")
        st.plotly_chart(fig4, use_container_width=True)

# ===== TAB 3: TRADES =====
with tab3:
    col_w, col_l = st.columns(2)
    with col_w:
        st.subheader("Top 5 Wins")
        st.dataframe(TOP_WINS.style.map(color_return, subset=["Return %"]),
                     use_container_width=True, hide_index=True)
    with col_l:
        st.subheader("Top 5 Losses")
        st.dataframe(TOP_LOSSES.style.map(color_return, subset=["Return %"]),
                     use_container_width=True, hide_index=True)

    col_pie, col_ls = st.columns(2)
    with col_pie:
        st.subheader("Exit Distribution")
        fig5 = px.pie(EXIT_DIST, values="Count", names="Exit Type", hole=0.4,
                      color_discrete_sequence=["#2ca02c", "#d62728", "#7f7f7f"],
                      template="plotly_white", height=350)
        st.plotly_chart(fig5, use_container_width=True)
    with col_ls:
        st.subheader("Long vs Short P&L")
        fig6 = px.bar(LONG_SHORT, x="Side", y="PnL (Rs.)", color="Side",
                      color_discrete_map={"Long": "#2ca02c", "Short": "#d62728"},
                      template="plotly_white", height=350, text="PnL (Rs.)")
        st.plotly_chart(fig6, use_container_width=True)

    st.subheader("Trade-Level Explorer")
    if JSON_PATH.exists():
        try:
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            trades = None
            if isinstance(data, dict):
                for key in ["trades", "trade_log", "all_trades", "results"]:
                    if isinstance(data.get(key), list):
                        trades = data[key]; break
                if trades is None:
                    st.info(f"results.json top-level keys: {list(data.keys())}")
            elif isinstance(data, list):
                trades = data
            if trades:
                tdf = pd.DataFrame(trades)
                st.caption(f"Loaded {len(tdf)} trade records")
                st.dataframe(tdf.head(200), use_container_width=True)
            else:
                st.json(data if isinstance(data, dict) else {"items": len(data)})
        except Exception as e:
            st.error(f"Could not parse results.json: {e}")
    else:
        st.info("results.json not found in backtest_results/")

# ===== TAB 4: METHODOLOGY =====
with tab4:
    st.subheader("Methodology")
    st.markdown("""
    - **Engine**: real MLEngine, SmartMoneyEngine, RiskManager, DecisionEngine (100-pt matrix, +/-30 trigger, RSI timing) reused at every scan date.
    - **Universe**: today's Nifty 50 constituents, scanned every 5th trading day (weekly).
    - **ML training cap**: 2000 most recent bars at each scan (compute limit; main.py uses full history).
    - **Entries**: next trading day's open; WAIT-FOR-DIP / BOUNCE become limit orders at halfway trigger, valid 5 trading days.
    - **Exits**: scanner SL / 2.5x TP, or time stop after 20 days. SL assumed first when SL and TP touch same day (pessimistic).
    - **Risk**: 2% of current equity per trade, ATR stop, 5/10/15% caps via RiskManager.
    - **Costs**: 0.10% of trade value per side.
    """)

    st.subheader("Known Biases")
    st.warning("""
    1. **Survivorship bias**: universe is today's Nifty 50 list; delisted names excluded.
    2. **DCF look-ahead**: fundamentals are one current snapshot. FULL includes that bias; NO-DCF zeroes the DCF factor and is the honest core result.
    3. **Weekly cadence** vs main.py's daily design.
    4. **Trailing-stop instructions** printed by main.py are not simulated.

    Treat all figures as an upper bound, not a forecast.
    """)

    with st.expander("Full BACKTEST_REPORT.txt"):
        if REPORT_PATH.exists():
            st.text(REPORT_PATH.read_text(encoding="utf-8", errors="replace"))
        else:
            st.info("Report file not found")