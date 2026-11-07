import streamlit as st
import json
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Hybrid Nifty 50 — Backtest Dashboard", layout="wide")

FOLDER = Path(__file__).parent
RESULTS_DIR = FOLDER / "backtest_results"

st.title("Hybrid Nifty 50 — Backtest Dashboard")
st.caption("10.8-year walk-forward simulation  |  Rs.100,000 starting capital  |  2,665 trades")

# 1. Equity curve
st.header("Equity Curve")
png = RESULTS_DIR / "equity_curve.png"
if png.exists():
    st.image(str(png), use_container_width=True)
else:
    st.warning("equity_curve.png not found in backtest_results/")

# 2. Headline metrics
st.header("Headline Metrics")
metrics = pd.DataFrame({
    "Metric":     ["Total return", "CAGR", "Max drawdown", "Sharpe", "Win rate", "Profit factor", "Avg win / Avg loss", "Trades", "Final equity"],
    "FULL":       ["133.7%", "8.2%",  "-26.6%", "0.76", "33.9%", "1.17", "7.5% / -3.3%",  "2,665", "Rs.233,680"],
    "NO-DCF":     ["32.2%",  "2.6%",  "-29.5%", "0.31", "32.3%", "1.06", "7.2% / -3.4%",  "2,557", "Rs.132,198"],
    "Nifty B&H":  ["194.4%", "10.5%", "-38.4%", "0.71", "-",      "-",    "-",              "-",     "-"],
})
st.dataframe(metrics, use_container_width=True, hide_index=True)

# 3. Per-year returns
st.header("Per-Year Returns")
per_year = pd.DataFrame({
    "Year":   [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026],
    "FULL":   [2.2,  27.2, -1.5, -1.3, 47.3, 16.5, -7.1,  23.1, 10.4, -1.9, -13.0],
    "NO-DCF": [1.1,  31.5, -2.7,  0.6,  4.3,  1.6, -11.6, 15.6, 12.4, -3.6, -13.3],
}).melt(id_vars="Year", var_name="Variant", value_name="Return %")
st.plotly_chart(
    px.bar(per_year, x="Year", y="Return %", color="Variant", barmode="group",
           title="Annual Returns: FULL vs NO-DCF"),
    use_container_width=True
)

# 4. Trade-level data (defensive — adapts to whatever's in the JSON)
st.header("Trade-Level Data")
json_path = RESULTS_DIR / "results.json"
if json_path.exists():
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        trades = None
        if isinstance(data, dict):
            for key in ["trades", "trade_log", "all_trades", "results"]:
                if isinstance(data.get(key), list):
                    trades = data[key]; break
            if trades is None:
                st.write("Top-level keys:", list(data.keys()))
        elif isinstance(data, list):
            trades = data
        if trades:
            df = pd.DataFrame(trades)
            st.dataframe(df.head(100), use_container_width=True)
            st.caption(f"Showing first 100 of {len(df)} records")
    except Exception as e:
        st.error(f"Could not parse results.json: {e}")
else:
    st.info("results.json not found")

# 5. Full report text
with st.expander("Full BACKTEST_REPORT.txt"):
    report = RESULTS_DIR / "BACKTEST_REPORT.txt"
    if report.exists():
        st.text(report.read_text(encoding="utf-8", errors="replace"))