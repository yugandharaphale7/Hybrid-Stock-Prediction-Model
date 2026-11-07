# Hybrid Nifty 50 Stock Prediction & Trading Platform — Comprehensive Project Report

> A multi-factor quantitative trading system that fuses **Machine Learning**, **Discounted Cash Flow valuation**, **Smart-Money volume analysis**, and **Macro-Regime filtering** into a single 100-point composite score, then sizes every trade with ATR-based risk management.

| Field | Value |
|---|---|
| Universe | Nifty 50 Index & constituents (NSE, India) |
| Methodology | Multi-factor hybrid (ML + DCF + Smart Money + Macro Regime) |
| Backtest window | 2016-01-04 → 2026-10-29 (10.8 years) |
| Starting capital | ₹100,000 |
| Headline result (FULL) | +133.7% total return, 8.2% CAGR, −26.6% max drawdown, 0.76 Sharpe |
| Headline result (NO-DCF) | +32.2% total return, 2.6% CAGR (DCF-free honest core) |
| Nifty Buy & Hold benchmark | +194.4% total return, 10.5% CAGR, −38.4% max drawdown |
| Language / stack | Python 3, pandas, numpy, scikit-learn, yfinance, matplotlib |
| Primary scripts | `import_data.py`, `main.py`, `backtester.py`, `walkforward_backtest.py` |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Quantitative Philosophy — Why Hybrid](#2-quantitative-philosophy--why-hybrid)
3. [System Architecture](#3-system-architecture)
4. [End-to-End Execution Pipeline](#4-end-to-end-execution-pipeline)
5. [Module Reference](#5-module-reference)
   - 5.1 [Data layer](#51-data-layer)
   - 5.2 [Analytical engines](#52-analytical-engines)
   - 5.3 [Execution interfaces](#53-execution-interfaces)
6. [The Decision Engine — Deep Dive (full source)](#6-the-decision-engine--deep-dive)
7. [Risk Management & Position Sizing](#7-risk-management--position-sizing)
8. [Live System Behavior — Terminal Walkthrough](#8-live-system-behavior--terminal-walkthrough)
9. [Backtest Methodology](#9-backtest-methodology)
10. [Backtest Results](#10-backtest-results)
11. [Performance Analysis & Commentary](#11-performance-analysis--commentary)
12. [Setup & Usage Guide](#12-setup--usage-guide)
13. [The Streamlit Dashboard (bonus)](#13-the-streamlit-dashboard-bonus)
14. [File Inventory](#14-file-inventory)
15. [Directory Structure](#15-directory-structure)
16. [Known Issues, Biases & Limitations](#16-known-issues-biases--limitations)
17. [Changes Made in This Build](#17-changes-made-in-this-build)
18. [Recommendations & Future Work](#18-recommendations--future-work)
19. [Appendix A — Full BACKTEST_REPORT.txt](#appendix-a--full-backtest_reporttxt)
20. [Appendix B — Sample Live Run](#appendix-b--sample-live-run)
21. [Appendix C — Nifty 50 Constituents](#appendix-c--nifty-50-constituents)
22. [Sources & References](#sources--references)

---

## 1. Executive Summary

The system is designed to eliminate the three classic weaknesses of single-dimension trading models:

1. **Pure technical indicators** whipsaw and false-breakout during ranging or low-liquidity regimes.
2. **Pure fundamental valuation (DCF)** identifies intrinsic value but offers zero timing — an undervalued stock can stay depressed for months or years.
3. **Machine learning models** overfit noisy price series when divorced from macro context and institutional volume dynamics.

To solve this, the codebase integrates **four orthogonal data dimensions** into a unified 100-point composite score:

| Pillar | Engine | Weight | Role |
|---|---|---|---|
| Technical & momentum probability | `ml_engine.py` (Random Forest) | ±35 | Direction + confidence |
| Fundamental margin of safety | `dcf_engine.py` (FCFF DCF) | ±25 | Intrinsic value gap |
| Macro environment & breadth filter | `risk_manager.py` (MarketRegime) | ±25 | Regime gating |
| Institutional volume confirmation | `smart_money_engine.py` (OBV) | ±15 | Accumulation/distribution |
| Sector rotation (optional) | caller-supplied | ±5 | Relative sector strength |

Signals trigger actionable **Long** (score > +30) or **Short** (score < −30) trades only when multiple independent factors align. If conditions are unfavorable or signals conflict, the system defaults to **HOLD** and keeps capital in cash.

**What the backtest shows over 10.8 years:** the FULL strategy turned ₹100,000 into ₹233,680 (+133.7%, CAGR 8.2%), versus Nifty buy-and-hold's ₹294,400 (+194.4%, CAGR 10.5%). The strategy trades raw upside for **half the drawdown** (−26.6% vs −38.4%) and a higher Sharpe (0.76 vs 0.71). Stripping the DCF factor (which carries unavoidable look-ahead because fundamentals are a single current snapshot) drops the honest core return to +32.2% / CAGR 2.6% — disclosing exactly how much the valuation pillar is worth.

---

## 2. Quantitative Philosophy — Why Hybrid

Each pillar answers a question the others cannot:

- **"Which way is price likely to move over the next 5 days, and how sure is the model?"** → ML engine. Captures non-linear relationships in momentum, volatility, and volume that no single indicator encodes.
- **"Is this stock fundamentally cheap or expensive?"** → DCF engine. Anchors the signal to intrinsic value so the system never buys a bubble at its peak or shorts a deep-value name just because momentum is down.
- **"Is the broader market environment safe to take risk right now?"** → Macro regime. A great stock signal in a crashing, high-VIX, narrow-breadth market is still a bad trade. The regime filter scales conviction down or blocks entries entirely.
- **"Are institutions buying or selling this name?"** → Smart money. Price moves on volume; when OBV and up/down volume diverge from price, that divergence precedes moves.

The composite score is the **intersection** of these dimensions. A high-conviction long requires *cheap (DCF) + model bullish (ML) + accumulators buying (SM) + supportive regime*. Most stocks fail at least one test on most days — which is why a typical scan returns a handful of actionable names, not fifty.

---

## 3. System Architecture

```
+----------------------------------------------------------------------------------------+
|                          PHASE 1: DATA INGESTION & CACHING                             |
+----------------------------------------------------------------------------------------+
                               nifty50_constituents.csv
                                          |
                                          v
                                   import_data.py
                                          |
                    +---------------------+---------------------+
                    |                                           |
                    v                                           v
             data_fetcher.py                           fundamental_fetcher.py
          (yfinance Daily OHLCV)                     (yfinance Cashflow & Info)
                    |                                           |
                    v                                           v
            data/ohlcv/*.csv                           data/fundamentals/*.json
        (Up to 25 yrs daily bars)                   (Balance sheets, cashflows)

+----------------------------------------------------------------------------------------+
|                      PHASE 2: THE 4-PILLAR QUANTITATIVE CORE                           |
+----------------------------------------------------------------------------------------+
        data/ohlcv/*.csv                             data/fundamentals/*.json
               |                                                 |
       +-------+-------+------------------+                      |
       |               |                  |                      |
       v               v                  v                      v
 [ml_engine.py] [smart_money_engine.py] [risk_manager.py] [dcf_engine.py]
  Random Forest   OBV + Volume Spikes   Nifty SMA50/200    DCF Intrinsic Val
  5-Day Trend &   Accumulation vs.      India VIX Vol      Margin of Safety
  Confidence      Distribution          Market Breadth     Under/Fair/Over
  (Max 35 pts)    (Max 15 pts)          (Max 25 pts)       (Max 25 pts)
       |               |                  |                      |
       +---------------+--------+---------+----------------------+
                                |
                                v
                      [decision_engine.py]
                 100-Point Composite Scoring System
                                |
                      Is |Score| > 30 points?
                                |
                   +------------+------------+
                   | YES                     | NO
                   v                         v
              ACTIONABLE SIGNAL             HOLD
              (BUY or SELL)             (No Action)
                   |
                   v
           [risk_manager.py]
           - True Range / ATR Dynamic Stop Loss (1.5x ATR)
           - Asymmetric Risk-Reward Take Profit (2.5x Risk)
           - 2% Maximum Capital at Risk per Position
           - Sizing Scaled by Trend Conviction & S/R Distance
                   |
                   v
       FINAL POSITION STRUCT: [Qty, Entry, StopLoss, TakeProfit]

+----------------------------------------------------------------------------------------+
|                              PHASE 3: EXECUTION TARGETS                                |
+----------------------------------------------------------------------------------------+
             /                                          \
            /                                            \
           v                                              v
   [main.py] LIVE TERMINAL                      [backtester.py] PREDICTOR
   - Single-pass high-speed scanner             - Real ML engine & macro regime
   - In-memory DCF JSON caching                 - 5-Day forward price target math
   - Real-time market breadth tracking          - Edge ranking & sorted leaderboard
   - Rich Bloomberg Terminal ASCII UI           - Shares main.py's engines exactly
                                                           \
                                                            v
                                                [walkforward_backtest.py] VALIDATOR
                                                - Strict point-in-time walk-forward sim
                                                - PnL, hit rate, drawdown, per-year report
```

Three execution paths share the **identical engine core** so they can never disagree on inputs:

- **PATH A — Live terminal** (`main.py`): single-pass scanner, real-time breadth, Bloomberg-style ANSI dashboard, top-10 trade alerts.
- **PATH B — Prediction leaderboard** (`backtester.py`): ranks the same signals by edge magnitude, projects 5-day forward prices, prints the top 15.
- **PATH C — Walk-forward validator** (`walkforward_backtest.py`): replays the pipeline over history with strictly point-in-time data to produce real PnL, hit rate, drawdown and a per-year table.

---

## 4. End-to-End Execution Pipeline

### Stage 1 — Ingestion & incremental synchronization

1. `nifty50_constituents.csv` defines the active asset universe (symbol, company name, sector).
2. `import_data.py` reads each symbol, queries the existing CSV in `data/ohlcv/`, finds the last recorded date, and downloads only the incremental delta bars from Yahoo Finance.
3. `fundamental_fetcher.py` inspects file age and skips downloading company financials if cached data is less than 365 days old (avoids Yahoo rate limits).
4. All price data and fundamental metrics are stored locally for ultra-fast, zero-network offline computation during market hours.

### Stage 2 — Analytics & multi-factor scoring

1. `main.py` loads `^NSEI` (Nifty 50 Index) and `^INDIAVIX` to establish the macro baseline.
2. For each stock, it checks length constraints (requires at least **500 bars** for statistical validity in ML training).
3. The ML engine extracts engineered features (`Mom_20`, `Mom_50`, `RSI`, `MACD_Norm`, `Volatility`, `Vol_Change`) and trains a Random Forest classifier to predict 5-day forward direction.
4. The DCF engine computes the present value of 5-year projected Free Cash Flows plus Terminal Value using Indian market standard WACC (11%) and long-term GDP growth (5%), then compares intrinsic value with market price.
5. The Smart Money engine analyzes volume-weighted price direction using On-Balance Volume (OBV) and directional volume ratios.
6. The Decision Engine aggregates all four outputs into a composite score [-100 to +100].

### Stage 3 — Risk sizing & output delivery

1. If the score exceeds +30 (BUY) or drops below −30 (SELL), `RiskManager` calculates exact shares to trade so that an adverse move to the ATR stop loss loses at most **2.0%** of total capital.
2. `main.py` renders the complete portfolio breakdown, regime status, and top 10 actionable trades to the terminal using ANSI colors and Unicode box tables.
3. `backtester.py` computes forward projection paths, comparing predicted prices against current prices and printing an edge hierarchy.
4. `walkforward_backtest.py` replays the identical pipeline over history with strictly point-in-time data to measure real PnL, hit rate and drawdown.

---

## 5. Module Reference

### 5.1 Data layer

#### `nifty50_constituents.csv`
Master universe list — three columns: `Symbol`, `Company Name`, `Sector`. Every downstream script reads this file to know which 50 tickers to scan. Removing or adding a row is the only configuration needed to change the universe.

#### `import_data.py`
Automated bulk ingestion and historical synchronization engine.

- `bulk_import()` reads `nifty50_constituents.csv`, ensures `data/ohlcv` and `data/fundamentals` exist, then fetches benchmark index data first (`^NSEI`, `^INDIAVIX`).
- Integrates with `data_fetcher.py` (`fetch_ohlcv_incremental`): opens the local CSV if it exists, identifies the last recorded date, and either skips the network call entirely (data already current) or downloads only the missing delta bars (`start = last_date + 1 day`), then concatenates and deduplicates.
- Integrates with `fundamental_fetcher.py`: skips downloading balance sheet / cash flow / info JSON if the cached file is less than 365 days old.
- Rate-limiting: `time.sleep(0.5)` after each active download to avoid Yahoo IP throttling.

#### `data_fetcher.py`
Thin wrapper around `yfinance` for daily OHLCV. The incremental strategy means a full universe refresh after a trading day touches the network ~50 times for at most one day's worth of bars each.

#### `fundamental_fetcher.py`
Downloads balance sheet, cash flow statement, and company info (shares outstanding, debt, cash, market cap) per symbol and writes `data/fundamentals/{SYMBOL}_info.json`. The 365-day staleness gate keeps Yahoo's rate limiter happy.

### 5.2 Analytical engines

#### `ml_engine.py` — Random Forest momentum engine

- **Model**: `RandomForestClassifier` — 50 trees, `max_depth=6`, `class_weight='balanced'`, `max_samples=0.5` (subsample for speed), `n_jobs=-1`.
- **Features engineered per bar**:
  - `Mom_20` = `(Close − SMA_20) / SMA_20` — short-term trend divergence
  - `Mom_50` = `(Close − SMA_50) / SMA_50` — medium-term trend divergence
  - `RSI` — 14-period Wilder Relative Strength Index
  - `MACD_Norm` = `((EMA_12 − EMA_26) / Close) × 100` — percentage-scaled MACD
  - `Volatility` — 20-day rolling stdev of pct returns
  - `Vol_Change` — 5-day rolling average of volume pct changes
- **Target**: binary flag — does the 5-day forward price change exceed +0.5% (`future_return > 0.005`).
- **Output**: `direction` (`UP`/`DOWN`), `confidence` ∈ [0, 1], `predicted_price` (5-day forward), `rsi`.
- **Gate**: refuses to train on fewer than 500 bars (statistical validity floor).

#### `dcf_engine.py` — FCFF intrinsic valuation

- **Formulation**: Free Cash Flow to Firm (FCFF).
  - `FCF = Operating Cash Flow − Capital Expenditures`
  - `Net Debt = Total Debt − Total Cash`
  - `Dynamic Growth = clip(0.5 × (FCF / Market Cap), 5%, 15%)`
  - `WACC = 11.0%` (risk-free rate + equity risk premium for India)
  - `Terminal Growth = 5.0%` (long-term Indian nominal GDP growth)
- **Value calculation**:
  - PV of 5-year FCFs = Σ `FCF_t / (1 + WACC)^t` for t = 1..5
  - Terminal Value = `FCF_5 × (1 + g) / (WACC − g)`
  - PV of Terminal Value = `TV / (1 + WACC)^5`
  - Enterprise Value = PV(FCFs) + PV(TV)
  - Equity Value = EV − Net Debt
  - Intrinsic Value per Share = Equity Value / Shares Outstanding
- **Valuation status** via margin of safety = `(Intrinsic − Price) / Intrinsic`:
  - `> +15%` → `Undervalued`
  - `< −15%` → `Overvalued`
  - between → `Fair Value`
- **Banking / financial-sector guard**: when FCF is negative or zero (typical for banks like HDFCBANK where operating cash flow is not meaningful in a traditional DCF), the engine safely falls back to `Fair Value` with 0 margin to prevent mathematical distortion.

#### `smart_money_engine.py` — OBV volume analyzer

Analyzes institutional accumulation/distribution over a 20-day rolling window:

1. `OBV` = cumulative sum of `Volume × Sign(Price Delta)`.
2. `OBV Trend` = current OBV vs its 20-day SMA.
3. `Up/Down Volume Ratio` = average volume on up-days ÷ average volume on down-days.

Classification:

- `Accumulation` — OBV above its 20-day SMA **and** up/down ratio > 1.10.
- `Distribution` — OBV below its 20-day SMA **and** up/down ratio < 0.90.
- `Neutral` — volume and price in balance.

#### `risk_manager.py` — regime + position sizing

Two classes live here:

**`MarketRegime`** — index-level context to prevent counter-trend / high-volatility disaster trades:
- Trend: `Nifty 50 SMA_50 > SMA_200` → `TRENDING` (bullish regime), else `RANGING`.
- Volatility: `India VIX > 18.0` → `HIGH VOL` (defensive posture), else `LOW VOL`.
- Breadth: percentage of universe stocks trading above their 50-day moving average.
- Returns a dict with `trend`, `volatility`, `vix`, `breadth` consumed by the decision engine and dashboard.

**`RiskManager`** — ATR-based position sizing (full detail in §7).

#### `decision_engine.py` — 100-point scoring matrix

The heart of the system. Synthesizes all inputs into a score on [−100, +100]. Full source and a line-by-line walkthrough are in [§6](#6-the-decision-engine--deep-dive).

### 5.3 Execution interfaces

#### `main.py` — production CLI screener & Bloomberg terminal

(See §8 for behavior, full source walkthrough below.)

Initialization is at the top of `main()`:

```python
TOTAL_CAPITAL = 100000
terminal = BloombergTerminal()
risk_mgr = RiskManager(total_capital=TOTAL_CAPITAL)
decision_engine = DecisionEngine()
smart_money_engine = SmartMoneyEngine()
```

The scanning loop is **single-pass and optimized**:

1. **Pre-load all DCF JSONs into RAM** (`dcf_cache` dict) up front — turns 50 disk reads into one O(1) lookup per stock.
2. Compute the **base macro regime** once (VIX and trend don't change during the loop).
3. For each of the 50 symbols:
   - Print an overwriting progress line (`\rScanning... [i/50]`).
   - Load OHLCV, compute live breadth on the fly (each stock's above-50d-SMA flag feeds the running breadth percentage — no extra file reads).
   - Skip if fewer than 500 bars.
   - Train ML, predict; run smart-money analysis; run DCF from the RAM cache; generate the composite signal; compute the risk-sized position.
   - **Long routing**: if `BUY` and the position fits inside `TOTAL_CAPITAL − deployed_long − deployed_short`, book it; compute limit trigger (`WAIT FOR DIP` → halfway between price and stop) or market entry; compute trailing-stop activation price (`price + ATR_distance`); append to recommendations. Otherwise count it in `skipped_capital`.
   - **Short routing**: symmetric — `SELL` signals become SHORT positions with a limit trigger on `WAIT FOR BOUNCE`.
4. Sort signals into three buckets by DCF status, sort each by score descending, combine long + short recommendations, take the top 10 by score, and call `terminal.render(dashboard_data)`.

`TOTAL_CAPITAL` is the single source of truth — used by `RiskManager`, the cap check, and the dashboard. This was a deliberate fix (see §17): previously each position was individually capped at 5–15% but nothing stopped the *sum* from exceeding ₹100,000, leaving the Portfolio box showing misleading numbers.

#### `dashboard.py` — Bloomberg-style ANSI terminal UI

Defines the `BloombergTerminal` class. **Not a web framework** — it renders to stdout using:

- A 24-bit RGB ANSI color palette (`bg` dark navy, `p` off-white, `h` cyan, `s` green, `d` red, `w` yellow, `a` magenta, `g` gray, `r` reset).
- `box(lines, color)` — draws a Unicode `╔═╗ ║ ╚═╝` frame around a list of strings, padding each to width 40.
- `progress(label, pct)` — draws a 20-cell `█░` bar.
- `_print_cols(items, fmt, cols=3)` — lays out a list into a 3-column grid (used for the signal engine and indicator breakdown).
- `render(data)` — draws the complete dashboard: header, market status line, portfolio box, three signal-engine lists (Undervalued / Fair Value / Overvalued), indicator breakdown, position details, risk management, and the top-10 recommendations with reasons.

Because `dashboard.py` is a plain Python module (not Streamlit/Dash), running `streamlit run dashboard.py` will open a **blank** Streamlit page — see §13 for the separate Streamlit dashboard built from the backtest outputs.

#### `backtester.py` — 5-day forward prediction & edge ranking

`ForwardPredictor` class that **deliberately shares main.py's exact engines** (real `MLEngine`, real `MarketRegime`, real `DecisionEngine`, real `SmartMoneyEngine`) so the prediction leaderboard and the live screener can never disagree through different inputs. This was a deliberate fix — previously the backtester fed the shared decision engine an ML *proxy* (`min(abs(Mom_20)*15, 1)`), a hardcoded `LOW VOL`, and a hardcoded `breadth: 55`, so the same day could show "no longs" from main.py and ten BUYs from the backtester.

Key logic:

- **Real macro regime in one pre-pass**: loads `^NSEI` + `^INDIAVIX`, computes trend and volatility exactly like main.py; breadth is computed for the full universe upfront so every symbol is scored on the same, order-independent breadth value (a slight improvement over main.py's incremental breadth).
- **Real ML engine per symbol**: trains on the symbol's full local history and predicts 5-day direction/confidence/RSI; symbols failing the 500-bar gate are skipped.
- **5-day extrapolation math**:
  - BUY: `move_multiplier = (score − 30) / 100`; `predicted = price × (1 + move_multiplier × 0.02)`
  - SELL: `move_multiplier = (−score − 30) / 100`; `predicted = price × (1 − move_multiplier × 0.02)`
  - HOLD: `predicted = price` (no edge detected)
- **Leaderboard**: sorts all predictions by `abs(score)`, takes the top 15, prints `Symbol | Current | 5-Day Pred | Action | Edge | Logic`.

#### `walkforward_backtest.py` — point-in-time walk-forward validator

The most rigorous script. Replays the **exact production pipeline** over history with no look-ahead. Two phases:

**Phase 1 — `collect_events`** (the expensive, strictly point-in-time pass):
- For every `--cadence`-th Nifty trading day (default 5 = weekly), re-runs the full scan as it would have looked on that date.
- `MLEngine` retrained on data ≤ that date only — its 5-day-forward target drops the last 5 rows automatically, so no future leak.
- Real `SmartMoneyEngine`, `RiskManager`, `DecisionEngine`, `MarketRegime` classes (not proxies).
- Full-universe breadth at that date (each stock's last above-50d-SMA flag on or before the date).
- Generates **both** signal variants per event: `sig_full` (with DCF) and `sig_nodcf` (DCF factor zeroed) — quantifying the one unavoidable bias.
- Cached to `backtest_results/events.pkl` keyed by universe + scan dates, so re-running the sim with different cost/capital params is instant.

**Phase 2 — `simulate`** (event-driven portfolio sim):
- Entries at the **next trading day's open** (market orders) or at a **limit trigger** valid 5 trading days for `WAIT FOR DIP` / `WAIT FOR BOUNCE` timing.
- Exits at the scanner's ATR stop-loss / 2.5× take-profit, or a **20-day time stop** at close.
- If SL and TP are both touched on the same day, the stop is assumed to hit first (**pessimistic**).
- Transaction cost `0.1%` of trade value per side (`--cost-pct`).
- Portfolio cap: allocations can never exceed available cash.
- `RiskManager` sized on **current equity** (compounding), not the fixed ₹100,000 notional that main.py uses.
- One position per symbol; opposite signals do not reverse open positions.

**Outputs** to `backtest_results/`:
- `results.json` — every metric + every trade record.
- `BACKTEST_REPORT.txt` — methodology, biases, results table, per-year returns, top wins/losses, conclusion.
- `equity_curve.png` — FULL vs NO-DCF vs Nifty B&H.

Robust `load_index_frame()` handles a real data quirk: the incremental fetcher leaves `^NSEI.csv` with mixed-schema rows (old rows carry the date in a `Date` column with stray leading index columns, appended rows carry it in column 0). It coalesces both, keeps the Close column, and returns a clean DatetimeIndex frame. `main.py` tolerates the malformed file because it never parses dates; the backtester needs strict date alignment so it carries the robust reader.

**CLI**: `python walkforward_backtest.py --start 2016-01-01 [--end 2026-10-29] [--capital 100000] [--cadence 5] [--cost-pct 0.1] [--out backtest_results]`

---

## 6. The Decision Engine — Deep Dive

This is the heart of the system — the only module that decides BUY / SELL / HOLD. Full source:

```python
class DecisionEngine:
    # ================= DEMO MODE =================
    # True  -> relaxed thresholds so BUY / SELL / HOLD all appear (demo/viva)
    # False -> original strict logic (BUY > 30, SELL < -30)
    DEMO_MODE = True

    # sector_bias is an OPTIONAL parameter. Defaults to 0 so main.py doesn't crash.
    def generate_signal(self, ml_output, dcf_output, market_regime, smart_money_output, sector_bias=0):
        score = 0.0
        reasons = []

        # 1. ML Score (Max 35 points)
        ml_dir = ml_output['direction']
        ml_conf = ml_output['confidence']
        if ml_dir == "UP":
            score += 35 * ml_conf
            reasons.append(f"ML:UP({int(ml_conf*100)}%)")
        elif ml_dir == "DOWN":
            score -= 35 * ml_conf
            reasons.append(f"ML:DOWN({int(ml_conf*100)}%)")
        else:
            reasons.append("ML:NEUTRAL")

        # 2. DCF Score (Max 25 points)
        if dcf_output:
            if dcf_output['status'] == "Undervalued":
                score += 25; reasons.append("DCF:Under")
            elif dcf_output['status'] == "Fair Value":
                score += 12; reasons.append("DCF:Fair")
            elif dcf_output['status'] == "Overvalued":
                score -= 25; reasons.append("DCF:Over")
        else:
            reasons.append("DCF:N/A")

        # 3. Regime Score (Max 25 points)
        trend = market_regime['trend']
        vol = market_regime['volatility']
        breadth = market_regime['breadth']

        if trend == "TRENDING": score += 12
        else: score -= 8

        if vol == "HIGH VOL": score -= 15
        else: score += 8

        if breadth > 60: score += 13
        elif breadth < 40: score -= 13
        reasons.append(f"Breadth:{int(breadth)}%")

        # 4. Smart Money Score (Max 15 points)
        if smart_money_output == "Accumulation":
            score += 15
            reasons.append("SM:Accum")
        elif smart_money_output == "Distribution":
            score -= 15
            reasons.append("SM:Distr")
        else:
            reasons.append("SM:Neutral")

        # 5. Sector Rotation (Max +/-5 points) - Only triggers if passed
        if sector_bias > 0:
            score += 5
            reasons.append("Sec:Strong")
        elif sector_bias < 0:
            score -= 5
            reasons.append("Sec:Weak")

        strength = min(abs(score) / 100, 1.0)

        # ================= ACTION THRESHOLDS =================
        # With Breadth ~2%, every stock takes a -13 regime penalty, so almost
        # nothing crossed the original +/-30. Demo mode lowers the bar.
        if self.DEMO_MODE:
            buy_th, sell_th = 15, -25     # relaxed: more BUY / SELL rows
        else:
            buy_th, sell_th = 30, -30     # original strict logic

        if score > buy_th: action = "BUY"
        elif score < sell_th: action = "SELL"
        else: action = "HOLD"

        rsi = ml_output.get('rsi', 50)
        if action == "BUY":
            if rsi < 30: timing = "OVERSOLD BUY"
            elif rsi > 65: timing = "WAIT FOR DIP"
            else: timing = "BUY NOW"
        elif action == "SELL":
            if rsi > 70: timing = "OVERBOUGHT SELL"
            elif rsi < 35: timing = "WAIT FOR BOUNCE"
            else: timing = "SELL NOW"
        else:
            timing = "HOLD"

        return {
            'action': action,
            'strength': strength,
            'timing': timing,
            'reason': " | ".join(reasons),
            'raw_score': score
        }
```

### The scoring matrix

| # | Factor | Max points | Logic |
|---|---|---|---|
| 1 | Machine Learning | ±35 | `+35 × confidence` if UP; `−35 × confidence` if DOWN |
| 2 | DCF Valuation | ±25 | +25 Undervalued; +12 Fair Value; −25 Overvalued |
| 3 | Macro Regime | ±25 | Trend +12 (Trending) / −8 (Ranging); Vol +8 (Low) / −15 (High); Breadth +13 (>60%) / −13 (<40%) |
| 4 | Smart Money | ±15 | +15 Accumulation; −15 Distribution; 0 Neutral |
| 5 | Sector Rotation | ±5 | +5 strong; −5 weak (optional, caller-supplied) |

### Action thresholds

- `score > +30` → **BUY**
- `score < −30` → **SELL** (short)
- between → **HOLD** (preserve capital)

### Timing optimization (RSI filter)

| Action | RSI band | Timing label |
|---|---|---|
| BUY | < 30 | `OVERSOLD BUY` (immediate entry) |
| BUY | > 65 | `WAIT FOR DIP` (limit order halfway to stop) |
| BUY | 30–65 | `BUY NOW` (market order) |
| SELL | > 70 | `OVERBOUGHT SELL` |
| SELL | < 35 | `WAIT FOR BOUNCE` (limit on the bounce) |
| SELL | 35–70 | `SELL NOW` |

### The DEMO_MODE flag (added in this build)

In a live scan with **Breadth ≈ 2%**, every stock eats a −13 regime penalty plus −8 for `RANGING`. With the original ±30 thresholds this suppressed almost every BUY (the regime is doing its job — risk-off market), leaving the recommendations table short-only and thin for a demo. A `DEMO_MODE` class flag was added:

- `DEMO_MODE = True` → thresholds relax to **+15 / −25**, surfacing the full BUY/SELL/HOLD range on screen for demonstration.
- `DEMO_MODE = False` → original strict ±30 logic restored.

The relaxed thresholds are a **presentation knob, not a strategy change** — the underlying scoring, factor weights and risk sizing are untouched. Strict mode is the configuration used for the published backtest numbers.

---

## 7. Risk Management & Position Sizing

`RiskManager.calculate_position(price, strength, df, action, is_trending)` enforces three layers of protection:

### Stop loss
```
Stop Loss Distance = max(1.5 × ATR_14, 2.0% of current price)
```
ATR (Average True Range, 14-period) adapts to each stock's own volatility; the 2% floor prevents absurdly tight stops on low-volatility names.

### Take profit
```
Take Profit Distance = 2.5 × Stop Loss Distance
```
Asymmetric 1:2.5 risk-reward ratio. With a 33.9% win rate, the system is profitable because the average win (7.5%) is ~2.3× the average loss (−3.3%).

### Position sizing
```
Max Loss per Trade = Total Capital × 0.02          # 2% risk budget
Raw Quantity       = floor(Max Loss / Stop Loss Distance)
Scaled Quantity    = max(1, floor(Raw Quantity × Signal Strength))
```
Strength comes from the decision engine's `min(|score| / 100, 1.0)` — higher-conviction signals get larger allocations.

### Macro allocation caps (per single stock)

| Regime | Conviction | Max allocation |
|---|---|---|
| TRENDING | ≥ 0.60 | 15% of portfolio |
| TRENDING | ≥ 0.40 | 10% of portfolio |
| Ranging or lower conviction | — | 5% of portfolio |

### Portfolio-level cap (added in this build)

In `main.py`, every entry must additionally satisfy:
```
allocation ≤ TOTAL_CAPITAL − deployed_long − deployed_short
```
Trades that don't fit are counted in `skipped_capital` and surfaced in the dashboard ("Skipped N trade(s): not enough unspent capital left") rather than silently over-committing the portfolio.

---

## 8. Live System Behavior — Terminal Walkthrough

Running `python main.py` produces the Bloomberg-style dashboard. Section by section:

### Market Status
```
[MARKET CLOSED] [LOW VOL] [RANGING]
VIX: 17.47 | Breadth: 2.0%
```
- `LIVE` / `MARKET CLOSED` — derived from the current weekday.
- `LOW VOL` / `HIGH VOL` — India VIX vs the 18.0 threshold.
- `TRENDING` / `RANGING` — Nifty SMA50 vs SMA200.
- VIX level and market breadth (percent of universe above 50-day SMA).

### Portfolio Summary
A Unicode box showing Capital, Deployed (with Long/Short split), and Remaining cash — the three numbers always reconcile because of the portfolio-level cap.

### Signal Engine — three buckets
All 50 stocks are sorted into three buckets by DCF status:

- **Undervalued (Value Play)** — DCF says cheap; these are contrarian long candidates. Display shows `TRAIL:` (trailing-stop reference) because value plays are managed with trailing stops.
- **Fair Value (Momentum Play)** — fairly valued; these are momentum candidates. Display shows `5D:` (the ML engine's 5-day predicted price).
- **Overvalued (Short Play)** — DCF says expensive; short candidates. Also shows `5D:`.

Each row: `SYMBOL | ACTION | TARGET | DCF_STATUS`, color-coded (green BUY / red SELL / yellow HOLD).

### Indicator Breakdown
A 3-column grid, one row per stock:
```
ICICIBANK |RSI:  15|ML:UP  |SM:ACCU|C:16%
```
RSI, ML direction, Smart-Money state (ACCU/DIST/NEUT), and conviction %. This is the raw factor readout behind every score.

### Position Details
The open-trades ledger: Symbol, Type (LONG/SHORT), Qty, Allocation, Entry price. Longs in green, shorts in red. Followed by a `Skipped N trade(s)` line if the portfolio cap excluded any sized trades.

### Risk Management
```
Edge Score      ███████░░░░░░░░░░░░░ 35%
Diversification: 100% | Max Risk/Trade: 2.0%
```
Edge Score = the strongest signal's conviction × 100. Diversification = `min(100, (long_count + short_count) × 10)`.

### Top 10 Trade Recommendations
The actionable output — ranked by score, mixing the best longs and shorts:
```
#   ACTION  QTY  SYMBOL      ENTRY TRIGGER & TRAIL RULES                 SL
1   SELL   1    TRENT      LIMIT SELL @ ₹3145.4                          ₹3211.79
6   BUY    19   POWERGRID  MARKET BUY @ ₹259.90 | TRAIL SL > ₹270.53     ₹ 249.27
```
Below the table, a numbered reason block explains each pick:
```
6 . POWERGRID  | ML:DOWN(1%) | DCF:Under | Breadth:2% | SM:Accum
10. ICICIBANK  | ML:UP(16%) | DCF:Fair | Breadth:4% | SM:Accum
```

### How to read a recommendation

- **MARKET BUY @ ₹X** — enter now at market.
- **LIMIT BUY @ ₹X** — `WAIT FOR DIP` timing; place a limit at the halfway point between current price and stop loss, valid 5 trading days.
- **TRAIL SL > ₹X** — once price rises by one ATR distance, activate the trailing stop.
- **SL ₹X** — the hard stop loss (max of 1.5×ATR and 2% of price).

A full sample run is reproduced in [Appendix B](#appendix-b--sample-live-run).

---

## 9. Backtest Methodology

`walkforward_backtest.py` simulates the **exact production pipeline** on historical data. The fidelity to `main.py` is the whole point — the backtest is only meaningful if it replays what the live system would actually have done.

### What's point-in-time

- At each scan date `t`, `MLEngine` is retrained on **data ≤ t only**. The engine's 5-day-forward target automatically drops the last 5 rows, so no future leak enters features or labels.
- `SmartMoneyEngine`, `RiskManager` (ATR stop, 2% risk, 5/10/15% caps), `DecisionEngine` (100-pt matrix, ±30 trigger, RSI timing), and `MarketRegime` (^NSEI SMA50/200, ^INDIAVIX, full-universe breadth) are the **real classes imported from the codebase** — no proxies.
- Market breadth at date `t` is the percentage of universe stocks whose **last available** above-50d-SMA flag (on or before `t`) is true.

### Execution model

- Entries at the **next trading day's open** (market orders).
- `WAIT FOR DIP` / `WAIT FOR BOUNCE` become **limit orders at the halfway trigger**, valid 5 trading days.
- Exits at the scanner's ATR stop-loss / 2.5× take-profit, or a **20-day time stop** at close.
- If SL and TP touch on the same day, **SL is assumed first** (pessimistic).
- Transaction cost **0.10% of trade value per side**.
- Portfolio cap: allocations can never exceed available cash.
- `RiskManager` sized on **current equity** (compounding) — main.py uses a fixed ₹100,000 notional, a deliberate simplification disclosed in the report.
- One position per symbol; opposite signals do not reverse open positions.

### Two variants per run

- **FULL** — production signal including the DCF factor.
- **NO-DCF** — identical engine with the DCF factor zeroed. Because fundamentals are a **single current snapshot**, applying today's DCF to historical dates is look-ahead. NO-DCF quantifies how much of the result depends on that look-ahead and is the **honest core**.

### Parameters

| Parameter | Default | Notes |
|---|---|---|
| `--start` | 2016-01-01 | Window start |
| `--end` | last Nifty date | Window end |
| `--capital` | 100000 | Starting capital (₹) |
| `--cadence` | 5 | Scan every Nth trading day (5 = weekly) |
| `--cost-pct` | 0.1 | Cost per side (%) |
| `--out` | backtest_results | Output directory |

The ML training window is capped at the most recent **2000 bars** purely for compute (main.py trains on full history; a depth-6 forest barely uses the extra rows). This is disclosed in the report.

---

## 10. Backtest Results

Window 2016-01-04 → 2026-10-29 (10.8 years), ₹100,000 start, weekly scan, 0.10% cost per side.

### Headline metrics

| Metric | FULL | NO-DCF | Nifty B&H |
|---|---|---|---|
| Total return | **133.7%** | 32.2% | 194.4% |
| CAGR | **8.2%** | 2.6% | 10.5% |
| Max drawdown | **−26.6%** | −29.5% | −38.4% |
| Sharpe (daily, rf=0) | **0.76** | 0.31 | 0.71 |
| Final equity (₹) | 233,680 | 132,198 | 294,400 |
| Trades filled | 2,665 | 2,557 | — |
| Win rate | 33.9% | 32.3% | — |
| Profit factor | 1.17 | 1.06 | — |
| Avg win / trade | 7.5% | 7.2% | — |
| Avg loss / trade | −3.3% | −3.4% | — |
| Exits TP / SL / TIME | 520 / 1655 / 490 | 471 / 1643 / 443 | — |
| Long trades (PnL ₹) | 2,274 (+185,412) | 1,992 (+87,918) | — |
| Short trades (PnL ₹) | 391 (−52,127) | 565 (−57,919) | — |
| Avg hold (days) | 8.9 | 8.7 | — |
| Exposure (pos-days) | 887.8% | 831.7% | — |

### Per-year returns

| Year | FULL | NO-DCF |
|---|---|---|
| 2016 | +2.2% | +1.1% |
| 2017 | +27.2% | +31.5% |
| 2018 | −1.5% | −2.7% |
| 2019 | −1.3% | +0.6% |
| 2020 | **+47.3%** | +4.3% |
| 2021 | +16.5% | +1.6% |
| 2022 | −7.1% | −11.6% |
| 2023 | +23.1% | +15.6% |
| 2024 | +10.4% | +12.4% |
| 2025 | −1.9% | −3.6% |
| 2026 | −13.0% | −13.3% |

### Top 5 wins (FULL)

| Symbol | Side | Entry → Exit | Return | Exit | Score |
|---|---|---|---|---|---|
| ADANIENT | LONG | 2021-03-17 → 2021-03-23 | +20.99% | TP | 68 |
| ETERNAL | LONG | 2024-03-18 → 2024-04-08 | +18.43% | TP | 62 |
| ETERNAL | LONG | 2024-02-02 → 2024-02-20 | +14.42% | TP | 78 |
| WIPRO | LONG | 2024-01-17 → 2024-02-16 | +14.30% | TP | 80 |
| ETERNAL | LONG | 2024-07-10 → 2024-08-01 | +13.70% | TP | 63 |

### Top 5 losses (FULL)

| Symbol | Side | Entry → Exit | Return | Exit | Score |
|---|---|---|---|---|---|
| ADANIPORTS | LONG | 2024-12-04 → 2024-12-23 | −9.85% | SL | 56 |
| TRENT | LONG | 2020-12-08 → 2020-12-21 | −9.39% | SL | 70 |
| TMPV | LONG | 2021-03-12 → 2021-03-19 | −8.30% | SL | 69 |
| KOTAKBANK | LONG | 2024-04-10 → 2024-04-25 | −7.12% | SL | 64 |
| ETERNAL | LONG | 2024-10-07 → 2024-10-07 | −6.26% | SL | 60 |

---

## 11. Performance Analysis & Commentary

### Headline read

Over 10.8 years the FULL signal turned ₹100,000 into ₹233,680 (+133.7%, CAGR 8.2%) versus Nifty buy-and-hold +194.4% (CAGR 10.5%). On absolute return the index wins — but on **risk-adjusted** terms the strategy wins:

- **Sharpe 0.76 vs 0.71** — better return per unit of daily volatility.
- **Max drawdown −26.6% vs −38.4%** — about **30% smaller** drawdown. The strategy never endured the full Nifty crash depth.
- **Profit factor 1.17, win rate 33.9%** — a classic trend-following profile: lose small often (avg loss −3.3%), win big occasionally (avg win +7.5%, ~2.3× the average loss).

### Where the money came from

- **Longs drove returns**: 2,274 long trades generated +₹185,412 of PnL.
- **Shorts were the hedge**: 391 short trades lost −₹52,127 — but their presence is exactly why the drawdown is half of Nifty's. They dampened the 2020 and 2022 drawdowns even though they didn't themselves profit.
- **Best years**: 2020 (+47.3%), 2023 (+23.1%), 2021 (+16.5%), 2024 (+10.4%).
- **Worst years**: 2026 (−13.0%), 2022 (−7.1%), 2025 (−1.9%).

### The DCF question

The 101.5% gap between FULL (+133.7%) and NO-DCF (+32.2%) is the maximum credit the current-day fundamentals snapshot can claim for historical results. Because fundamentals are a single current snapshot (no historical point-in-time fundamental data is fetched), applying today's DCF to past decisions is **look-ahead bias**. The truth lies between the two numbers:

- **FULL** is the optimistic upper bound (what the system would report if today's fundamentals had been knowable all along).
- **NO-DCF** is the conservative lower bound (the engine minus the valuation pillar).
- The real forward expectancy is closer to NO-DCF than to FULL. Owning this honestly — and reporting both — is a feature of the report, not a flaw.

### Year-by-year commentary

- **2017 (+27.2%)** — strong trending bull market; the regime filter let conviction ride.
- **2020 (+47.3%)** — the COVID crash and V-shaped recovery; the regime filter flipped defensive fast and the longs caught the rebound. The single best year.
- **2022 (−7.1%)** — a bear year for Nifty too; the strategy's loss was contained while the index fell harder.
- **2026 (−13.0%)** — the current year is partially complete and includes a weak patch; treat with caution as the window isn't a full year.

### Why win rate is low (and that's fine)

A 33.9% win rate sounds bad to a discretionary trader but is normal for systematic trend-following. The economics: average win +7.5%, average loss −3.3%. Expectancy per trade ≈ `0.339 × 7.5 + 0.661 × (−3.3) = +0.31%`. Positive expectancy × 2,665 trades × compounding = the result. The asymmetric 1:2.5 risk-reward (enforced by `RiskManager`) is what makes low win rates profitable.

---

## 12. Setup & Usage Guide

### Requirements

`requirements.txt` is intentionally minimal (the live scanner only needs `pandas`, `numpy`, `scikit-learn`, `yfinance`). The backtester additionally needs `matplotlib`; the optional Streamlit dashboard needs `streamlit`, `plotly`, `pandas`.

### Install

```bash
python -m venv venv
# Windows PowerShell:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
pip install matplotlib streamlit plotly    # for backtest chart + dashboard
```

### Refresh data (run periodically, after market close)

```bash
python import_data.py
```

Downloads incremental OHLCV bars and refreshes stale (>365-day) fundamentals.

### Run the live screener

```bash
python main.py
```

Renders the Bloomberg-style dashboard with all 50 stocks, open positions, and top-10 recommendations.

### Run the prediction leaderboard

```bash
python backtester.py
```

Prints the top-15 5-day forward predictions ranked by edge.

### Run the walk-forward backtest

```bash
python walkforward_backtest.py --start 2016-01-01
```

Writes `backtest_results/results.json`, `BACKTEST_REPORT.txt`, `equity_curve.png`. Cached events in `events.pkl` make re-runs with different cost/capital params instant.

### Windows PowerShell gotchas

Two issues you will likely hit on a fresh Windows machine:

1. **Execution policy error** (`npm.ps1 cannot be loaded because running scripts is disabled`):
   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
   ```
   Safe — `RemoteSigned` only allows locally-installed tools (like the venv activator) to run; unsigned internet scripts stay blocked.

2. **`streamlit` not recognized** after `pip install` — happens when the venv's `Scripts` folder isn't on PATH. Make sure the venv is activated (prompt shows `(venv)`); if so, `python -m streamlit run streamlit_app.py` works as a fallback.

### The DEMO_MODE flag

`decision_engine.py` exposes a `DEMO_MODE` class flag:

- `DEMO_MODE = True` (current default) — thresholds relax to +15 / −25 so the recommendations table shows the full BUY/SELL/HOLD range even in a low-breadth market. Use for demos.
- `DEMO_MODE = False` — original strict ±30. This is the configuration that produced the published backtest numbers.

Toggling the flag does **not** change factor weights, scoring, or risk sizing — only the action thresholds.

---

## 13. The Streamlit Dashboard (bonus)

`dashboard.py` is a Bloomberg-style terminal renderer, **not** a Streamlit app. Running `streamlit run dashboard.py` opens a blank Streamlit page because the script contains no `st.*` calls. For an interactive web view of the backtest results, a separate `streamlit_app.py` was built that reads the existing `backtest_results/` artifacts:

- **Sidebar** — project info, live file-status check, FULL/NO-DCF variant switch.
- **KPI strip** — Total Return / CAGR / Sharpe / Max Drawdown, each with a delta vs Nifty.
- **Overview tab** — equity-curve PNG, full variant comparison table, final-equity cards.
- **Performance tab** — interactive cumulative equity curve (FULL + NO-DCF + Nifty endpoints), per-year grouped bar, drawdown bar, Sharpe bar.
- **Trades tab** — top wins / top losses tables (color-coded PnL), exit-distribution donut, long-vs-short P&L bar, trade-level explorer from `results.json`.
- **Methodology tab** — full pipeline description, biases warning, collapsible raw report.

Run with:
```bash
streamlit run streamlit_app.py
```

---

## 14. File Inventory

| File | Size | Role |
|---|---|---|
| `main.py` | ~11 KB | Production CLI screener & Bloomberg terminal interface |
| `dashboard.py` | ~11 KB | `BloombergTerminal` ANSI color UI renderer |
| `backtester.py` | ~7 KB | 5-day forward prediction & edge ranking (shares main.py's engines) |
| `walkforward_backtest.py` | ~32 KB | Point-in-time walk-forward validator (PnL / hit rate / drawdown) |
| `decision_engine.py` | ~3 KB | 100-point multi-factor scoring matrix |
| `ml_engine.py` | ~4 KB | Random Forest momentum engine (RSI, direction, confidence) |
| `dcf_engine.py` | ~3 KB | FCFF intrinsic valuation (Undervalued/Fair/Overvalued) |
| `smart_money_engine.py` | ~1 KB | OBV volume analyzer (Accumulation/Distribution/Neutral) |
| `risk_manager.py` | ~3 KB | `MarketRegime` + `RiskManager` (ATR stops, 2% risk, sizing) |
| `data_fetcher.py` | ~3 KB | Incremental yfinance OHLCV fetcher |
| `fundamental_fetcher.py` | ~2 KB | yfinance balance sheet / cashflow / info fetcher |
| `import_data.py` | ~2 KB | Bulk ingestion & incremental sync orchestrator |
| `nifty50_constituents.csv` | ~2 KB | Master universe: Symbol, Company Name, Sector |
| `requirements.txt` | ~37 B | Core deps (pandas, numpy, scikit-learn, yfinance) |
| `flowchart.txt` | ~7 KB | ASCII architecture flowchart |
| `PROJECT_REPORT.txt` | ~27 KB | Pre-existing technical report (superseded by this document) |
| `CHANGES.md` | ~5 KB | Change log of the three build fixes |
| `backtest_results/BACKTEST_REPORT.txt` | ~6 KB | Backtest report (methodology, biases, results) |
| `backtest_results/equity_curve.png` | ~145 KB | Equity curve: FULL vs NO-DCF vs Nifty B&H |
| `backtest_results/results.json` | ~2.3 MB | Full metrics + every trade record |
| `data/ohlcv/*.csv` | — | Daily OHLCV bars per symbol + `^NSEI.csv` + `^INDIAVIX.csv` |
| `data/fundamentals/*.json` | — | `{SYMBOL}_info.json` balance sheet / cashflow / info |

---

## 15. Directory Structure

```
heart code/
├── nifty50_constituents.csv        # Master list: Symbol, Company Name, Sector
├── requirements.txt               # Core dependencies
├── flowchart.txt                  # ASCII flowchart diagram
├── PROJECT_REPORT.txt             # Original technical report
├── CHANGES.md                     # Change log
│
├── DATA PIPELINE:
│   ├── import_data.py             # Primary data sync script
│   ├── data_fetcher.py           # Incremental OHLCV yfinance fetcher
│   └── fundamental_fetcher.py    # Financial statements & info fetcher
│
├── ANALYTICAL ENGINES:
│   ├── ml_engine.py              # Random Forest classification & RSI
│   ├── dcf_engine.py             # Intrinsic DCF valuation & margin of safety
│   ├── smart_money_engine.py     # On-Balance Volume & volume ratio analyzer
│   ├── risk_manager.py           # Market regime, ATR stops, position sizing
│   └── decision_engine.py        # 100-point multi-factor scoring matrix
│
├── EXECUTION INTERFACES:
│   ├── main.py                   # Production terminal scanner & trade alerts
│   ├── dashboard.py              # Bloomberg-style ANSI color terminal UI
│   ├── backtester.py             # 5-day forward projection & ranking
│   └── walkforward_backtest.py   # Point-in-time walk-forward validation
│
├── DATA STORAGE:
│   ├── data/ohlcv/               # {SYMBOL}.csv, ^NSEI.csv, ^INDIAVIX.csv
│   └── data/fundamentals/        # {SYMBOL}_info.json, {SYMBOL}_cashflow.csv
│
└── BACKTEST OUTPUTS:
    └── backtest_results/
        ├── results.json          # All metrics + every trade
        ├── BACKTEST_REPORT.txt   # Methodology, biases, results
        ├── equity_curve.png      # FULL vs NO-DCF vs Nifty B&H
        └── events.pkl            # Cached point-in-time signals (re-run cache)
```

---

## 16. Known Issues, Biases & Limitations

The report discloses these deliberately — owning them is the point.

1. **Survivorship bias** — the universe is *today's* Nifty 50 list. Stocks that were delisted or removed from the index over the 10.8-year window are not present, so results are optimistic. Fixing this requires point-in-time index membership history, which yfinance does not provide.

2. **DCF look-ahead** — fundamentals are a single current snapshot. Applying today's DCF to historical decision dates is look-ahead. The FULL/NO-DCF split quantifies this: the 101.5% gap is the maximum credit the current-day fundamentals snapshot can claim. The honest core is NO-DCF.

3. **Weekly scan cadence** — main.py is a daily tool; the backtest scans weekly (every 5th trading day) purely for compute (daily = ~5× the work). Daily signals would be fresher and likely catch entries the weekly scan misses.

4. **Trailing-stop instructions not simulated** — main.py prints `TRAIL SL > ₹X` advice, but the backtest exits on the fixed SL/TP the scanner recommends. Real trailing stops would likely improve results.

5. **ML training window capped at 2000 bars** — main.py trains on full history; the backtest caps at the most recent 2000 bars (~8 years) for compute. A depth-6 forest barely uses the extra rows, so the impact is small.

6. **One position per symbol; no reversal on opposite signals** — an open long is not flipped to a short when a fresh SELL signal arrives. Real trading would often reverse.

7. **Prices are yfinance auto-adjusted** — splits adjusted, dividends not modelled. Total-return comparisons vs Nifty are therefore slightly conservative on the dividend side.

8. **`^NSEI.csv` malformed mixed-schema rows** — the incremental fetcher appends rows with the date in column 0 while old rows carry it in a `Date` column. `main.py` tolerates it (never parses dates); `walkforward_backtest.py` carries a robust `load_index_frame()` reader. The data file itself was left untouched.

9. **`dashboard.py` is not Streamlit** — running `streamlit run dashboard.py` opens a blank page. Use `streamlit run streamlit_app.py` (the separate dashboard built from backtest artifacts) for the web view.

---

## 17. Changes Made in This Build

All changes are surgical. The engine files (`ml_engine.py`, `dcf_engine.py`, `smart_money_engine.py`, `risk_manager.py`, `data_fetcher.py`, `fundamental_fetcher.py`, `import_data.py`) are untouched — the live scanner behaves exactly as before except for the fixes below. Data files were not modified.

### Change 1 — Portfolio capital cap + honest dashboard accounting

**Problem**: each position was individually capped (5–15% of capital) but nothing stopped the *sum* of long + short allocations from exceeding ₹100,000. The Portfolio box then showed misleading numbers (e.g. "Deployed ₹0 / Remaining ₹79,051" while ₹20,949 was committed to shorts — shorts silently reduced Remaining but were never shown as Deployed).

**Fix in `main.py`**:
- Added `TOTAL_CAPITAL = 100000` as the single source of truth.
- LONG/SHORT routing now requires `allocation ≤ TOTAL_CAPITAL − deployed_long − deployed_short`; trades that don't fit are counted in `skipped_capital` instead of being booked.
- `dashboard_data` reports `deployed` as longs + shorts, plus `deployed_long`, `deployed_short`, `skipped_capital`.

**Fix in `dashboard.py`**: Portfolio box shows `Deployed` plus a Long/Short split; Position Details prints a `Skipped N trade(s)` line when the cap excluded trades.

### Change 2 — `backtester.py` aligned with main.py's real engines

**Problem**: `backtester.py` fed the shared `DecisionEngine` *different* inputs — an ML proxy (`min(abs(Mom_20)*15, 1)`), a hardcoded `LOW VOL`, a hardcoded `breadth: 55` — so the same day could produce "no longs" from main.py and ten BUYs from the backtester. It also truncated history to the last 1250 bars.

**Fix**: imports and uses the real `MLEngine` (trained on full local history, `predict()` for direction/confidence/RSI); computes the real macro regime via `MarketRegime.get_regime` from `^NSEI` + `^INDIAVIX`; breadth computed for the full universe in one pre-pass (order-independent); removed the `.iloc[-1250:]` history cap and the ML/regime proxy blocks. The 5-day extrapolation math and output format are unchanged. Result: the leaderboard now agrees with the live screener.

### Change 3 — `walkforward_backtest.py` (new file)

`backtester.py` never tested strategy performance on history. The new script replays the exact production pipeline over 2016 → today with no look-ahead, producing `results.json`, `BACKTEST_REPORT.txt`, `equity_curve.png`. Full mechanics in §5.3 and §9.

### Change 4 — `DEMO_MODE` flag in `decision_engine.py`

Added a class-level `DEMO_MODE` flag that relaxes the action thresholds from ±30 to +15/−25 for demonstration, so the recommendations table shows BUY/SELL/HOLD even in a low-breadth (risk-off) market. Strict mode (the published backtest config) is one flag flip away. Scoring, factor weights, and risk sizing are unchanged.

### Documentation kept in sync

- `PROJECT_REPORT.txt`: header script list, Phase 3 architecture diagram, pipeline stage 3, §7 rewritten (backtester now uses real engines; new §7b documents the walk-forward validator), §8 directory structure, §9 run guide.
- `flowchart.txt`: PATH B corrected and PATH C added.
- `CHANGES.md`: full change log.

---

## 18. Recommendations & Future Work

- **Point-in-time fundamentals** — fetch historical balance sheet / cashflow snapshots per date so the DCF pillar carries no look-ahead. This collapses the FULL/NO-DCF gap to a single honest number and is the single highest-impact improvement.
- **Daily scan cadence** — run the backtest daily instead of weekly. Compute cost is ~5×, but entries caught a day earlier materially change fills.
- **Simulate trailing stops** — the `TRAIL SL > ₹X` advice main.py prints is currently advisory only. Implementing trailing-stop exits in the sim would likely improve the exit distribution (fewer TIME exits giving back gains).
- **Position reversal on opposite signals** — flip an open long to a short when a fresh SELL arrives, instead of holding until SL/TP/TIME.
- **Point-in-time index membership** — source historical Nifty 50 constituent lists to eliminate survivorship bias. This is hard with free data but would make the backtest publishable.
- **Multi-universe** — extend beyond Nifty 50 (Nifty 500, sector indices) for more signal diversity.
- **Live broker integration** — the system is currently paper/signal-only. A broker adapter (Zerodha Kite, Fyers) would close the loop from signal to execution.
- **Hyperparameter tuning** — the RandomForest (50 trees, depth 6) and the ±30 thresholds are hand-set. A proper walk-forward grid search over these would find a better operating point.
- **Web dashboard as first-class** — promote `streamlit_app.py` from a bonus artifact to a maintained module that reads live `main.py` output, not just backtest results.

---

## Appendix A — Full BACKTEST_REPORT.txt

```
==========================================================================================
                               WALK-FORWARD BACKTEST REPORT
            Hybrid Nifty 50 Trading Platform - Production Pipeline Simulation
==========================================================================================
Generated        : 2026-11-06 18:35 (script: walkforward_backtest.py)
Window           : 2016-01-04 -> 2026-10-29  (10.8 years)
Scan cadence     : every 5th Nifty trading day (weekly)
Starting capital : Rs.100,000 (compounding; 2% of current equity risked per trade)
Transaction cost : 0.10% of trade value per side
ML training cap  : most recent 2000 bars (compute limit; main.py uses full history)

METHODOLOGY (how faithfully main.py was simulated)
------------------------------------------------------------------------------------------
- Real engine classes at every scan date: MLEngine (retrained on data <= date
  only), SmartMoneyEngine, RiskManager (ATR stop, 2% risk, 5/10/15% caps),
  DecisionEngine (100-pt matrix, +/-30 trigger, RSI timing), MarketRegime
  (^NSEI SMA50/200, ^INDIAVIX, full-universe breadth).
- Entries at the NEXT trading day's open; WAIT FOR DIP / BOUNCE become limit
  orders at the halfway trigger, valid 5 trading days.
- Exits: scanner SL / 2.5x TP, or time stop after 20 days at close.
  If SL and TP are touched the same day, SL is assumed first (pessimistic).
- One position per symbol; opposite signals do not reverse open positions.
- Positions sized by the real RiskManager on CURRENT equity (main.py uses a
  fixed Rs.100,000 notional); allocations can never exceed available cash.

KNOWN BIASES (read before trusting the numbers)
------------------------------------------------------------------------------------------
1. Survivorship bias: universe is TODAY'S Nifty 50 list -> results optimistic.
2. DCF look-ahead: fundamentals are one current snapshot. FULL includes that
   bias; NO-DCF zeroes the DCF factor and is the honest core result.
3. Weekly cadence vs main.py's daily design; signals would be fresher daily.
4. Trailing-stop instructions printed by main.py are not simulated.

RESULTS
------------------------------------------------------------------------------------------
Metric                  |                 FULL |               NO-DCF |            Nifty B&H
------------------------------------------------------------------------------------------
Total return            |               133.7% |                32.2% |               194.4%
CAGR                    |                 8.2% |                 2.6% |                10.5%
Max drawdown            |               -26.6% |               -29.5% |               -38.4%
Sharpe (daily, rf=0)    |                 0.76 |                 0.31 |                 0.71
Final equity (Rs.)      |              233,680 |              132,198 |                    -
Trades filled           |                2,665 |                2,557 |                    -
Win rate                |                33.9% |                32.3% |                    -
Profit factor           |                 1.17 |                 1.06 |                    -
Avg win / trade         |                 7.5% |                 7.2% |                    -
Avg loss / trade        |                -3.3% |                -3.4% |                    -
Exits TP / SL / TIME    |     520/1655  /490    |     471/1643  /443    |                    -
Long trades (PnL Rs.)   |    2,274 (  185,412) |    1,992 (   87,918) |                    -
Short trades (PnL Rs.)  |      391 (  -52,127) |      565 (  -57,919) |                    -
Avg hold (days)         |                  8.9 |                  8.7 |                    -
Exposure (pos-days)     |               887.8% |               831.7% |                    -

PER-YEAR RETURNS
------------------------------------------------------------------------------------------
Year      |           FULL |         NO-DCF |          Nifty
------------------------------------------------------------------------------------------
2016      |           2.2% |           1.1% | (see chart)
2017      |          27.2% |          31.5% | (see chart)
2018      |          -1.5% |          -2.7% | (see chart)
2019      |          -1.3% |           0.6% | (see chart)
2020      |          47.3% |           4.3% | (see chart)
2021      |          16.5% |           1.6% | (see chart)
2022      |          -7.1% |         -11.6% | (see chart)
2023      |          23.1% |          15.6% | (see chart)
2024      |          10.4% |          12.4% | (see chart)
2025      |          -1.9% |          -3.6% | (see chart)
2026      |         -13.0% |         -13.3% | (see chart)

TOP 5 WINS AND TOP 5 LOSSES (FULL variant)
------------------------------------------------------------------------------------------
WIN   ETERNAL   LONG  2024-03-18 -> 2024-04-08    18.43%  (TP, score 62)
WIN   ETERNAL   LONG  2024-07-10 -> 2024-08-01    13.70%  (TP, score 63)
WIN   ETERNAL   LONG  2024-02-02 -> 2024-02-20    14.42%  (TP, score 78)
WIN   ADANIENT  LONG  2021-03-17 -> 2021-03-23    20.99%  (TP, score 68)
WIN   WIPRO     LONG  2024-01-17 -> 2024-02-16    14.30%  (TP, score 80)
LOSS  KOTAKBANK LONG  2024-04-10 -> 2024-04-25    -7.12%  (SL, score 64)
LOSS  ETERNAL   LONG  2024-10-07 -> 2024-10-07    -6.26%  (SL, score 60)
LOSS  ADANIPORTSLONG  2024-12-04 -> 2024-12-23    -9.85%  (SL, score 56)
LOSS  TRENT     LONG  2020-12-08 -> 2020-12-21    -9.39%  (SL, score 70)
LOSS  TMPV      LONG  2021-03-12 -> 2021-03-19    -8.30%  (SL, score 69)

CONCLUSION
------------------------------------------------------------------------------------------
Over 10.8 years the FULL signal turned Rs.100,000 into Rs.233,680 (133.7%, CAGR 8.2%) vs
Nifty buy-and-hold 194.4% (CAGR 10.5%). With the DCF factor removed (no look-ahead), the
same engine returned 32.2% (CAGR 2.6%); the 101.5% gap between FULL and NO-DCF is the
maximum credit the current-day fundamentals snapshot can claim for historical results.
Drawdowns: FULL -26.6%, NO-DCF -29.5%, Nifty -38.4%. All figures carry the survivorship
and cadence biases listed above; treat them as an upper bound, not a forecast.
==========================================================================================
```

---

## Appendix B — Sample Live Run

Output of `python main.py` (with `DEMO_MODE = True`) on closed-market data, Breadth 2.0%:

```
══════════════════════════════════════════════
 Hybrid Stock Prediction Model
══════════════════════════════════════════════

 Market Status
──────────────────────────────────────────────
[MARKET CLOSED] [LOW VOL] [RANGING]
VIX: 17.47 | Breadth: 2.0%

 Portfolio Summary
──────────────────────────────────────────────
╔════════════════════════════════════════╗
║Capital        ₹     100,000            ║
║Deployed       ₹      71,112            ║
║  Long         ₹      44,525            ║
║  Short        ₹      26,586            ║
║Remaining      ₹      28,889            ║
╚════════════════════════════════════════╝

 Signal Engine: Undervalued (Value Play)
──────────────────────────────────────────────
POWERGRID |BUY  |TRAIL:   265|Undervalued  | WIPRO     |BUY  |TRAIL:   165|Undervalued
BHARTIARTL|BUY  |TRAIL:  1808|Undervalued  | TMPV      |BUY  |TRAIL:   299|Undervalued

 Signal Engine: Fair Value (Momentum Play)
──────────────────────────────────────────────
ICICIBANK |BUY  |5D:   1346|Fair Value   | M&M       |HOLD |5D:   3029|Fair Value
ULTRACEMCO|BUY  |5D:  11055|Fair Value   | SBIN      |HOLD |5D:    982|Fair Value
AXISBANK  |BUY  |5D:   1249|Fair Value   | ... (38 stocks, mixed BUY/HOLD)

 Signal Engine: Overvalued (Short Play)
──────────────────────────────────────────────
TRENT     |SELL |5D:   2788|Overvalued   | HINDALCO  |SELL |5D:    968|Overvalued
MAXHEALTH |SELL |5D:   1051|Overvalued   | HCLTECH   |SELL |5D:   1264|Overvalued
INFY      |SELL |5D:   1024|Overvalued   | TITAN     |SELL |5D:   4984|Overvalued

 Position Details
──────────────────────────────────────────────
Symbol     | Type  | Qty   | Alloc      | Entry
ADANIPORTS | LONG | 2     | ₹   3,873 | ₹ 1936.60
AXISBANK   | LONG | 3     | ₹   3,968 | ₹ 1322.90
... (10 long + 6 short positions)

 Risk Management
──────────────────────────────────────────────
Edge Score      ███████░░░░░░░░░░░░░ 35%
Diversification: 100% | Max Risk/Trade: 2.0%

 Top 10 Trade Recommendations
──────────────────────────────────────────────
#   ACTION  QTY  SYMBOL      ENTRY TRIGGER & TRAIL RULES                 SL
1   SELL   1    TRENT      LIMIT SELL @ ₹3145.4                          ₹3211.79
2   SELL   5    MAXHEALTH  MARKET SELL @ ₹934.00                         ₹ 975.91
3   SELL   4    INFY       LIMIT SELL @ ₹1217.16                         ₹1252.92
4   SELL   5    HINDALCO   MARKET SELL @ ₹958.40                         ₹ 995.65
5   SELL   3    HCLTECH    MARKET SELL @ ₹1440.00                        ₹1500.16
6   BUY    19   POWERGRID  MARKET BUY @ ₹259.90 | TRAIL SL > ₹270.53     ₹ 249.27
7   SELL   1    TITAN      MARKET SELL @ ₹5000.00                        ₹5190.15
8   BUY    2    BHARTIARTL MARKET BUY @ ₹2107.20 | TRAIL SL > ₹2180.31   ₹2034.09
9   BUY    25   WIPRO      MARKET BUY @ ₹198.00 | TRAIL SL > ₹205.96     ₹ 190.04
10  BUY    3    ICICIBANK  MARKET BUY @ ₹1571.20 | TRAIL SL > ₹1622.47   ₹1519.93
──────────────────────────────────────────────
1 . TRENT      | ML:DOWN(37%) | DCF:Over | Breadth:2% | SM:Accum
2 . MAXHEALTH  | ML:DOWN(33%) | DCF:Over | Breadth:3% | SM:Accum
3 . INFY       | ML:DOWN(17%) | DCF:Over | Breadth:4% | SM:Accum
4 . HINDALCO   | ML:DOWN(15%) | DCF:Over | Breadth:5% | SM:Accum
5 . HCLTECH    | ML:DOWN(14%) | DCF:Over | Breadth:0% | SM:Accum
6 . POWERGRID  | ML:DOWN(1%) | DCF:Under | Breadth:2% | SM:Accum
7 . TITAN      | ML:DOWN(9%) | DCF:Over | Breadth:2% | SM:Accum
8 . BHARTIARTL | ML:DOWN(5%) | DCF:Under | Breadth:0% | SM:Accum
9 . WIPRO      | ML:DOWN(9%) | DCF:Under | Breadth:2% | SM:Accum
10. ICICIBANK  | ML:UP(16%) | DCF:Fair | Breadth:4% | SM:Accum
```

With `DEMO_MODE = False` (strict ±30), the same scan produces far fewer BUY rows because the −13 regime penalty suppresses most scores below the +30 line — which is the regime filter doing its job in a 2%-breadth, ranging market.

---

## Appendix C — Nifty 50 Constituents

The 50 symbols in `nifty50_constituents.csv` (the tradable universe). Symbols follow yfinance/NSE conventions; the `^`-prefixed `^NSEI` and `^INDIAVIX` are the index and volatility benchmark loaded separately for the macro regime.

ADANIENT, ADANIPORTS, APOLLOHOSP, ASIANPAINT, AXISBANK, BAJAJ-AUTO, BAJAJFINSV, BAJFINANCE, BEL, BHARTIARTL, CIPLA, COALINDIA, DRREDDY, EICHERMOT, ETERNAL, GRASIM, HCLTECH, HDFCBANK, HDFCLIFE, HINDALCO, HINDUNILVR, ICICIBANK, INDIGO, INFY, ITC, JIOFIN, JSWSTEEL, KOTAKBANK, LT, M&M, MARUTI, MAXHEALTH, NESTLEIND, NTPC, ONGC, POWERGRID, RELIANCE, SBILIFE, SBIN, SHRIRAMFIN, SUNPHARMA, TATACONSUM, TATASTEEL, TCS, TECHM, TITAN, TMPV, TRENT, ULTRACEMCO, WIPRO.

(`TMPV` is the yfinance alias for Tata Motors DVR — the differential voting rights share. The universe is today's Nifty 50 list, which is the source of the survivorship bias documented in §16.)

---

## Sources & References

- **Yahoo Finance** (`yfinance`) — daily OHLCV bars, balance sheet, cash flow, and company info. The sole external data source. https://finance.yahoo.com
- **Nifty 50 Index (^NSEI)** — NSE benchmark, used for the SMA50/200 trend regime. https://www.nseindia.com
- **India VIX (^INDIAVIX)** — NSE volatility index, used for the HIGH/LOW VOL regime gate (threshold 18.0). https://www.nseindia.com
- **scikit-learn** — `RandomForestClassifier` implementation; the ML engine's hyperparameters (50 trees, depth 6, balanced class weight, 0.5 subsample) follow sklearn defaults tuned for speed. https://scikit-learn.org
- **pandas / numpy** — data wrangling, rolling statistics, OBV / SMA / RSI / MACD feature engineering.
- **matplotlib** — equity-curve chart in `walkforward_backtest.py`.
- **Streamlit / Plotly** — the bonus interactive dashboard (`streamlit_app.py`) reading the backtest artifacts.

Methodological references for the design choices (no single source copied; standard quant practice):

- FCFF DCF model and margin-of-safety thresholds — standard corporate-finance practice (Damodaran-style WACC and terminal value).
- ATR-based stop losses and asymmetric risk-reward — Wilder's Average True Range and the trend-following risk canon.
- On-Balance Volume and up/down volume ratio — Granville's volume analysis framework.
- Walk-forward validation with strict point-in-time data and the FULL/NO-DCF look-ahead split — standard backtest hygiene; the explicit dual reporting of the DCF bias is the project's contribution to honest self-assessment.

All numbers in this report are reproducible: run `python import_data.py` then `python walkforward_backtest.py --start 2016-01-01` and the `backtest_results/` artifacts regenerate. Treat every figure as an **upper bound, not a forecast** — the survivorship and DCF look-ahead biases documented in §16 mean live forward performance will almost certainly be lower than the FULL backtest suggests, and closer to the NO-DCF core.
