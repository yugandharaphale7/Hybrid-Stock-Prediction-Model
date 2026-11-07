# CHANGES — heart code (3 requested builds)

All changes are surgical: engine files (`ml_engine.py`, `dcf_engine.py`,
`smart_money_engine.py`, `decision_engine.py`, `risk_manager.py`,
`data_fetcher.py`, `fundamental_fetcher.py`, `import_data.py`) are untouched,
and the live scanner behaves exactly as before except for the two fixes listed
below. Data files were not modified.

## 1. Portfolio capital cap + honest dashboard accounting

Problem: each position was individually capped (5–15% of capital), but nothing
stopped the *sum* of long + short allocations from exceeding the ₹100,000
portfolio. The Portfolio box then showed misleading numbers (observed:
"Deployed ₹0 / Remaining ₹79,051" while ₹20,949 was committed to shorts —
shorts silently reduced Remaining but were never shown as Deployed).

### `main.py`
- Added `TOTAL_CAPITAL = 100000` as the single source of truth (used by
  `RiskManager`, the cap check and the dashboard).
- LONG/SHORT trade routing now requires
  `allocation <= TOTAL_CAPITAL - deployed_long - deployed_short`; trades that
  don't fit are counted in a new `skipped_capital` counter instead of being
  booked.
- `dashboard_data` now reports `deployed` as longs + shorts, and adds
  `deployed_long`, `deployed_short` and `skipped_capital`.

### `dashboard.py`
- Portfolio box now shows `Deployed` plus a `Long` / `Short` split.
- Position Details prints `Skipped N trade(s): not enough unspent capital left.`
  when the cap excluded trades.

## 2. backtester.py aligned with main.py's real engines

Problem: `backtester.py` fed the shared `DecisionEngine` *different* inputs —
an ML *proxy* (`min(abs(Mom_20)*15, 1)`), a hardcoded `LOW VOL`, a hardcoded
`breadth: 55` — so the same day could produce "no longs" from `main.py` and ten
BUYs from the backtester. It also truncated history to the last 1250 bars.

### `backtester.py`
- Imports and uses the real `MLEngine` (trained on the symbol's full local
  history, `predict()` for direction/confidence/RSI) exactly like `main.py`;
  symbols failing the 500-bar gate are skipped, same as `main.py`.
- Computes the real macro regime via `MarketRegime.get_regime` from `^NSEI` +
  `^INDIAVIX`; breadth is computed for the full universe in one pre-pass so it
  is order-independent (a slight improvement over main.py's incremental breadth).
- Removed the `.iloc[-1250:]` history cap and the ML-proxy / regime-proxy blocks.
- The 5-day extrapolation leaderboard math and output format are unchanged.
- Result: the leaderboard now agrees with the live screener (verified on the
  bundled data: both produce the same short-side setups instead of contradicting
  each other).

## 3. walkforward_backtest.py — real walk-forward backtest (new file)

`backtester.py` never tested strategy performance on history, so a new script
replays the *exact production pipeline* over 2016 → today with no look-ahead:

- Every 5th Nifty trading day (weekly), the full scan is re-run as it would
  have looked on that day: real `MLEngine` retrained strictly on data ≤ that
  date (the engine's own 5-day-forward target drops unlabeled rows, so no
  leakage), real `SmartMoneyEngine`, `RiskManager` (ATR stop, 2% risk,
  5/10/15% caps on current equity), `DecisionEngine` and `MarketRegime`
  (^NSEI trend, ^INDIAVIX, full-universe breadth at that date).
- Execution: entries at the next day's open (or a 5-day limit trigger for
  WAIT FOR DIP / BOUNCE); exits at the scanner's SL / 2.5× TP or a 20-day time
  stop; if SL and TP are touched on one day the SL is assumed first
  (pessimistic); 0.1% cost per side; one position per symbol; allocations can
  never exceed available cash.
- Two variants per run: **FULL** (production signal incl. DCF) and **NO-DCF**
  (DCF factor zeroed) — this quantifies the one unavoidable bias: fundamentals
  are a single current snapshot, which is look-ahead for historical dates.
- The ML training window is capped at the most recent 2000 bars purely for
  compute (main.py uses full history; disclosed in the report).

Run: `python walkforward_backtest.py --start 2016-01-01`
Outputs to `backtest_results/`: `results.json` (all metrics + every trade),
`BACKTEST_REPORT.txt` (methodology, biases, results, per-year table, top
wins/losses, conclusion), `equity_curve.png` (FULL vs NO-DCF vs Nifty B&H).

### Known biases (also printed in the report)
1. Survivorship bias: the universe is today's Nifty 50 list.
2. DCF look-ahead: see FULL vs NO-DCF above.
3. Weekly cadence (main.py is a daily tool; daily = 5× compute).
4. Trailing-stop *instructions* printed by main.py are not simulated.

## Documentation kept in sync
- `PROJECT_REPORT.txt`: header script list, Phase 3 architecture diagram,
  pipeline stage 3, §7 rewritten (backtester now uses real engines; new §7b
  documents the walk-forward validator), §8 directory structure, §9 run guide.
- `flowchart.txt`: PATH B corrected (it previously described fills / trailing
  stops / expectancy outputs that never existed) and PATH C added.

## Also noted (not changed, by design)
- `data/ohlcv/^NSEI.csv` has malformed mixed-schema rows (the incremental
  fetcher appended rows with the date in column 0). `main.py` tolerates it
  because it never parses dates; `walkforward_backtest.py` contains a robust
  reader (`load_index_frame`) for it. The data file itself was left untouched.
