"""
Walk-Forward Backtest for the Hybrid Nifty 50 Trading Platform
==============================================================
Simulates the EXACT production pipeline (the same engine classes main.py uses)
on historical data, retraining the ML model only on bars available at each
decision date. No look-ahead enters the features, targets, indicators or regime.

How it works
------------
- Every `--cadence` Nifty trading days (default 5 = weekly) the full scan is
  re-run exactly as it would have looked on that day:
    * MLEngine retrained on data up to that date only (its 5-day forward target
      drops the last 5 rows automatically, so no future leak)
    * SmartMoneyEngine, RiskManager (ATR stops, 2% risk, 5/10/15% caps),
      DecisionEngine (100-pt matrix, +/-30 trigger, RSI timing) and
      MarketRegime (^NSEI SMA50/200, ^INDIAVIX, full-universe breadth)
  are the REAL classes imported from the codebase.
- Execution model:
    * entries at the NEXT trading day's open (market orders), or at a limit
      trigger valid 5 trading days for WAIT FOR DIP / WAIT FOR BOUNCE timing
    * exits at the scanner's ATR stop-loss / 2.5x take-profit, or a time stop
      after 20 trading days (close). If SL and TP are both touched on the same
      day, the stop is assumed to hit first (pessimistic)
    * transaction cost 0.1% of trade value per side (--cost-pct)
    * portfolio cap: allocations can never exceed available cash
- Two signal variants are simulated side by side over the same dates:
    * FULL   - production signal including the DCF factor. NOTE: fundamentals
               are a single current snapshot, which is look-ahead for past dates.
    * NO-DCF - identical engine with the DCF factor zeroed, quantifying how much
               of the result depends on that look-ahead.

Known limitations (disclosed on purpose)
----------------------------------------
- Survivorship bias: the universe is TODAY'S Nifty 50 list.
- DCF look-ahead: see the FULL vs NO-DCF split above.
- ML training window capped at the most recent 2000 bars (~8 years) purely for
  compute; main.py trains on full history (a depth-6 forest barely uses them).
- Weekly scan cadence (main.py is a daily tool; daily = 5x compute).
- Trailing-stop *instructions* printed by main.py are not simulated; exits use
  the fixed SL/TP levels the scanner recommends.
- Open positions are not reversed on opposite signals; one position per symbol.
- Prices are yfinance auto-adjusted (splits adjusted, dividends not modelled).

Usage
-----
    python walkforward_backtest.py --start 2016-01-01 [--end 2026-10-29]
Outputs to ./backtest_results/: results.json, BACKTEST_REPORT.txt, equity_curve.png
"""

import os
import sys
import json
import time
import pickle
import argparse
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ml_engine import MLEngine
from dcf_engine import run_dcf_from_local
from risk_manager import MarketRegime, RiskManager
from decision_engine import DecisionEngine
from smart_money_engine import SmartMoneyEngine

MAX_HOLD_DAYS = 20      # time stop: exit at close after this many trading days
LIMIT_TTL_DAYS = 5      # WAIT FOR DIP / BOUNCE limit orders live this many days
MARKET_FILL_SLACK = 3   # market orders may fill within this many days (halts)
TRAIN_WINDOW = 2000     # ML training rows cap (most recent bars), see docstring

TRAIN_GATE = 500        # same minimum-bar gate as main.py / ml_engine.train


# --------------------------------------------------------------------------- #
# Data loading & pre-computation
# --------------------------------------------------------------------------- #
def load_stock(symbol):
    path = os.path.join('data/ohlcv', f"{symbol}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True,
                         date_format='%Y-%m-%d')
    except Exception:
        return None
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    df = df.dropna()
    if len(df) < TRAIN_GATE or float(df['Close'].iloc[-1]) <= 0:
        return None
    return df


def load_index_frame(path):
    """Robust ^NSEI/^INDIAVIX reader. The incremental fetcher leaves ^NSEI.csv
    with mixed rows: old rows carry the date in a 'Date' column (with stray
    leading index columns), appended rows carry it in column 0 instead.
    Coalesce both, keep the Close column, return a clean DatetimeIndex frame."""
    raw = pd.read_csv(path)
    raw.columns = [str(c).strip() for c in raw.columns]
    if 'Date' in raw.columns:
        dcol = raw['Date'].astype(str).str.strip()
        col0 = raw[raw.columns[0]].astype(str).str.strip()
        dates = dcol.where((dcol != '') & (dcol.str.lower() != 'nan'), col0)
    else:
        dates = raw[raw.columns[0]].astype(str)
    idx = pd.to_datetime(dates, errors='coerce', format='mixed')
    out = raw[['Close']].apply(pd.to_numeric, errors='coerce')
    out.index = idx
    out = out[out.index.notna()]
    return out[~out.index.duplicated(keep='last')].sort_index()


def build_stock_data(symbols):
    """Per-stock structures for O(1) date lookups during the simulation."""
    stock_data = {}
    for sym in symbols:
        df = load_stock(sym)
        if df is None:
            continue
        stock_data[sym] = {
            'df': df,
            'dates': df.index.values,
            'row_by_date': {ts: i for i, ts in enumerate(df.index)},
            'above50': (df['Close'] > df['Close'].rolling(50).mean()).values,
        }
    return stock_data


def load_info_cache(symbols):
    """Same RAM caching of fundamentals JSONs as main.py."""
    cache = {}
    for sym in symbols:
        path = os.path.join('data/fundamentals', f"{sym}_info.json")
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    cache[sym] = json.load(f)
            except Exception:
                pass
    return cache


# --------------------------------------------------------------------------- #
# Phase 1: event collection (the expensive, strictly point-in-time pass)
# --------------------------------------------------------------------------- #
def collect_events(symbols, stock_data, info_cache, nifty, vix, scan_dates,
                   out_dir):
    """Run the real engines at every scan date using only data <= that date.

    Returns {scan_date(Timestamp): [event dicts sorted by |score| desc]}.
    """
    ckpt = os.path.join(out_dir, 'events.pkl')
    params = {'symbols': list(stock_data.keys()),
              'scan_dates': [str(d.date()) for d in scan_dates]}
    if os.path.exists(ckpt):
        try:
            with open(ckpt, 'rb') as f:
                saved = pickle.load(f)
            if saved.get('params') == params:
                print("Reusing cached events (same universe & dates).")
                return saved['events']
        except Exception:
            pass

    decision_engine = DecisionEngine()
    sm_engine = SmartMoneyEngine()
    events_by_date = {}
    t0 = time.time()

    for k, t in enumerate(scan_dates):
        # Full-universe breadth at date t (each stock's last flag on/before t)
        flags = []
        for sym, sd in stock_data.items():
            i = sd['dates'].searchsorted(np.datetime64(t), side='right') - 1
            if i >= 49:
                flags.append(1 if sd['above50'][i] else 0)
        breadth = (sum(flags) / len(flags) * 100) if flags else 50.0

        regime = MarketRegime.get_regime(nifty[nifty.index <= t],
                                         vix[vix.index <= t] if vix is not None else None,
                                         breadth)
        is_trending = regime['trend'] == 'TRENDING'
        day_events = []

        for sym, sd in stock_data.items():
            i = sd['dates'].searchsorted(np.datetime64(t), side='right') - 1
            if i < TRAIN_GATE - 1:          # same 500-bar gate as main.py
                continue
            df_t = sd['df'].iloc[:i + 1]    # strictly data <= t
            price = float(df_t['Close'].iloc[-1])
            if price <= 0:
                continue
            ml = MLEngine()
            ml.model.n_jobs = 1             # many small fits: parallel overhead hurts
            if TRAIN_WINDOW and len(df_t) > TRAIN_WINDOW:
                ml_df = df_t.iloc[-TRAIN_WINDOW:]
            else:
                ml_df = df_t
            if not ml.train(ml_df):
                continue
            ml_res = ml.predict(df_t)
            sm_res = sm_engine.analyze(df_t)
            dcf_res = run_dcf_from_local(sym, price, info_override=info_cache.get(sym))

            sig_full = decision_engine.generate_signal(ml_res, dcf_res, regime, sm_res)
            sig_nodcf = decision_engine.generate_signal(ml_res, None, regime, sm_res)

            day_events.append({
                'symbol': sym, 'date': t, 'i': i, 'price': price,
                'is_trending': is_trending,
                'dcf_status': dcf_res['status'] if dcf_res else None,
                'sig_full': sig_full, 'sig_nodcf': sig_nodcf,
            })

        day_events.sort(key=lambda e: abs(e['sig_full'].get('raw_score', 0)),
                        reverse=True)
        events_by_date[t] = day_events

        if (k + 1) % 20 == 0 or k == len(scan_dates) - 1:
            el = time.time() - t0
            eta = el / (k + 1) * (len(scan_dates) - k - 1)
            print(f"  scan {k + 1}/{len(scan_dates)} ({t.date()}) "
                  f"elapsed {el:.0f}s eta {eta:.0f}s", flush=True)

    with open(ckpt, 'wb') as f:
        pickle.dump({'params': params, 'events': events_by_date}, f)
    return events_by_date

# --------------------------------------------------------------------------- #
# Phase 2: portfolio simulation over the collected signal events
# --------------------------------------------------------------------------- #
def simulate(events_by_date, scan_dates, use_dcf, stock_data, all_days,
             day_index, capital0, cost_pct):
    """Event-driven portfolio sim. Entries next open (or limit trigger),
    exits at SL/TP/time-stop. Returns (trades, equity Series, open-at-end)."""
    cash = float(capital0)
    positions = {}          # symbol -> open position
    pending = []            # live orders
    pending_syms = set()
    trades = []
    eq_index, eq_values = [], []
    scan_set = set(scan_dates)
    n_days = len(all_days)

    def mark_value(sym, p):
        return p['qty'] * p['last_close'] if p['dir'] == 'LONG' \
            else p['qty'] * (2 * p['entry_price'] - p['last_close'])

    def mark_equity():
        return cash + sum(mark_value(s, p) for s, p in positions.items())

    def try_fill(o, fill_price, k, d):
        nonlocal cash
        if o['qty'] * fill_price > cash:    # portfolio cap: never exceed cash
            return False
        fee_in = cost_pct / 100.0 * o['qty'] * fill_price
        cash -= o['qty'] * fill_price + fee_in
        positions[o['symbol']] = {
            'dir': o['dir'], 'qty': o['qty'], 'entry_price': fill_price,
            'entry_date': d, 'entry_k': k, 'sl': o['sl'], 'tp': o['tp'],
            'timing': o['timing'], 'score': o['score'],
            'dcf_status': o['dcf_status'], 'order_type': o['type'],
            'fee_in': fee_in, 'last_close': fill_price,
        }
        return True

    def close_position(sym, exit_price, k, d, reason):
        nonlocal cash
        p = positions.pop(sym)
        fee_out = cost_pct / 100.0 * p['qty'] * exit_price
        cash += p['qty'] * exit_price - fee_out if p['dir'] == 'LONG' else \
            p['qty'] * p['entry_price'] + p['qty'] * (p['entry_price'] - exit_price) - fee_out
        pnl = p['qty'] * ((exit_price - p['entry_price']) if p['dir'] == 'LONG'
                          else (p['entry_price'] - exit_price)) - p['fee_in'] - fee_out
        trades.append({
            'symbol': sym, 'dir': p['dir'],
            'entry_date': str(p['entry_date'].date()),
            'entry_price': round(p['entry_price'], 2), 'qty': p['qty'],
            'exit_date': str(d.date()), 'exit_price': round(exit_price, 2),
            'reason': reason, 'hold_days': k - p['entry_k'],
            'pnl': round(pnl, 2),
            'pnl_pct': round(pnl / (p['qty'] * p['entry_price']), 6),
            'score': p['score'], 'timing': p['timing'],
            'dcf_status': p['dcf_status'], 'order_type': p['order_type'],
        })

    for k, d in enumerate(all_days):
        # --- 1. fills at the open -------------------------------------------
        for o in pending[:]:
            if o['type'] == 'MARKET':
                if o['fill_date'] != d:
                    continue
                pending.remove(o)
                pending_syms.discard(o['symbol'])
                ri = stock_data[o['symbol']]['row_by_date'].get(d)
                if ri is None:
                    continue
                try_fill(o, float(stock_data[o['symbol']]['df']['Open'].iloc[ri]), k, d)
                continue
            # LIMIT order: fill when the trigger is touched, expire after TTL
            sd = stock_data[o['symbol']]
            ri = sd['row_by_date'].get(d)
            if ri is None:
                if d > o['expiry']:
                    pending.remove(o)
                    pending_syms.discard(o['symbol'])
                continue
            row = sd['df'].iloc[ri]
            fill_price = None
            if o['dir'] == 'LONG':
                if row['Open'] <= o['trigger']:
                    fill_price = float(row['Open'])
                elif row['Low'] <= o['trigger']:
                    fill_price = o['trigger']
            else:
                if row['Open'] >= o['trigger']:
                    fill_price = float(row['Open'])
                elif row['High'] >= o['trigger']:
                    fill_price = o['trigger']
            if fill_price is not None:
                pending.remove(o)
                pending_syms.discard(o['symbol'])
                try_fill(o, fill_price, k, d)
            elif d >= o['expiry']:
                pending.remove(o)
                pending_syms.discard(o['symbol'])

        # --- 2. exits during the day (SL checked before TP: pessimistic) ----
        for sym in list(positions.keys()):
            p = positions[sym]
            sd = stock_data[sym]
            ri = sd['row_by_date'].get(d)
            if ri is None:
                continue
            row = sd['df'].iloc[ri]
            p['last_close'] = float(row['Close'])
            exit_price, reason = None, None
            if p['dir'] == 'LONG':
                if row['Low'] <= p['sl']:
                    exit_price, reason = min(float(row['Open']), p['sl']), 'SL'
                elif row['High'] >= p['tp']:
                    exit_price, reason = max(float(row['Open']), p['tp']), 'TP'
            else:
                if row['High'] >= p['sl']:
                    exit_price, reason = max(float(row['Open']), p['sl']), 'SL'
                elif row['Low'] <= p['tp']:
                    exit_price, reason = min(float(row['Open']), p['tp']), 'TP'
            if exit_price is None and k - p['entry_k'] >= MAX_HOLD_DAYS:
                exit_price, reason = float(row['Close']), 'TIME'
            if exit_price is not None:
                close_position(sym, exit_price, k, d, reason)

        # --- 3. new orders at scan dates (signals ranked by |score|) --------
        if d in scan_set:
            equity_now = mark_equity()
            for ev in events_by_date.get(d, []):
                sig = ev['sig_full'] if use_dcf else ev['sig_nodcf']
                sym = ev['symbol']
                if sig['action'] == 'HOLD' or sym in positions or sym in pending_syms:
                    continue
                sd = stock_data[sym]
                df_tail = sd['df'].iloc[:ev['i'] + 1].tail(60)
                risk_mgr = RiskManager(total_capital=max(equity_now, 1.0))
                pos = risk_mgr.calculate_position(
                    ev['price'], sig['strength'], df_tail,
                    action=sig['action'], is_trending=ev['is_trending'])
                if pos['quantity'] <= 0:
                    continue
                o = {'symbol': sym,
                     'dir': 'LONG' if sig['action'] == 'BUY' else 'SHORT',
                     'qty': pos['quantity'], 'sl': pos['stop_loss'],
                     'tp': pos['take_profit'], 'timing': sig['timing'],
                     'score': sig.get('raw_score', 0),
                     'dcf_status': ev['dcf_status'], 'signal_date': d}
                if "DIP" in sig['timing']:
                    o['type'] = 'LIMIT'
                    o['trigger'] = round(ev['price'] - (ev['price'] - pos['stop_loss']) / 2, 2)
                elif "BOUNCE" in sig['timing']:
                    o['type'] = 'LIMIT'
                    o['trigger'] = round(ev['price'] + (pos['stop_loss'] - ev['price']) / 2, 2)
                else:
                    o['type'] = 'MARKET'
                    fill_day = next((all_days[j] for j in range(k + 1, min(k + 1 + MARKET_FILL_SLACK, n_days))
                                     if sd['row_by_date'].get(all_days[j]) is not None), None)
                    if fill_day is None:
                        continue
                    o['fill_date'] = fill_day
                if o['type'] == 'LIMIT':
                    if k + LIMIT_TTL_DAYS >= n_days:
                        continue
                    o['expiry'] = all_days[k + LIMIT_TTL_DAYS]
                pending.append(o)
                pending_syms.add(sym)

        # --- 4. daily equity mark -------------------------------------------
        eq_index.append(d)
        eq_values.append(mark_equity())

    return trades, pd.Series(eq_values, index=pd.DatetimeIndex(eq_index)), len(positions)

# --------------------------------------------------------------------------- #
# Metrics, report, chart
# --------------------------------------------------------------------------- #
def compute_metrics(trades, eq, benchmark_close, capital0, n_days):
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    final = float(eq.iloc[-1])
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    gross_win = sum(t['pnl'] for t in wins)
    gross_loss = -sum(t['pnl'] for t in losses)
    daily = eq.pct_change().dropna()

    yearly_idx = eq.resample('YE').last()
    yearly = [yearly_idx.iloc[0] / capital0 - 1.0] + \
        list(yearly_idx.pct_change().dropna())
    yearly = [float(v) for v in yearly]

    def ret_stats(series):
        tot = float(series.iloc[-1] / series.iloc[0] - 1)
        c = (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1 if series.iloc[0] > 0 else float('nan')
        d = series.pct_change().dropna()
        s = float(d.mean() / d.std() * np.sqrt(252)) if d.std() > 0 else 0.0
        mdd = float((series / series.cummax() - 1).min())
        return tot, float(c), mdd, s

    tot, cagr, mdd, sharpe = ret_stats(eq)
    b_tot, b_cagr, b_mdd, b_sharpe = ret_stats(benchmark_close)

    return {
        'total_return': tot, 'cagr': cagr, 'max_drawdown': mdd, 'sharpe': sharpe,
        'final_equity': round(final, 2),
        'trades': len(trades),
        'win_rate': len(wins) / len(trades) if trades else 0.0,
        'profit_factor': gross_win / gross_loss if gross_loss > 0 else float('inf'),
        'avg_win_pct': float(np.mean([t['pnl_pct'] for t in wins])) if wins else 0.0,
        'avg_loss_pct': float(np.mean([t['pnl_pct'] for t in losses])) if losses else 0.0,
        'exits': {r: sum(1 for t in trades if t['reason'] == r) for r in ('TP', 'SL', 'TIME')},
        'long_trades': len([t for t in trades if t['dir'] == 'LONG']),
        'short_trades': len([t for t in trades if t['dir'] == 'SHORT']),
        'long_pnl': round(sum(t['pnl'] for t in trades if t['dir'] == 'LONG'), 2),
        'short_pnl': round(sum(t['pnl'] for t in trades if t['dir'] == 'SHORT'), 2),
        'avg_hold_days': float(np.mean([t['hold_days'] for t in trades])) if trades else 0.0,
        'exposure_pct': sum(t['hold_days'] for t in trades) / max(n_days, 1),
        'yearly': yearly,
        'bench_total_return': b_tot, 'bench_cagr': b_cagr,
        'bench_max_drawdown': b_mdd, 'bench_sharpe': b_sharpe,
    }


def _pct(x):
    return f"{x * 100:,.1f}%" if np.isfinite(x) else "n/a"


def _num(x):
    return f"{x:,.2f}" if np.isfinite(x) else "n/a"


def write_report(out_path, params, m_full, m_nodcf, trades_full, nifty_span):
    w = 90
    L = []
    L.append("=" * w)
    L.append("WALK-FORWARD BACKTEST REPORT".center(w))
    L.append("Hybrid Nifty 50 Trading Platform - Production Pipeline Simulation".center(w))
    L.append("=" * w)
    L.append(f"Generated        : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} "
             f"(script: walkforward_backtest.py)")
    L.append(f"Window           : {params['start']} -> {params['end']}  "
             f"({nifty_span:.1f} years)")
    L.append(f"Scan cadence     : every {params['cadence']}th Nifty trading day (weekly)")
    L.append(f"Starting capital : Rs.{params['capital']:,.0f} (compounding; "
             f"2% of current equity risked per trade)")
    L.append(f"Transaction cost : {params['cost_pct']:.2f}% of trade value per side")
    L.append(f"ML training cap  : most recent {TRAIN_WINDOW} bars (compute limit; "
             f"main.py uses full history)")
    L.append("")
    L.append("METHODOLOGY (how faithfully main.py was simulated)")
    L.append("-" * w)
    for line in [
        "- Real engine classes at every scan date: MLEngine (retrained on data <= date",
        "  only), SmartMoneyEngine, RiskManager (ATR stop, 2% risk, 5/10/15% caps),",
        "  DecisionEngine (100-pt matrix, +/-30 trigger, RSI timing), MarketRegime",
        "  (^NSEI SMA50/200, ^INDIAVIX, full-universe breadth).",
        "- Entries at the NEXT trading day's open; WAIT FOR DIP / BOUNCE become limit",
        f"  orders at the halfway trigger, valid {LIMIT_TTL_DAYS} trading days.",
        f"- Exits: scanner SL / 2.5x TP, or time stop after {MAX_HOLD_DAYS} days at close.",
        "  If SL and TP are touched the same day, SL is assumed first (pessimistic).",
        "- One position per symbol; opposite signals do not reverse open positions.",
        "- Positions sized by the real RiskManager on CURRENT equity (main.py uses a",
        "  fixed Rs.100,000 notional); allocations can never exceed available cash.",
    ]:
        L.append(line)
    L.append("")
    L.append("KNOWN BIASES (read before trusting the numbers)")
    L.append("-" * w)
    for line in [
        "1. Survivorship bias: universe is TODAY'S Nifty 50 list -> results optimistic.",
        "2. DCF look-ahead: fundamentals are one current snapshot. FULL includes that",
        "   bias; NO-DCF zeroes the DCF factor and is the honest core result.",
        "3. Weekly cadence vs main.py's daily design; signals would be fresher daily.",
        "4. Trailing-stop instructions printed by main.py are not simulated.",
    ]:
        L.append(line)
    L.append("")
    L.append("RESULTS")
    L.append("-" * w)
    L.append(f"{'Metric':<24}| {'FULL':>20} | {'NO-DCF':>20} | {'Nifty B&H':>20}")
    L.append("-" * w)
    rows = [
        ("Total return", _pct(m_full['total_return']), _pct(m_nodcf['total_return']), _pct(m_full['bench_total_return'])),
        ("CAGR", _pct(m_full['cagr']), _pct(m_nodcf['cagr']), _pct(m_full['bench_cagr'])),
        ("Max drawdown", _pct(m_full['max_drawdown']), _pct(m_nodcf['max_drawdown']), _pct(m_full['bench_max_drawdown'])),
        ("Sharpe (daily, rf=0)", _num(m_full['sharpe']), _num(m_nodcf['sharpe']), _num(m_full['bench_sharpe'])),
        ("Final equity (Rs.)", f"{m_full['final_equity']:>20,.0f}", f"{m_nodcf['final_equity']:>20,.0f}", "-"),
        ("Trades filled", f"{m_full['trades']:>20,d}", f"{m_nodcf['trades']:>20,d}", "-"),
        ("Win rate", _pct(m_full['win_rate']), _pct(m_nodcf['win_rate']), "-"),
        ("Profit factor", _num(m_full['profit_factor']), _num(m_nodcf['profit_factor']), "-"),
        ("Avg win / trade", _pct(m_full['avg_win_pct']), _pct(m_nodcf['avg_win_pct']), "-"),
        ("Avg loss / trade", _pct(m_full['avg_loss_pct']), _pct(m_nodcf['avg_loss_pct']), "-"),
        ("Exits TP / SL / TIME", f"{m_full['exits']['TP']:>7d}/{m_full['exits']['SL']:<6d}/{m_full['exits']['TIME']:<6d}",
         f"{m_nodcf['exits']['TP']:>7d}/{m_nodcf['exits']['SL']:<6d}/{m_nodcf['exits']['TIME']:<6d}", "-"),
        ("Long trades (PnL Rs.)", f"{m_full['long_trades']:>8,d} ({m_full['long_pnl']:>9,.0f})",
         f"{m_nodcf['long_trades']:>8,d} ({m_nodcf['long_pnl']:>9,.0f})", "-"),
        ("Short trades (PnL Rs.)", f"{m_full['short_trades']:>8,d} ({m_full['short_pnl']:>9,.0f})",
         f"{m_nodcf['short_trades']:>8,d} ({m_nodcf['short_pnl']:>9,.0f})", "-"),
        ("Avg hold (days)", f"{m_full['avg_hold_days']:>20,.1f}", f"{m_nodcf['avg_hold_days']:>20,.1f}", "-"),
        ("Exposure (pos-days)", _pct(m_full['exposure_pct']), _pct(m_nodcf['exposure_pct']), "-"),
    ]
    for name, a, b, c in rows:
        L.append(f"{name:<24}| {a:>20} | {b:>20} | {c:>20}")
    L.append("")
    L.append("PER-YEAR RETURNS")
    L.append("-" * w)
    L.append(f"{'Year':<10}| {'FULL':>14} | {'NO-DCF':>14} | {'Nifty':>14}")
    L.append("-" * w)
    n_years = max(len(m_full['yearly']), len(m_nodcf['yearly']))
    base = int(str(params['start'])[:4])
    for i in range(n_years):
        y = base + i
        a = _pct(m_full['yearly'][i]) if i < len(m_full['yearly']) else "-"
        b = _pct(m_nodcf['yearly'][i]) if i < len(m_nodcf['yearly']) else "-"
        L.append(f"{y:<10}| {a:>14} | {b:>14} | (see chart)")
    L.append("")
    L.append("TOP 5 WINS AND TOP 5 LOSSES (FULL variant)")
    L.append("-" * w)
    ranked = sorted(trades_full, key=lambda t: t['pnl'], reverse=True)
    for label, chunk in (("WIN  ", ranked[:5]), ("LOSS ", sorted(ranked[-5:], key=lambda t: t['pnl']))):
        for t in chunk:
            L.append(f"{label} {t['symbol']:<10}{t['dir']:<6}{t['entry_date']} -> {t['exit_date']}  "
                     f"{t['pnl_pct'] * 100:>7.2f}%  ({t['reason']}, score {t['score']:.0f})")
    L.append("")
    L.append("CONCLUSION")
    L.append("-" * w)
    gap = m_full['total_return'] - m_nodcf['total_return']
    L.append(f"Over {nifty_span:.1f} years the FULL signal turned Rs.{params['capital']:,.0f} into "
             f"Rs.{m_full['final_equity']:,.0f} ({_pct(m_full['total_return'])}, "
             f"CAGR {_pct(m_full['cagr'])}) vs Nifty buy-and-hold {_pct(m_full['bench_total_return'])} "
             f"(CAGR {_pct(m_full['bench_cagr'])}).")
    L.append(f"With the DCF factor removed (no look-ahead), the same engine returned "
             f"{_pct(m_nodcf['total_return'])} (CAGR {_pct(m_nodcf['cagr'])}); the "
             f"{_pct(gap)} gap between FULL and NO-DCF is the maximum credit the current-")
    L.append("day fundamentals snapshot can claim for historical results.")
    L.append(f"Drawdowns: FULL {_pct(m_full['max_drawdown'])}, NO-DCF {_pct(m_nodcf['max_drawdown'])}, "
             f"Nifty {_pct(m_full['bench_max_drawdown'])}. All figures carry the survivorship")
    L.append("and cadence biases listed above; treat them as an upper bound, not a forecast.")
    L.append("=" * w)

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(L) + "\n")


def plot_equity(out_path, eq_full, eq_nodcf, bench_close, capital0):
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    ax.plot(eq_full.index, eq_full.values / capital0 * 100, color='#d62728',
            lw=1.4, label='Strategy FULL (with DCF)')
    ax.plot(eq_nodcf.index, eq_nodcf.values / capital0 * 100, color='#1f77b4',
            lw=1.4, label='Strategy NO-DCF (no look-ahead)')
    ax.plot(bench_close.index, bench_close.values / bench_close.iloc[0] * 100,
            color='#7f7f7f', lw=1.2, ls='--', label='Nifty 50 buy & hold')
    ax.set_title('Walk-forward equity: hybrid strategy vs Nifty 50 buy & hold (start = 100)')
    ax.set_ylabel('Equity (start = 100)')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Walk-forward backtest of the hybrid Nifty 50 strategy.")
    ap.add_argument('--start', default='2016-01-01')
    ap.add_argument('--end', default=None)
    ap.add_argument('--capital', type=float, default=100000)
    ap.add_argument('--cadence', type=int, default=5)
    ap.add_argument('--cost-pct', type=float, default=0.1)
    ap.add_argument('--out', default='backtest_results')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    symbols = pd.read_csv('nifty50_constituents.csv')['Symbol'].tolist()
    stock_data = build_stock_data(symbols)
    print(f"Loaded {len(stock_data)}/{len(symbols)} constituents with data.", flush=True)

    nifty = load_index_frame('data/ohlcv/^NSEI.csv')
    vix_path = 'data/ohlcv/^INDIAVIX.csv'
    vix = load_index_frame(vix_path) if os.path.exists(vix_path) else None

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else nifty.index[-1]
    window_days = list(nifty.index[(nifty.index >= start) & (nifty.index <= end)])
    scan_dates = window_days[::args.cadence]
    day_index = {d: i for i, d in enumerate(window_days)}
    span_years = (window_days[-1] - window_days[0]).days / 365.25
    print(f"Window {window_days[0].date()} -> {window_days[-1].date()} "
          f"({span_years:.1f}y, {len(window_days)} days, {len(scan_dates)} weekly scans)", flush=True)

    info_cache = load_info_cache(symbols)
    events = collect_events(symbols, stock_data, info_cache, nifty, vix,
                            scan_dates, args.out)

    bench_close = nifty['Close'].reindex(window_days).ffill()
    results = {'params': {'start': str(window_days[0].date()), 'end': str(window_days[-1].date()),
                          'capital': args.capital, 'cadence': args.cadence,
                          'cost_pct': args.cost_pct, 'train_window': TRAIN_WINDOW,
                          'scan_dates': len(scan_dates)}}

    sims = {}
    for label, use_dcf in (('FULL', True), ('NO-DCF', False)):
        print(f"Simulating {label} variant...", flush=True)
        trades, eq, open_end = simulate(events, scan_dates, use_dcf, stock_data,
                                        window_days, day_index, args.capital, args.cost_pct)
        m = compute_metrics(trades, eq, bench_close, args.capital, len(window_days))
        m['open_at_end'] = open_end
        sims[label] = {'trades': trades, 'eq': eq, 'metrics': m}
        print(f"  {label}: {m['trades']} trades, total return {_pct(m['total_return'])}, "
              f"CAGR {_pct(m['cagr'])}, maxDD {_pct(m['max_drawdown'])}", flush=True)

    results['FULL'] = {'metrics': sims['FULL']['metrics'], 'trades': sims['FULL']['trades']}
    results['NO-DCF'] = {'metrics': sims['NO-DCF']['metrics'], 'trades': sims['NO-DCF']['trades']}
    results['nifty'] = {'total_return': sims['FULL']['metrics']['bench_total_return'],
                        'cagr': sims['FULL']['metrics']['bench_cagr'],
                        'max_drawdown': sims['FULL']['metrics']['bench_max_drawdown']}

    with open(os.path.join(args.out, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    write_report(os.path.join(args.out, 'BACKTEST_REPORT.txt'), results['params'],
                 sims['FULL']['metrics'], sims['NO-DCF']['metrics'],
                 sims['FULL']['trades'], span_years)
    plot_equity(os.path.join(args.out, 'equity_curve.png'),
                sims['FULL']['eq'], sims['NO-DCF']['eq'], bench_close, args.capital)
    print(f"Done. Results in {args.out}/ (results.json, BACKTEST_REPORT.txt, equity_curve.png)")


if __name__ == '__main__':
    main()
