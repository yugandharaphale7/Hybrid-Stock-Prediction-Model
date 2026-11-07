import sys
import os

# Ensure stdout supports UTF-8 on Windows
try:
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import pandas as pd
import numpy as np
import json
import time # Added for the visual speed timer
from ml_engine import MLEngine
from dcf_engine import run_dcf_from_local
from risk_manager import MarketRegime, RiskManager
from decision_engine import DecisionEngine
from dashboard import BloombergTerminal
from smart_money_engine import SmartMoneyEngine

def main():
    TOTAL_CAPITAL = 100000
    terminal = BloombergTerminal()
    risk_mgr = RiskManager(total_capital=TOTAL_CAPITAL)
    decision_engine = DecisionEngine()
    smart_money_engine = SmartMoneyEngine()
    
    data_dir = 'data/ohlcv'
    if not os.path.exists(data_dir):
        print("Error: 'data/ohlcv' folder not found. Please run import_data.py first.")
        return

    nifty_path = os.path.join(data_dir, "^NSEI.csv")
    vix_path = os.path.join(data_dir, "^INDIAVIX.csv")
    
    if not os.path.exists(nifty_path):
        print("Error: Nifty 50 index data missing.")
        return
        
    # SPEED BOOST: Removed parse_dates for index files too (saves 0.2s)
    nifty_df = pd.read_csv(nifty_path, index_col=0)[['Close']].astype(float)
    vix_df = pd.read_csv(vix_path, index_col=0)[['Close']].astype(float) if os.path.exists(vix_path) else None
    
    # 1. Generate Symbol List
    symbols = [f.replace('.csv', '') for f in os.listdir(data_dir) if f.endswith('.csv') and not f.startswith('^')]
    
    # SPEED BOOST: Pre-load all DCF JSONs into RAM (takes 0.05 seconds)
    dcf_cache = {}
    fund_dir = 'data/fundamentals'
    if os.path.exists(fund_dir):
        for sym in symbols:
            info_path = os.path.join(fund_dir, f"{sym}_info.json")
            if os.path.exists(info_path):
                try:
                    with open(info_path, 'r') as f:
                        dcf_cache[sym] = json.load(f)
                except Exception:
                    pass
    
    # 2. Calculate Base Regime once (VIX and Trend don't change during the loop)
    base_regime = MarketRegime.get_regime(nifty_df, vix_df, 50.0) # 50.0 is a temporary placeholder
    
    signals = []
    indicators = []
    positions = []         # LONG positions
    short_positions = []   # SHORT positions
    recommendations = []
    short_recommendations = []
    total_deployed = 0
    total_short_deployed = 0
    skipped_capital = 0   # Trades dropped because the portfolio ran out of unspent capital
    breadth_data = []      # Track breadth dynamically to save disk read time

    # VISUAL SPEED TIMER START
    print("Initializing scan...", end="", flush=True)
    start_time = time.time()

    # 3. THE OPTIMIZED SINGLE-PASS SCANNING LOOP
    for i, symbol in enumerate(symbols):
        # \r makes it overwrite the current line instead of printing a new line
        print(f"\rScanning stocks... [{i+1}/{len(symbols)}]", end="", flush=True)
        
        file_path = os.path.join(data_dir, f"{symbol}.csv")
        try:
            # SPEED BOOST: Removed parse_dates=True (saves 1.5 seconds total)
            df = pd.read_csv(file_path, index_col=0)[['Close', 'High', 'Low', 'Volume']].astype(float)
        except Exception:
            continue 
            
        # OPTIMIZATION: Calculate Breadth for FREE using the data we just loaded
        if len(df) >= 50:
            is_bullish = 1 if df['Close'].iloc[-1] > df['Close'].rolling(50).mean().iloc[-1] else 0
            breadth_data.append(is_bullish)
            
        if len(df) < 500: continue # Enforce ML minimum data requirement
        
        try:
            current_price = float(df['Close'].iloc[-1])
            if current_price <= 0: continue
            
            # OPTIMIZATION: Update regime with live breadth calculated SO FAR in this loop
            live_breadth = (sum(breadth_data) / len(breadth_data) * 100) if breadth_data else 50
            regime = base_regime.copy()
            regime['breadth'] = live_breadth
            
            ml = MLEngine()
            if not ml.train(df): continue
            ml_res = ml.predict(df)
            
            # Smart Money Analysis
            sm_res = smart_money_engine.analyze(df)
            
            # SPEED BOOST: Pass cached RAM data instead of reading from disk
            dcf_res = run_dcf_from_local(symbol, current_price, info_override=dcf_cache.get(symbol))
            sig = decision_engine.generate_signal(ml_res, dcf_res, regime, sm_res)
            
            # Calculate position with Macro-Trend awareness
            is_trending = regime['trend'] == "TRENDING"
            pos = risk_mgr.calculate_position(current_price, sig['strength'], df, action=sig['action'], is_trending=is_trending)
            
            signals.append({
                'symbol': symbol, 'action': sig['action'], 
                'target': ml_res['predicted_price'], 
                'dcf': dcf_res['status'] if dcf_res else 'N/A',
                'score': sig['strength']
            })
            indicators.append({
                'symbol': symbol, 'rsi': ml_res['rsi'], 
                'dir': ml_res['direction'], 'conf': ml_res['confidence'],
                'sm': sm_res
            })
            
            # --- LONG POSITION LOGIC ---
            # Portfolio-level cap: allocations (longs + shorts) can never exceed total capital
            if sig['action'] == 'BUY' and pos['quantity'] > 0 and pos['allocation'] <= TOTAL_CAPITAL - total_deployed - total_short_deployed:
                total_deployed += pos['allocation']
                positions.append({'symbol': symbol, 'qty': pos['quantity'], 'alloc': int(pos['allocation']), 'entry': current_price, 'type': 'LONG'})
                
                if "DIP" in sig['timing']:
                    exact_trigger = round(current_price - ((current_price - pos['stop_loss']) / 2), 2)
                    trigger_str = f"LIMIT BUY @ ₹{exact_trigger}"
                else:
                    trigger_str = f"MARKET BUY @ ₹{current_price:.2f}"
                
                # Trailing Stop Instructions
                atr_distance = current_price - pos['stop_loss'] 
                trail_activation_price = round(current_price + atr_distance, 2)
                trigger_str += f" | TRAIL SL > ₹{trail_activation_price}"
                
                recommendations.append({
                    'action': sig['action'], 'qty': pos['quantity'], 'symbol': symbol,
                    'price': current_price, 'sl': pos['stop_loss'], 'tp': pos['take_profit'],
                    'trigger': trigger_str, 'reason': sig['reason'], 'score': sig['strength']
                })
                
            # Trade was sized correctly but there is no cash left to take it
            elif sig['action'] == 'BUY' and pos['quantity'] > 0:
                skipped_capital += 1

            # --- SHORT POSITION LOGIC ---
            elif sig['action'] == 'SELL' and pos['quantity'] > 0 and pos['allocation'] <= TOTAL_CAPITAL - total_deployed - total_short_deployed:
                total_short_deployed += pos['allocation']
                short_positions.append({'symbol': symbol, 'qty': pos['quantity'], 'alloc': int(pos['allocation']), 'entry': current_price, 'type': 'SHORT'})
                
                if "BOUNCE" in sig['timing']:
                    exact_trigger = round(current_price + ((pos['stop_loss'] - current_price) / 2), 2)
                    trigger_str = f"LIMIT SELL @ ₹{exact_trigger}"
                else:
                    trigger_str = f"MARKET SELL @ ₹{current_price:.2f}"
                    
                short_recommendations.append({
                    'action': sig['action'], 'qty': pos['quantity'], 'symbol': symbol,
                    'price': current_price, 'sl': pos['stop_loss'], 'tp': pos['take_profit'],
                    'trigger': trigger_str, 'reason': sig['reason'], 'score': sig['strength']
                })
            elif sig['action'] == 'SELL' and pos['quantity'] > 0:
                skipped_capital += 1
        except Exception:
            continue 

    # VISUAL SPEED TIMER END
    elapsed_time = time.time() - start_time
    # Print final time, overwriting the counter line
    print(f"\rScan completed in {elapsed_time:.2f} seconds.                ") 

    # 4. PREPARE DASHBOARD DATA
    undervalued_signals = [s for s in signals if s['dcf'] == 'Undervalued']
    fair_value_signals = [s for s in signals if s['dcf'] == 'Fair Value']
    overvalued_signals = [s for s in signals if s['dcf'] == 'Overvalued']
    
    # Sort them so the strongest signals appear at the top
    undervalued_signals.sort(key=lambda x: x.get('score', 0), reverse=True)
    fair_value_signals.sort(key=lambda x: x.get('score', 0), reverse=True)
    overvalued_signals.sort(key=lambda x: x.get('score', 0), reverse=True)

    tags = [
        ("LIVE" if pd.Timestamp.now().weekday() < 5 else "MARKET CLOSED", terminal.c['s']),
        (regime['volatility'], terminal.c['d'] if "HIGH" in regime['volatility'] else terminal.c['g']),
        (regime['trend'], terminal.c['h'] if regime['trend']=="TRENDING" else terminal.c['w'])
    ]

    # Combine Long and Short recommendations to find the absolute best trade
    all_recs = recommendations + short_recommendations
    all_recs.sort(key=lambda x: x['score'], reverse=True)

    dashboard_data = {
        'tags': tags,
        'vix': regime['vix'],
        'breadth': regime['breadth'],
        'capital': TOTAL_CAPITAL,
        'deployed': int(total_deployed + total_short_deployed),
        'deployed_long': int(total_deployed),
        'deployed_short': int(total_short_deployed),
        # Remaining cash = Total - Money in Longs - Money allocated to Shorts
        'remaining': TOTAL_CAPITAL - (int(total_deployed) + int(total_short_deployed)),
        'undervalued_signals': undervalued_signals,
        'fair_value_signals': fair_value_signals,
        'overvalued_signals': overvalued_signals, # Sends shorts to dashboard
        'indicators': indicators,
        'positions': positions,                   # Long positions
        'short_positions': short_positions,       # Short positions
        # Edge score is now whichever is highest: Best Long or Best Short
        'edge_score': int(all_recs[0]['score'] * 100) if all_recs else 0,
        # Diversification now counts both Long and Short stocks
        'diversification': min(100, (len(positions) + len(short_positions)) * 10), 
        'max_risk': risk_mgr.max_risk_pct,
        'skipped_capital': skipped_capital,
        # Top 10 will now mix the best Longs and best Shorts
        'recs': all_recs[:10] 
    }

    terminal.render(dashboard_data)

if __name__ == "__main__":
    main()