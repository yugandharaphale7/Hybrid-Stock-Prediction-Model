import pandas as pd
import numpy as np
import os
from dcf_engine import run_dcf_from_local
from decision_engine import DecisionEngine
from smart_money_engine import SmartMoneyEngine
from ml_engine import MLEngine
from risk_manager import MarketRegime

class ForwardPredictor:
    def __init__(self):
        # Use the exact same engines as main.py for perfect sync
        self.decision_engine = DecisionEngine()
        self.sm_engine = SmartMoneyEngine()

    def _calculate_indicators(self, df):
        close = df['Close']; volume = df['Volume']; high = df['High']; low = df['Low']
        sma_20 = close.rolling(window=20).mean()
        sma_50 = close.rolling(window=50).mean()
        sma_200 = close.rolling(window=200).mean()
        df['Mom_20'] = (close - sma_20) / sma_20
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['Regime_Trending'] = sma_50 > sma_200 
        return df

    def scan_and_predict(self, symbols):
        predictions = []

        # Real macro regime, same inputs as main.py: index trend + VIX + breadth.
        # Breadth is computed for the full universe in one pass (main.py does it
        # incrementally while scanning; here every stock gets the same final value).
        nifty_path = os.path.join('data/ohlcv', '^NSEI.csv')
        if not os.path.exists(nifty_path):
            print("Error: Nifty 50 index data missing. Run import_data.py first.")
            return predictions
        nifty_df = pd.read_csv(nifty_path, index_col=0)[['Close']].astype(float)
        vix_path = os.path.join('data/ohlcv', '^INDIAVIX.csv')
        vix_df = pd.read_csv(vix_path, index_col=0)[['Close']].astype(float) if os.path.exists(vix_path) else None

        breadth_data = []
        for symbol in symbols:
            file_path = os.path.join('data/ohlcv', f"{symbol}.csv")
            if not os.path.exists(file_path): continue
            try:
                bdf = pd.read_csv(file_path, index_col=0)[['Close']].astype(float)
            except Exception:
                continue
            if len(bdf) >= 50:
                breadth_data.append(1 if bdf['Close'].iloc[-1] > bdf['Close'].rolling(50).mean().iloc[-1] else 0)
        breadth = (sum(breadth_data) / len(breadth_data) * 100) if breadth_data else 50.0
        market_regime = MarketRegime.get_regime(nifty_df, vix_df, breadth)

        for symbol in symbols:
            file_path = os.path.join('data/ohlcv', f"{symbol}.csv")
            if not os.path.exists(file_path): continue

            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True, date_format='%Y-%m-%d')
                df = self._calculate_indicators(df)
                
                # Drop NaNs to prevent math errors on the last row
                df = df.dropna() 
                if df.empty: continue
                
                last_row = df.iloc[-1]
                current_price = last_row['Close']
                
                # 1. Get DCF Status
                dcf_res = run_dcf_from_local(symbol, current_price)
                dcf_status = dcf_res['status'] if dcf_res else "Fair Value"
                
                # 2. Real ML engine, exactly as main.py uses it (trained on full
                #    history up to today). Skips the stock if it fails the 500-bar gate.
                ml = MLEngine()
                if not ml.train(df):
                    continue
                ml_output = ml.predict(df)
                
                # 3. Build DCF Dict
                dcf_output = {'status': dcf_status}
                
                # 4. Get Real Smart Money Output
                sm_output = self.sm_engine.analyze(df)
                
                # RUN THROUGH OFFICIAL ENGINE
                sig = self.decision_engine.generate_signal(ml_output, dcf_output, market_regime, sm_output)
                score = sig.get('raw_score', 0)
                
                # 5-DAY EXTRAPOLATION MATH
                if sig['action'] == "BUY": 
                    move_multiplier = (score - 30) / 100 
                    predicted_price = round(current_price + (current_price * move_multiplier * 0.02), 2)
                    action = "BUY"
                elif sig['action'] == "SELL": 
                    move_multiplier = (-score - 30) / 100
                    predicted_price = round(current_price - (current_price * move_multiplier * 0.02), 2)
                    action = "SELL"
                else:
                    predicted_price = round(current_price, 2)
                    action = "HOLD"
                    
                predictions.append({
                    'symbol': symbol, 
                    'current': current_price, 
                    'predicted': predicted_price,
                    'action': action, 
                    'score': score, 
                    'dcf_status': dcf_status
                })
            except Exception:
                pass # Silent skip to keep terminal clean

        return predictions


def safe_float(val):
    if pd.isna(val) or val is None: return "     N/A"
    try: return f"{float(val):>9.2f}"
    except: return "     N/A"


def main():
    print("Generating 5-Day Forward Predictions...")
    print("="*90)
    
    constituents_path = 'nifty50_constituents.csv'
    if not os.path.exists(constituents_path):
        print("Error: nifty50_constituents.csv not found."); return
        
    symbols = pd.read_csv(constituents_path)['Symbol'].tolist()
    
    predictor = ForwardPredictor()
    all_predictions = predictor.scan_and_predict(symbols)
    
    # Sort by strongest absolute edge
    all_predictions.sort(key=lambda x: abs(x['score']), reverse=True)
    
    # Optional: Limit to top 15 to keep the terminal clean
    top_predictions = all_predictions[:15]

    # PRINT PREDICTIONS
    print(f"{'Symbol':<10} | {'Current':>10} | {'5-Day Pred':<10} | {'Action':<6} | {'Edge':<5} | {'Logic'}")
    print("-"*90)
    
    for f in top_predictions:
        edge = round(abs(f['score']), 0)
        if f['action'] == 'BUY': logic = "Value/Momentum Bullish"
        elif f['action'] == 'SELL': logic = "Overvalued/Distribution"
        else: logic = "No Clear Edge"
            
        print(f"{f['symbol']:<10} | {safe_float(f['current'])} | {safe_float(f['predicted'])} | {f['action']:<6} | {edge:<5} | {logic}")
        
    print("="*90)

if __name__ == "__main__":
    main()