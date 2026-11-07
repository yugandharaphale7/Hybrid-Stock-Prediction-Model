import pandas as pd
import numpy as np

class MarketRegime:
    @staticmethod
    def get_regime(nifty_df, vix_df, pre_calculated_breadth):
        vix_val = 15.0
        if vix_df is not None and not vix_df.empty:
            vix_val = float(vix_df['Close'].iloc[-1])
            
        volatility = "HIGH VOL" if vix_val > 18 else "LOW VOL"
        
        close = nifty_df['Close']
        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200).mean()
        trend = "TRENDING" if sma_50.iloc[-1] > sma_200.iloc[-1] else "RANGING"
        
        return {'volatility': volatility, 'trend': trend, 'vix': vix_val, 'breadth': pre_calculated_breadth}

class RiskManager:
    def __init__(self, total_capital):
        self.total_capital = total_capital
        self.max_risk_pct = 2.0 

    def calculate_position(self, current_price, signal_strength, df_ohlcv, action="BUY", is_trending=False):
        if np.isnan(current_price) or current_price <= 0:
            return {'quantity': 0, 'allocation': 0, 'stop_loss': 0, 'take_profit': 0}
        
        high = df_ohlcv['High']; low = df_ohlcv['Low']; close = df_ohlcv['Close']
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().dropna().iloc[-1]
        
        sl_distance = max(atr * 1.5, current_price * 0.02)
        tp_distance = sl_distance * 2.5 
        
        if action == "BUY":
            stop_loss = current_price - sl_distance
            take_profit = current_price + tp_distance
        elif action == "SELL": # SHORT SELLING
            stop_loss = current_price + sl_distance
            take_profit = current_price - tp_distance
        else:
            return {'quantity': 0, 'allocation': 0, 'stop_loss': 0, 'take_profit': 0}
        
        # Exact math: If SL hits, I lose exactly this much
        max_loss_amount = self.total_capital * (self.max_risk_pct / 100)
        raw_qty = int(max_loss_amount / sl_distance)
        
        # Scale down by signal strength
        final_qty = max(1, int(raw_qty * signal_strength)) if raw_qty > 0 else 0
        
        if final_qty <= 0:
            return {'quantity': 0, 'allocation': 0, 'stop_loss': round(stop_loss, 2), 'take_profit': round(take_profit, 2)}
        
        # NEW: MACRO-TAILORED ASYMMETRIC ALLOCATION
        # Proven in backtest: Unleash capital when the trend supports it
        if is_trending and signal_strength >= 0.6:
            max_allocation = self.total_capital * 0.15 # High conviction in uptrend: 15%
        elif is_trending and signal_strength >= 0.4:
            max_allocation = self.total_capital * 0.10 # Medium conviction in uptrend: 10%
        else:
            max_allocation = self.total_capital * 0.05  # Ranging/Low conviction: 5%
        
        actual_allocation = final_qty * current_price
        
        if actual_allocation > max_allocation:
            final_qty = int(max_allocation / current_price)
            actual_allocation = final_qty * current_price
            
        return {
            'quantity': final_qty,
            'allocation': actual_allocation,
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2)
        }