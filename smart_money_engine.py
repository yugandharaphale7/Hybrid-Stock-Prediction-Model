import pandas as pd
import numpy as np

class SmartMoneyEngine:
    def __init__(self):
        self.lookback = 20

    def analyze(self, df):
        try:
            close = df['Close']
            volume = df['Volume']
            
            delta = close.diff()
            obv = (volume * np.sign(delta)).fillna(0).cumsum()
            obv_sma = obv.rolling(window=self.lookback).mean()
            obv_trend_up = obv.iloc[-1] > obv_sma.iloc[-1]
            
            up_days = delta > 0
            down_days = delta < 0
            
            avg_up_vol = volume.where(up_days, 0).rolling(self.lookback).mean().iloc[-1]
            avg_down_vol = volume.where(down_days, 0).rolling(self.lookback).mean().iloc[-1]
            
            if avg_down_vol == 0: avg_down_vol = 1
            vol_ratio = avg_up_vol / avg_down_vol 
            
            if obv_trend_up and vol_ratio > 1.1:
                return "Accumulation"
            elif not obv_trend_up and vol_ratio < 0.9:
                return "Distribution"
            else:
                return "Neutral"
                
        except Exception:
            return "Neutral"