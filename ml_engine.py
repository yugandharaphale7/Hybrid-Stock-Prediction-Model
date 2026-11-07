import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class MLEngine:
    def __init__(self):
        # SPEED BOOST: Reduced trees from 150 to 50. Added max_samples for faster training.
        self.model = RandomForestClassifier(
            n_estimators=50, 
            random_state=42, 
            n_jobs=-1, 
            max_depth=6, 
            class_weight='balanced',
            max_samples=0.5 # Uses 50% of data per tree, drastically cuts CPU time
        )
        self.scaler = StandardScaler()
        self.feature_names = ['Mom_20', 'Mom_50', 'RSI', 'MACD_Norm', 'Volatility', 'Vol_Change']

    def _generate_features(self, df):
        # SPEED BOOST: Stopped using df.copy() to save memory allocation time
        close = df['Close']
        volume = df['Volume']
        
        sma_20 = close.rolling(window=20).mean()
        sma_50 = close.rolling(window=50).mean()
        mom_20 = (close - sma_20) / sma_20
        mom_50 = (close - sma_50) / sma_50
        
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd_norm = ((ema_12 - ema_26) / close) * 100 
        
        volatility = close.pct_change().rolling(window=20).std()
        vol_change = volume.pct_change().rolling(5).mean()
        
        # Return as a dataframe to prevent downstream crashes
        return pd.DataFrame({
            'Mom_20': mom_20, 'Mom_50': mom_50, 'RSI': rsi,
            'MACD_Norm': macd_norm, 'Volatility': volatility, 'Vol_Change': vol_change,
            'Close': close
        })

    def train(self, df):
        df_feat = self._generate_features(df)
        
        future_return = df_feat['Close'].shift(-5).pct_change(5)
        df_feat['Target'] = (future_return > 0.005).astype(int)
        
        df_clean = df_feat.dropna(subset=self.feature_names + ['Target'])
        
        if len(df_clean) < 500: 
            return False
        
        X = df_clean[self.feature_names]
        y = df_clean['Target']
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
        return True

    def predict(self, df):
        df_feat = self._generate_features(df)
        df_clean = df_feat.dropna(subset=self.feature_names)
        
        if df_clean.empty:
            return {'predicted_price': 0, 'direction': 'HOLD', 'confidence': 0.5, 'rsi': 50}
            
        last_row = df_clean[self.feature_names].iloc[-1:]
        last_row_scaled = self.scaler.transform(last_row) 
        
        proba = self.model.predict_proba(last_row_scaled)[0]
        prob_up = proba[1] if len(proba) > 1 else 0.5
        
        current_price = df_clean['Close'].iloc[-1]
        rsi = df_clean['RSI'].iloc[-1]
        
        direction = "UP" if prob_up > 0.5 else "DOWN"
        confidence = abs(prob_up - 0.5) * 2
            
        return {
            'predicted_price': round(current_price * (1 + (0.005 if direction == "UP" else -0.005)), 2),
            'direction': direction,
            'confidence': round(confidence, 2),
            'rsi': round(rsi, 2)
        }