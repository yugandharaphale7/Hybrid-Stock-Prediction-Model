import pandas as pd
import os
import time
from data_fetcher import fetch_ohlcv
from fundamental_fetcher import fetch_fundamentals

def bulk_import():
    constituents_path = 'nifty50_constituents.csv'
    if not os.path.exists(constituents_path):
        print("Error: nifty50_constituents.csv not found.")
        return
        
    df = pd.read_csv(constituents_path)
    symbols = df['Symbol'].tolist()
    
    print("=== Starting Smart Bulk Import ===")
    os.makedirs('data/ohlcv', exist_ok=True)
    os.makedirs('data/fundamentals', exist_ok=True)
    
    # Fetch Index and VIX (Saving is now handled automatically inside fetch_ohlcv)
    print("Checking Index Data (^NSEI, ^INDIAVIX)...")
    fetch_ohlcv("^NSEI")
    fetch_ohlcv("^INDIAVIX")
    
    for i, symbol in enumerate(symbols):
        print(f"[{i+1}/{len(symbols)}] {symbol}...", end=" ", flush=True)
        
        downloaded_something = False
        
        # 1. Check OHLCV Prices
        ohlcv_res = fetch_ohlcv(symbol)
        if ohlcv_res is not None:
            print("Prices updated.", end=" ", flush=True)
            downloaded_something = True
            
        # 2. Check Fundamentals
        funda_res = fetch_fundamentals(symbol)
        if funda_res is not None:
            print("Fundamentals updated.", end=" ", flush=True)
            downloaded_something = True
            
        if not downloaded_something:
            print("Up to date.", flush=True)
        else:
            print("", flush=True) # Just a newline
            # ONLY sleep if we actually made an API call to respect Yahoo limits
            time.sleep() 

    print("\n=== Import Complete ===")

if __name__ == "__main__":
    bulk_import()