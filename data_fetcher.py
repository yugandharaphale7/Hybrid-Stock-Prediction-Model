import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

def fetch_ohlcv_incremental(symbol, folder='data/ohlcv'):
    ticker = f"{symbol}" if symbol.startswith('^') else f"{symbol}.NS"
    file_path = os.path.join(folder, f"{symbol}.csv")
    
    # 1. Find out what data we already have
    existing_end_date = None
    if os.path.exists(file_path):
        try:
            # Read without auto-parsing to handle raw data safely
            existing_df = pd.read_csv(file_path, index_col=0)
            if not existing_df.empty:
                raw_date = existing_df.index[-1]
                # Convert whatever format the date is in (int, str) into a Pandas Timestamp
                existing_end_date = pd.to_datetime(raw_date, errors='coerce')
        except Exception:
            pass # If CSV is corrupted, we will just re-download it

    # 2. If we have data up to yesterday, SKIP downloading to save time
    yesterday = datetime.now() - timedelta(days=1)
    if pd.notna(existing_end_date) and existing_end_date.date() >= yesterday.date():
        return None # Signals to import_data.py that no download is needed

    # 3. If we need data, start from the day AFTER our last recorded date
    start_date = existing_end_date + timedelta(days=1) if pd.notna(existing_end_date) else datetime.now() - timedelta(days=25*365)
    end_date = datetime.now()

    try:
        # Fetch ONLY the missing chunk of data
        new_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if new_data.empty:
            return None
            
        if isinstance(new_data.columns, pd.MultiIndex):
            new_data.columns = new_data.columns.droplevel(1)
            
        # 4. Append new data to old data (if old data exists)
        if pd.notna(existing_end_date) and os.path.exists(file_path):
            old_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            combined_data = pd.concat([old_data, new_data])
            # Remove duplicate dates just in case of overlap
            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
            final_data = combined_data
        else:
            final_data = new_data
            
        # Ensure folder exists and save
        if not os.path.exists(folder):
            os.makedirs(folder)
        final_data.to_csv(file_path)
        return ticker
        
    except Exception as e:
        print(f"  -> Error fetching {ticker}: {e}")
        return None


# Backward compatibility wrappers so import_data.py doesn't break
def fetch_ohlcv(symbol, years=25):
    return fetch_ohlcv_incremental(symbol)

def save_data(data, symbol, folder='data/ohlcv'):
    pass # Saving is now handled automatically inside fetch_ohlcv_incremental