import yfinance as yf
import pandas as pd
import os
import json
import time

def _is_file_stale(file_path, max_age_days=30):
    """Returns True if the file doesn't exist or is older than max_age_days."""
    if not os.path.exists(file_path):
        return True
    file_age_seconds = time.time() - os.path.getmtime(file_path)
    return (file_age_seconds / 86400) > max_age_days

def fetch_fundamentals(symbol, folder='data/fundamentals'):
    info_path = os.path.join(folder, f"{symbol}_info.json")
    
    # PERFORMANCE BOOST: Skip API call if we downloaded this in the last 30 days
    if not _is_file_stale(info_path, max_age_days=365):
        return None # Returning None safely triggers the "skip" logic in import_data.py

    ticker = f"{symbol}" if symbol.startswith('^') else f"{symbol}.NS"
    stock = yf.Ticker(ticker)
    
    try:
        cashflow = stock.cashflow
        balance_sheet = stock.balance_sheet
        income_stmt = stock.financials
        info = stock.info
        
        # Clean info dict to prevent JSON serialization errors (dates, tuples, etc.)
        clean_info = {}
        for k, v in info.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                clean_info[k] = v
            else:
                clean_info[k] = str(v) # Convert complex types to string

        return {
            'cashflow': cashflow,
            'balance_sheet': balance_sheet,
            'income_stmt': income_stmt,
            'info': clean_info
        }
    except Exception as e:
        print(f"  -> Error fetching fundamentals for {ticker}: {e}")
        return None

def save_fundamentals(fundamentals, symbol, folder='data/fundamentals'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    for key in ['cashflow', 'balance_sheet', 'income_stmt']:
        if fundamentals[key] is not None and not fundamentals[key].empty:
            path = os.path.join(folder, f"{symbol}_{key}.csv")
            fundamentals[key].to_csv(path)
    
    info_path = os.path.join(folder, f"{symbol}_info.json")
    with open(info_path, 'w') as f:
        json.dump(fundamentals['info'], f, indent=4)