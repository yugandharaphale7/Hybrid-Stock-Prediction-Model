import pandas as pd
import numpy as np
import os
import json

def run_dcf_from_local(symbol, current_price, folder='data/fundamentals', info_override=None):
    try:
        # SPEED BOOST: Allow passing pre-loaded JSON data to avoid disk reads
        if info_override:
            info = info_override
        else:
            info_path = os.path.join(folder, f"{symbol}_info.json")
            if not os.path.exists(info_path): return None
            with open(info_path, 'r') as f:
                info = json.load(f)
            
        ocf = info.get('operatingCashflow', 0)
        capex = abs(info.get('capitalExpenditures', 0))
        fcf = ocf - capex
        
        shares = info.get('sharesOutstanding', 0)
        debt = info.get('totalDebt', 0)
        cash = info.get('totalCash', 0)
        net_debt = debt - cash
        
        # BRUTAL FIX: If FCF is missing or negative (e.g. Banks like HDFC), DCF is invalid.
        # Do not force bad math. Default to Fair Value.
        if fcf <= 0 or shares <= 0 or current_price <= 0:
            return {'symbol': symbol, 'intrinsic_value': current_price, 'current_price': current_price, 'status': 'Fair Value', 'margin_of_safety': 0.0}
            
        # Dynamic Growth: Cap at 15%, floor at 5% based on basic FCF yield
        fcf_yield = fcf / (current_price * shares)
        growth_rate = np.clip(fcf_yield * 0.5, 0.05, 0.15) 
        
        discount_rate = 0.11 # Indian market standard WACC
        terminal_growth = 0.05 # Indian long-term GDP growth proxy
        projection_years = 5
        
        projected_fcfs = [fcf * (1 + growth_rate)**i for i in range(1, projection_years + 1)]
        terminal_value = (projected_fcfs[-1] * (1 + terminal_growth)) / (discount_rate - terminal_growth)
        
        pv_fcfs = sum([fcf / (1 + discount_rate)**(i+1) for i, fcf in enumerate(projected_fcfs)])
        pv_tv = terminal_value / (1 + discount_rate)**projection_years
        
        equity_value = (pv_fcfs + pv_tv) - net_debt
        intrinsic_value = equity_value / shares
        
        if intrinsic_value <= 0:
            return {'symbol': symbol, 'intrinsic_value': current_price, 'current_price': current_price, 'status': 'Overvalued', 'margin_of_safety': 0.0}

        margin = (intrinsic_value - current_price) / intrinsic_value
        status = "Undervalued" if margin > 0.15 else "Overvalued" if margin < -0.15 else "Fair Value"
        
        return {
            'symbol': symbol,
            'intrinsic_value': round(intrinsic_value, 2),
            'current_price': current_price,
            'status': status,
            'margin_of_safety': round(margin, 4)
        }
    except Exception:
        return None