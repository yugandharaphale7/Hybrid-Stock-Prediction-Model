import os
import sys
import numpy as np

# Ensure terminal supports UTF-8 on Windows
try:
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# Enable ANSI escape code processing on Windows cmd / PowerShell
if os.name == 'nt':
    os.system('')

class BloombergTerminal:
    def __init__(self):
        self.c = {
            'bg': '\033[48;2;13;17;23m',
            'p': '\033[38;2;248;248;242m',
            'h': '\033[38;2;0;255;255m',
            's': '\033[38;2;0;255;0m',
            'd': '\033[38;2;255;0;0m',
            'w': '\033[38;2;255;255;0m',
            'a': '\033[38;2;255;0;255m',
            'g': '\033[38;2;211;211;211m',
            'r': '\033[0m'
        }

    def clear(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print(self.c['bg'], end="")

    def box(self, lines, color=None):
        col = color if color else self.c['h']
        width = 40
        print(f"{col}╔{'═'*width}╗{self.c['r']}")
        for line in lines:
            safe_line = str(line)
            pad_len = width - len(safe_line)
            if pad_len < 0: safe_line = safe_line[:width]
            print(f"{col}║{self.c['p']}{safe_line}{' ' * pad_len}{col}║{self.c['r']}")
        print(f"{col}╚{'═'*width}╝{self.c['r']}")

    def progress(self, label, pct):
        filled = int(pct / 5)
        bar = "█" * filled + "░" * (20 - filled)
        print(f"{self.c['p']}{label:15} {self.c['a']}{bar}{self.c['p']} {pct}%{self.c['r']}")

    def _print_cols(self, items, format_func, cols=3):
        if not items: return
        chunk_size = (len(items) + cols - 1) // cols
        for i in range(chunk_size):
            row_parts = []
            for j in range(cols):
                idx = i + (j * chunk_size)
                if idx < len(items):
                    row_parts.append(format_func(items[idx]))
                else:
                    row_parts.append("")
            formatted_row = " | ".join(part.ljust(44) for part in row_parts)
            print(self.c['p'] + formatted_row + self.c['r'])

    def render(self, data):
        self.clear()
        
        print(f"{self.c['h']}══════════════════════════════════════════════{self.c['r']}")
        print(f"{self.c['h']} Hybrid Stock Prediction Model{self.c['r']}")
        print(f"{self.c['h']}══════════════════════════════════════════════{self.c['r']}\n")

        print(f"{self.c['a']} Market Status{self.c['r']}")
        print(f"{self.c['h']}──────────────────────────────────────────────{self.c['r']}")
        for tag, col in data['tags']:
            print(f"{col}[{tag}]{self.c['r']} ", end="")
        print(f"\n{self.c['g']}VIX: {data['vix']} | Breadth: {data['breadth']}%{self.c['r']}\n")

        print(f"{self.c['a']} Portfolio Summary{self.c['r']}")
        print(f"{self.c['h']}──────────────────────────────────────────────{self.c['r']}")
        self.box([
            f"Capital        ₹{data['capital']:>12,}",
            f"Deployed       ₹{data['deployed']:>12,}",
            f"  Long         ₹{data.get('deployed_long', 0):>12,}",
            f"  Short        ₹{data.get('deployed_short', 0):>12,}",
            f"Remaining      ₹{data['remaining']:>12,}"
        ])

        def sig_fmt(sig):
            acol = self.c['s'] if sig['action']=='BUY' else self.c['d'] if sig['action']=='SELL' else self.c['w']
            dcf_col = self.c['s'] if sig['dcf']=='Undervalued' else self.c['d'] if sig['dcf']=='Overvalued' else self.c['g']
            
            # Label based on strategy type
            if sig['dcf'] == 'Undervalued':
                target_label = f"TRAIL:{sig['target']:>6.0f}"
            else:
                target_label = f"5D:{sig['target']:>7.0f}"
                
            return f"{self.c['p']}{sig['symbol']:<10}{self.c['r']}|{acol}{sig['action']:<5}{self.c['r']}|{self.c['g']}{target_label}{self.c['r']}|{dcf_col}{sig['dcf']:<12}{self.c['r']}"
        # LIST 1: UNDERVALUED (Value Plays)
        print(f"\n{self.c['s']} Signal Engine: Undervalued (Value Play){self.c['r']}")
        print(f"{self.c['h']}──────────────────────────────────────────────{self.c['r']}")
        if data['undervalued_signals']:
            self._print_cols(data['undervalued_signals'], sig_fmt, cols=3)
        else:
            print(f"{self.c['g']}No undervalued setups currently active.{self.c['r']}")

        # LIST 2: FAIR VALUE (Momentum Plays)
        print(f"\n{self.c['w']} Signal Engine: Fair Value (Momentum Play){self.c['r']}")
        print(f"{self.c['h']}──────────────────────────────────────────────{self.c['r']}")
        if data['fair_value_signals']:
            self._print_cols(data['fair_value_signals'], sig_fmt, cols=3)
        else:
            print(f"{self.c['g']}No fair value momentum setups currently active.{self.c['r']}")

        # LIST 3: OVERVALUED (Short Plays)
        print(f"\n{self.c['d']} Signal Engine: Overvalued (Short Play){self.c['r']}")
        print(f"{self.c['h']}──────────────────────────────────────────────{self.c['r']}")
        if data['overvalued_signals']:
            self._print_cols(data['overvalued_signals'], sig_fmt, cols=3)
        else:
            print(f"{self.c['g']}No overvalued short setups currently active.{self.c['r']}")

        # INDICATOR BREAKDOWN IN 3 COLUMNS
        print(f"\n{self.c['a']} Indicator Breakdown{self.c['r']}")
        print(f"{self.c['h']}──────────────────────────────────────────────{self.c['r']}")
        
        def ind_fmt(ind):
            sm = ind.get('sm', 'N/A')[:4].upper() # Get first 4 letters: ACCU, DIST, NEUT
            if sm == "ACCU": sm_col = self.c['s'] # Green
            elif sm == "DIST": sm_col = self.c['d'] # Red
            else: sm_col = self.c['g'] # Gray
            
            # Formatted tightly to fit the 3-column grid perfectly
            return f"{self.c['p']}{ind['symbol']:<9}{self.c['r']}|RSI:{ind['rsi']:>4.0f}|ML:{ind['dir']:<4}|SM:{sm_col}{sm:<4}{self.c['r']}|C:{int(ind['conf']*100):>2}%"
            
        self._print_cols(data['indicators'], ind_fmt, cols=3)

        print(f"\n{self.c['a']} Position Details{self.c['r']}")
        print(f"{self.c['h']}──────────────────────────────────────────────{self.c['r']}")
        if data['positions'] or data.get('short_positions'):
            print(f"{self.c['p']}{'Symbol':<10} | {'Type':<5} | {'Qty':<5} | {'Alloc':<10} | {'Entry':<10}{self.c['r']}")
            
            # Print Long Positions in Green
            for pos in data['positions']:
                print(f"{self.c['s']}{pos['symbol']:<10}{self.c['r']} | {self.c['s']}LONG {self.c['r']}| {pos['qty']:<5} | ₹{pos['alloc']:>8,} | ₹{pos['entry']:>8.2f}")
            
            # Print Short Positions in Red
            for pos in data.get('short_positions', []):
                print(f"{self.c['d']}{pos['symbol']:<10}{self.c['r']} | {self.c['d']}SHORT{self.c['r']}| {pos['qty']:<5} | ₹{pos['alloc']:>8,} | ₹{pos['entry']:>8.2f}")
        else:
            print(f"{self.c['w']}No positions met strict risk thresholds.{self.c['r']}")
        if data.get('skipped_capital'):
            print(f"{self.c['w']}Skipped {data['skipped_capital']} trade(s): not enough unspent capital left.{self.c['r']}")

        print(f"\n{self.c['a']} Risk Management{self.c['r']}")
        print(f"{self.c['h']}──────────────────────────────────────────────{self.c['r']}")
        self.progress("Edge Score", data['edge_score'])
        print(f"{self.c['g']}Diversification: {data['diversification']}% | Max Risk/Trade: {data['max_risk']}%{self.c['r']}")

        print(f"\n{self.c['a']} Top 10 Trade Recommendations{self.c['r']}")
        print(f"{self.c['h']}──────────────────────────────────────────────{self.c['r']}")
        recs = data['recs']
        if recs:
            # Updated Header for Trailing Stop Logic
            print(f"{self.c['g']}#   ACTION  QTY  SYMBOL      ENTRY TRIGGER & TRAIL RULES                 SL{self.c['r']}")
            for i, rec in enumerate(recs):
                # Color logic
                if rec['action'] == 'BUY':
                    acol = self.c['s'] # Green
                else: 
                    acol = self.c['d'] # Red
                
                # Get the trigger string (now includes trailing instructions from main.py)
                trigger_txt = rec['trigger']
                
                # Print the row (Removed TP, widened trigger column to 45 chars)
                line = f"{i+1:<2}  {acol}{rec['action']:<6}{self.c['r']} {rec['qty']:<4} {rec['symbol']:<10} {self.c['p']}{trigger_txt:<45}{self.c['r']} {self.c['d']}₹{rec['sl']:>7.2f}{self.c['r']}"
                print(line)
            
            # Print the reasons cleanly below
            print(f"\n{self.c['h']}──────────────────────────────────────────────{self.c['r']}")
            for i, rec in enumerate(recs):
                print(f"{self.c['p']}{i+1:<2}. {rec['symbol']:<10} | {self.c['g']}{rec['reason']}{self.c['r']}")
        else:
            print(f"{self.c['w']}No high-probability setups found. Capital protected.{self.c['r']}")
            
        print(f"\n{self.c['h']}══════════════════════════════════════════════{self.c['r']}")