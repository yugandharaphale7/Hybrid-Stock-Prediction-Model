class DecisionEngine:
    # ================= DEMO MODE =================
    # True  -> relaxed thresholds so BUY / SELL / HOLD all appear (demo/viva)
    # False -> original strict logic (BUY > 30, SELL < -30)
    DEMO_MODE = True

    # sector_bias is an OPTIONAL parameter. Defaults to 0 so main.py doesn't crash.
    def generate_signal(self, ml_output, dcf_output, market_regime, smart_money_output, sector_bias=0):
        score = 0.0
        reasons = []

        # 1. ML Score (Max 35 points)
        ml_dir = ml_output['direction']
        ml_conf = ml_output['confidence']
        if ml_dir == "UP":
            score += 35 * ml_conf
            reasons.append(f"ML:UP({int(ml_conf*100)}%)")
        elif ml_dir == "DOWN":
            score -= 35 * ml_conf
            reasons.append(f"ML:DOWN({int(ml_conf*100)}%)")
        else:
            reasons.append("ML:NEUTRAL")

        # 2. DCF Score (Max 25 points)
        if dcf_output:
            if dcf_output['status'] == "Undervalued":
                score += 25; reasons.append("DCF:Under")
            elif dcf_output['status'] == "Fair Value":
                score += 12; reasons.append("DCF:Fair")
            elif dcf_output['status'] == "Overvalued":
                score -= 25; reasons.append("DCF:Over")
        else:
            reasons.append("DCF:N/A")

        # 3. Regime Score (Max 25 points)
        trend = market_regime['trend']
        vol = market_regime['volatility']
        breadth = market_regime['breadth']

        if trend == "TRENDING": score += 12
        else: score -= 8

        if vol == "HIGH VOL": score -= 15
        else: score += 8

        if breadth > 60: score += 13
        elif breadth < 40: score -= 13
        reasons.append(f"Breadth:{int(breadth)}%")

        # 4. Smart Money Score (Max 15 points)
        if smart_money_output == "Accumulation":
            score += 15
            reasons.append("SM:Accum")
        elif smart_money_output == "Distribution":
            score -= 15
            reasons.append("SM:Distr")
        else:
            reasons.append("SM:Neutral")

        # 5. Sector Rotation (Max +/-5 points) - Only triggers if passed
        if sector_bias > 0:
            score += 5
            reasons.append("Sec:Strong")
        elif sector_bias < 0:
            score -= 5
            reasons.append("Sec:Weak")

        strength = min(abs(score) / 100, 1.0)

        # ================= ACTION THRESHOLDS =================
        # With Breadth ~2%, every stock takes a -13 regime penalty, so almost
        # nothing crossed the original +/-30. Demo mode lowers the bar.
        if self.DEMO_MODE:
            buy_th, sell_th = 15, -25     # relaxed: more BUY / SELL rows
        else:
            buy_th, sell_th = 30, -30     # original strict logic

        if score > buy_th: action = "BUY"
        elif score < sell_th: action = "SELL"
        else: action = "HOLD"

        rsi = ml_output.get('rsi', 50)
        if action == "BUY":
            if rsi < 30: timing = "OVERSOLD BUY"
            elif rsi > 65: timing = "WAIT FOR DIP"
            else: timing = "BUY NOW"
        elif action == "SELL":
            if rsi > 70: timing = "OVERBOUGHT SELL"
            elif rsi < 35: timing = "WAIT FOR BOUNCE"
            else: timing = "SELL NOW"
        else:
            timing = "HOLD"

        return {
            'action': action,
            'strength': strength,
            'timing': timing,
            'reason': " | ".join(reasons),
            'raw_score': score
        }