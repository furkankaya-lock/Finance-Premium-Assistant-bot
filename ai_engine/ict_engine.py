"""
ai_engine/ict_engine.py
=======================
ICT (Inner Circle Trader) Analysis Engine
Concepts: Order Blocks · Fair Value Gaps · Liquidity Levels
          Market Structure · Optimal Trade Entry · Swing Points
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("CryptoBot.ICT")


# ─────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class OrderBlock:
    direction: str          # 'bullish' | 'bearish'
    top: float
    bottom: float
    midpoint: float
    index: int
    strength: float         # 0.0 – 1.0
    mitigated: bool = False
    candle_count: int = 0

@dataclass
class FairValueGap:
    direction: str          # 'bullish' | 'bearish'
    top: float
    bottom: float
    size_pct: float
    index: int
    filled: bool = False
    fill_pct: float = 0.0

@dataclass
class LiquidityLevel:
    level_type: str         # 'buy_side' | 'sell_side' | 'equal_highs' | 'equal_lows'
    price: float
    strength: float         # swing count at this level
    swept: bool = False

@dataclass
class SwingPoint:
    direction: str          # 'high' | 'low'
    price: float
    index: int
    confirmed: bool = False

@dataclass
class MarketStructure:
    trend: str              # 'bullish' | 'bearish' | 'ranging'
    last_bos: Optional[float] = None     # Break of Structure price
    last_choch: Optional[float] = None  # Change of Character price
    higher_highs: list = field(default_factory=list)
    higher_lows: list  = field(default_factory=list)
    lower_highs: list  = field(default_factory=list)
    lower_lows: list   = field(default_factory=list)

@dataclass
class ICTSignal:
    action: str             # 'LONG' | 'SHORT' | 'WAIT'
    confidence: float       # 0.0 – 1.0
    entry_price: float
    ote_zone: tuple         # (low, high) Optimal Trade Entry
    reasons: list
    order_block: Optional[OrderBlock] = None
    fvg: Optional[FairValueGap] = None
    sl_price: Optional[float] = None
    tp1_price: Optional[float] = None
    tp2_price: Optional[float] = None
    tp3_price: Optional[float] = None
    liquidity_target: Optional[float] = None


# ─────────────────────────────────────────────────────────────
# ICT ENGINE
# ─────────────────────────────────────────────────────────────

class ICTEngine:
    """
    Full ICT methodology implementation for crypto markets.
    Works on OHLCV DataFrame with columns: open, high, low, close, volume
    """

    def __init__(self,
                 swing_lookback: int = 10,
                 ob_strength_threshold: float = 0.6,
                 fvg_min_size_pct: float = 0.001,
                 liquidity_tolerance: float = 0.002):
        self.swing_lookback        = swing_lookback
        self.ob_strength_threshold = ob_strength_threshold
        self.fvg_min_size          = fvg_min_size_pct
        self.liq_tolerance         = liquidity_tolerance

    # ── MASTER ANALYSIS ───────────────────────────────────

    def analyze(self, df: pd.DataFrame, current_price: float) -> ICTSignal:
        """
        Run full ICT analysis and return a trade signal.
        """
        if len(df) < 50:
            return ICTSignal(action='WAIT', confidence=0.0,
                             entry_price=current_price,
                             ote_zone=(0, 0), reasons=['Insufficient data'])

        df = df.copy()
        swings     = self._find_swings(df)
        structure  = self._analyze_market_structure(df, swings)
        obs        = self._find_order_blocks(df, swings)
        fvgs       = self._find_fair_value_gaps(df)
        liquidity  = self._find_liquidity_levels(df, swings)
        ote_zone   = self._calculate_ote(df, swings, structure)

        signal = self._generate_signal(
            df, current_price, structure, obs, fvgs, liquidity, ote_zone
        )

        log.info(
            f"[ICT] Trend:{structure.trend} | "
            f"OBs:{len(obs)} | FVGs:{len(fvgs)} | "
            f"Action:{signal.action} ({signal.confidence:.0%})"
        )
        return signal

    # ── SWING POINTS ──────────────────────────────────────

    def _find_swings(self, df: pd.DataFrame) -> list[SwingPoint]:
        swings = []
        lb = self.swing_lookback
        highs = df['high'].values
        lows  = df['low'].values
        n = len(df)

        for i in range(lb, n - lb):
            # Swing High
            if highs[i] == max(highs[i-lb:i+lb+1]):
                swings.append(SwingPoint(
                    direction='high', price=highs[i],
                    index=i, confirmed=True
                ))
            # Swing Low
            if lows[i] == min(lows[i-lb:i+lb+1]):
                swings.append(SwingPoint(
                    direction='low', price=lows[i],
                    index=i, confirmed=True
                ))

        return sorted(swings, key=lambda x: x.index)

    # ── MARKET STRUCTURE ──────────────────────────────────

    def _analyze_market_structure(self, df: pd.DataFrame,
                                  swings: list[SwingPoint]) -> MarketStructure:
        swing_highs = [s for s in swings if s.direction == 'high']
        swing_lows  = [s for s in swings if s.direction == 'low']

        ms = MarketStructure(trend='ranging')

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return ms

        # Last 4 swing points
        recent_h = swing_highs[-4:]
        recent_l = swing_lows[-4:]

        hh = all(recent_h[i].price < recent_h[i+1].price for i in range(len(recent_h)-1))
        hl = all(recent_l[i].price < recent_l[i+1].price for i in range(len(recent_l)-1))
        lh = all(recent_h[i].price > recent_h[i+1].price for i in range(len(recent_h)-1))
        ll = all(recent_l[i].price > recent_l[i+1].price for i in range(len(recent_l)-1))

        if hh and hl:
            ms.trend = 'bullish'
            ms.higher_highs = [s.price for s in recent_h]
            ms.higher_lows  = [s.price for s in recent_l]
        elif lh and ll:
            ms.trend = 'bearish'
            ms.lower_highs = [s.price for s in recent_h]
            ms.lower_lows  = [s.price for s in recent_l]
        else:
            ms.trend = 'ranging'

        # Break of Structure (BOS)
        if len(swing_highs) >= 2:
            prev_high = swing_highs[-2].price
            last_close = df['close'].iloc[-1]
            if ms.trend == 'bullish' and last_close > prev_high:
                ms.last_bos = prev_high
            elif ms.trend == 'bearish' and last_close < swing_lows[-2].price:
                ms.last_bos = swing_lows[-2].price

        # Change of Character (CHoCH)
        if ms.trend == 'bullish' and len(swing_lows) >= 2:
            if swing_lows[-1].price < swing_lows[-2].price:
                ms.last_choch = swing_lows[-1].price
        elif ms.trend == 'bearish' and len(swing_highs) >= 2:
            if swing_highs[-1].price > swing_highs[-2].price:
                ms.last_choch = swing_highs[-1].price

        return ms

    # ── ORDER BLOCKS ──────────────────────────────────────

    def _find_order_blocks(self, df: pd.DataFrame,
                           swings: list[SwingPoint]) -> list[OrderBlock]:
        obs = []
        closes = df['close'].values
        opens  = df['open'].values
        highs  = df['high'].values
        lows   = df['low'].values
        vols   = df['volume'].values
        avg_vol = np.mean(vols)
        n = len(df)

        for i in range(2, n - 3):
            # Bullish Order Block: bearish candle before strong up move
            if closes[i] < opens[i]:  # bearish candle
                # Next 3 candles move up strongly
                up_move = (closes[i+1] - closes[i]) / closes[i]
                if up_move > 0.003 and vols[i+1] > avg_vol * 1.3:
                    strength = min(1.0, up_move * 100 + vols[i+1]/avg_vol * 0.2)
                    if strength >= self.ob_strength_threshold:
                        obs.append(OrderBlock(
                            direction='bullish',
                            top=opens[i], bottom=closes[i],
                            midpoint=(opens[i]+closes[i])/2,
                            index=i, strength=round(strength, 3),
                            candle_count=i
                        ))

            # Bearish Order Block: bullish candle before strong down move
            if closes[i] > opens[i]:  # bullish candle
                down_move = (closes[i] - closes[i+1]) / closes[i]
                if down_move > 0.003 and vols[i+1] > avg_vol * 1.3:
                    strength = min(1.0, down_move * 100 + vols[i+1]/avg_vol * 0.2)
                    if strength >= self.ob_strength_threshold:
                        obs.append(OrderBlock(
                            direction='bearish',
                            top=closes[i], bottom=opens[i],
                            midpoint=(opens[i]+closes[i])/2,
                            index=i, strength=round(strength, 3),
                            candle_count=i
                        ))

        current_price = closes[-1]
        # Mark mitigated (price has already passed through)
        for ob in obs:
            if ob.direction == 'bullish' and current_price < ob.bottom:
                ob.mitigated = True
            elif ob.direction == 'bearish' and current_price > ob.top:
                ob.mitigated = True

        # Return only recent unmitigated OBs
        active = [ob for ob in obs if not ob.mitigated]
        return sorted(active, key=lambda x: x.strength, reverse=True)[:10]

    # ── FAIR VALUE GAPS ───────────────────────────────────

    def _find_fair_value_gaps(self, df: pd.DataFrame) -> list[FairValueGap]:
        fvgs = []
        highs  = df['high'].values
        lows   = df['low'].values
        closes = df['close'].values
        n = len(df)

        for i in range(1, n - 1):
            # Bullish FVG: gap between candle[i-1] high and candle[i+1] low
            gap_top    = lows[i+1]
            gap_bottom = highs[i-1]
            if gap_top > gap_bottom:
                size_pct = (gap_top - gap_bottom) / closes[i]
                if size_pct >= self.fvg_min_size:
                    fvg = FairValueGap(
                        direction='bullish',
                        top=gap_top, bottom=gap_bottom,
                        size_pct=round(size_pct, 5), index=i
                    )
                    # Check if filled
                    if closes[-1] >= gap_bottom:
                        fill = min(1.0, (closes[-1]-gap_bottom)/(gap_top-gap_bottom))
                        fvg.fill_pct = round(fill, 3)
                        fvg.filled = fill >= 0.9
                    fvgs.append(fvg)

            # Bearish FVG
            gap_top2    = lows[i-1]
            gap_bottom2 = highs[i+1]
            if gap_top2 > gap_bottom2:
                size_pct = (gap_top2 - gap_bottom2) / closes[i]
                if size_pct >= self.fvg_min_size:
                    fvg = FairValueGap(
                        direction='bearish',
                        top=gap_top2, bottom=gap_bottom2,
                        size_pct=round(size_pct, 5), index=i
                    )
                    if closes[-1] <= gap_top2:
                        fill = min(1.0, (gap_top2-closes[-1])/(gap_top2-gap_bottom2))
                        fvg.fill_pct = round(fill, 3)
                        fvg.filled = fill >= 0.9
                    fvgs.append(fvg)

        active = [f for f in fvgs if not f.filled]
        return sorted(active, key=lambda x: x.size_pct, reverse=True)[-20:]

    # ── LIQUIDITY LEVELS ──────────────────────────────────

    def _find_liquidity_levels(self, df: pd.DataFrame,
                               swings: list[SwingPoint]) -> list[LiquidityLevel]:
        levels = []
        tol    = self.liq_tolerance

        swing_highs = [s for s in swings if s.direction == 'high']
        swing_lows  = [s for s in swings if s.direction == 'low']

        # Equal Highs (sell-side liquidity above)
        for i in range(len(swing_highs)-1):
            for j in range(i+1, len(swing_highs)):
                h1, h2 = swing_highs[i].price, swing_highs[j].price
                if abs(h1-h2)/h1 < tol:
                    levels.append(LiquidityLevel(
                        level_type='sell_side',
                        price=max(h1, h2),
                        strength=2.0
                    ))

        # Equal Lows (buy-side liquidity below)
        for i in range(len(swing_lows)-1):
            for j in range(i+1, len(swing_lows)):
                l1, l2 = swing_lows[i].price, swing_lows[j].price
                if abs(l1-l2)/l1 < tol:
                    levels.append(LiquidityLevel(
                        level_type='buy_side',
                        price=min(l1, l2),
                        strength=2.0
                    ))

        # Previous highs/lows as liquidity
        if swing_highs:
            levels.append(LiquidityLevel(
                level_type='sell_side',
                price=swing_highs[-1].price,
                strength=1.0
            ))
        if swing_lows:
            levels.append(LiquidityLevel(
                level_type='buy_side',
                price=swing_lows[-1].price,
                strength=1.0
            ))

        return levels

    # ── OPTIMAL TRADE ENTRY (OTE) ─────────────────────────

    def _calculate_ote(self, df: pd.DataFrame,
                       swings: list[SwingPoint],
                       structure: MarketStructure) -> tuple:
        """
        OTE = Fibonacci 61.8% – 79% retracement of the last impulse move.
        Bullish: retracement from swing low to swing high
        Bearish: retracement from swing high to swing low
        """
        swing_highs = [s for s in swings if s.direction == 'high']
        swing_lows  = [s for s in swings if s.direction == 'low']

        if not swing_highs or not swing_lows:
            price = df['close'].iloc[-1]
            return (price * 0.99, price * 1.01)

        last_high = swing_highs[-1].price
        last_low  = swing_lows[-1].price
        move_size = last_high - last_low

        if structure.trend == 'bullish':
            # Pullback into OTE (61.8% – 79% from low)
            ote_low  = last_high - move_size * 0.79
            ote_high = last_high - move_size * 0.618
            return (round(ote_low, 4), round(ote_high, 4))
        elif structure.trend == 'bearish':
            # Rally into OTE (61.8% – 79% from high)
            ote_low  = last_low + move_size * 0.618
            ote_high = last_low + move_size * 0.79
            return (round(ote_low, 4), round(ote_high, 4))
        else:
            price = df['close'].iloc[-1]
            return (price * 0.99, price * 1.01)

    # ── SIGNAL GENERATION ─────────────────────────────────

    def _generate_signal(self, df: pd.DataFrame, price: float,
                         structure: MarketStructure,
                         obs: list[OrderBlock],
                         fvgs: list[FairValueGap],
                         liquidity: list[LiquidityLevel],
                         ote_zone: tuple) -> ICTSignal:
        reasons  = []
        score    = 0.0
        action   = 'WAIT'
        best_ob  = None
        best_fvg = None

        # 1. Market Structure bias
        if structure.trend == 'bullish':
            score += 0.20
            reasons.append(f'Bullish market structure (HH/HL)')
        elif structure.trend == 'bearish':
            score -= 0.20
            reasons.append(f'Bearish market structure (LH/LL)')

        # 2. BOS / CHoCH
        if structure.last_bos:
            score += 0.15 if structure.trend == 'bullish' else -0.15
            reasons.append(f'Break of Structure at ${structure.last_bos:.2f}')
        if structure.last_choch:
            reasons.append(f'Change of Character detected — potential reversal')

        # 3. Order Block proximity
        bullish_obs = [ob for ob in obs if ob.direction == 'bullish' and ob.bottom <= price <= ob.top * 1.005]
        bearish_obs = [ob for ob in obs if ob.direction == 'bearish' and ob.bottom * 0.995 <= price <= ob.top]

        if bullish_obs:
            best_ob = max(bullish_obs, key=lambda x: x.strength)
            score  += 0.25 * best_ob.strength
            reasons.append(f'Price in Bullish OB zone (str:{best_ob.strength:.2f})')
        if bearish_obs:
            best_ob = max(bearish_obs, key=lambda x: x.strength)
            score  -= 0.25 * best_ob.strength
            reasons.append(f'Price in Bearish OB zone (str:{best_ob.strength:.2f})')

        # 4. FVG proximity
        bull_fvgs = [f for f in fvgs if f.direction=='bullish' and f.bottom<=price<=f.top]
        bear_fvgs = [f for f in fvgs if f.direction=='bearish' and f.bottom<=price<=f.top]

        if bull_fvgs:
            best_fvg = max(bull_fvgs, key=lambda x: x.size_pct)
            score   += 0.15
            reasons.append(f'Inside Bullish FVG (size:{best_fvg.size_pct:.3%})')
        if bear_fvgs:
            best_fvg = max(bear_fvgs, key=lambda x: x.size_pct)
            score   -= 0.15
            reasons.append(f'Inside Bearish FVG (size:{best_fvg.size_pct:.3%})')

        # 5. OTE zone
        ote_low, ote_high = ote_zone
        if ote_low <= price <= ote_high:
            bonus = 0.20 if structure.trend == 'bullish' else -0.20
            score += bonus
            reasons.append(f'Price in OTE zone (Fib 61.8-79%): ${ote_low:.2f}-${ote_high:.2f}')

        # 6. Liquidity targets
        buy_liq  = [l for l in liquidity if l.level_type == 'buy_side'  and l.price < price]
        sell_liq = [l for l in liquidity if l.level_type == 'sell_side' and l.price > price]

        nearest_tp = None
        if sell_liq and score > 0:
            nearest_tp = min(sell_liq, key=lambda x: abs(x.price - price)).price
            reasons.append(f'Sell-side liquidity target: ${nearest_tp:.2f}')
        elif buy_liq and score < 0:
            nearest_tp = max(buy_liq, key=lambda x: abs(x.price - price)).price
            reasons.append(f'Buy-side liquidity target: ${nearest_tp:.2f}')

        # 7. Final decision
        confidence = min(0.95, abs(score))

        if score >= 0.35:
            action = 'LONG'
        elif score <= -0.35:
            action = 'SHORT'
        else:
            action = 'WAIT'
            confidence = max(0, confidence - 0.1)

        # SL / TP calculation
        atr = df['high'].rolling(14).mean().iloc[-1] - df['low'].rolling(14).mean().iloc[-1]
        sl_long  = price - atr * 1.5
        sl_short = price + atr * 1.5

        tp1_long  = price + atr * 2.0
        tp2_long  = price + atr * 3.5
        tp3_long  = nearest_tp if nearest_tp and nearest_tp > price else price + atr * 5.0

        tp1_short  = price - atr * 2.0
        tp2_short  = price - atr * 3.5
        tp3_short  = nearest_tp if nearest_tp and nearest_tp < price else price - atr * 5.0

        return ICTSignal(
            action=action,
            confidence=round(confidence, 4),
            entry_price=round(price, 4),
            ote_zone=ote_zone,
            reasons=reasons,
            order_block=best_ob,
            fvg=best_fvg,
            sl_price=round(sl_long if action=='LONG' else sl_short, 4),
            tp1_price=round(tp1_long if action=='LONG' else tp1_short, 4),
            tp2_price=round(tp2_long if action=='LONG' else tp2_short, 4),
            tp3_price=round(tp3_long if action=='LONG' else tp3_short, 4),
            liquidity_target=nearest_tp,
        )
