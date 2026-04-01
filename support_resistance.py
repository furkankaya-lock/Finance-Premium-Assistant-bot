"""
ai_engine/support_resistance.py
================================
Support & Resistance Engine
Methods: Classic Pivot · Fibonacci Retracement · Swing High/Low
         Volume Profile · Dynamic S/R · Partial Exit Levels
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass

log = logging.getLogger("CryptoBot.SR")


@dataclass
class SRLevel:
    price: float
    level_type: str      # 'support' | 'resistance' | 'pivot' | 'fib'
    strength: float      # 0.0 – 1.0 (touch count based)
    label: str           # 'S1', 'R1', 'Fib 61.8%', etc.
    touches: int = 0


@dataclass
class SRAnalysis:
    current_price: float
    nearest_support: float
    nearest_resistance: float
    pivot: float
    supports: list
    resistances: list
    fib_levels: dict
    partial_exit_levels: list   # Levels to take partial profit
    distance_to_resistance_pct: float
    distance_to_support_pct: float
    near_resistance: bool       # Price within 0.5% of resistance
    near_support: bool


class SupportResistanceEngine:
    """
    Comprehensive Support & Resistance analysis.
    Calculates Pivot Points, Fibonacci levels, and Swing-based S/R.
    """

    TOUCH_TOLERANCE = 0.003   # 0.3% price band for level touches

    def __init__(self, lookback: int = 100):
        self.lookback = lookback

    # ── MASTER ANALYSIS ───────────────────────────────────

    def analyze(self, df: pd.DataFrame, current_price: float) -> SRAnalysis:
        df = df.tail(self.lookback).copy()

        pivots    = self._calculate_pivots(df)
        fib_levels = self._calculate_fibonacci(df)
        swing_sr  = self._find_swing_sr(df)
        all_levels = pivots + swing_sr

        # Merge close levels
        merged = self._merge_levels(all_levels, current_price)

        supports     = sorted([l for l in merged if l.level_type == 'support' and l.price < current_price],
                              key=lambda x: x.price, reverse=True)
        resistances  = sorted([l for l in merged if l.level_type == 'resistance' and l.price > current_price],
                              key=lambda x: x.price)

        nearest_sup = supports[0].price    if supports    else current_price * 0.97
        nearest_res = resistances[0].price if resistances else current_price * 1.03

        pivot_price = pivots[0].price if pivots else current_price

        dist_res = (nearest_res - current_price) / current_price * 100
        dist_sup = (current_price - nearest_sup) / current_price * 100

        # Partial exit levels (every resistance toward target)
        partial_exits = self._partial_exit_levels(resistances, current_price)

        return SRAnalysis(
            current_price            = current_price,
            nearest_support          = round(nearest_sup, 4),
            nearest_resistance       = round(nearest_res, 4),
            pivot                    = round(pivot_price, 4),
            supports                 = supports[:5],
            resistances              = resistances[:5],
            fib_levels               = {k: round(v, 4) for k, v in fib_levels.items()},
            partial_exit_levels      = partial_exits,
            distance_to_resistance_pct = round(dist_res, 3),
            distance_to_support_pct    = round(dist_sup, 3),
            near_resistance          = dist_res <= 0.5,
            near_support             = dist_sup <= 0.5,
        )

    # ── PIVOT POINTS ──────────────────────────────────────

    def _calculate_pivots(self, df: pd.DataFrame) -> list[SRLevel]:
        """Classic Floor Pivot Points from last complete candle"""
        prev = df.iloc[-2]
        H, L, C = float(prev['high']), float(prev['low']), float(prev['close'])

        pivot = (H + L + C) / 3
        r1 = 2 * pivot - L
        r2 = pivot + (H - L)
        r3 = H + 2 * (pivot - L)
        s1 = 2 * pivot - H
        s2 = pivot - (H - L)
        s3 = L - 2 * (H - pivot)

        levels = [
            SRLevel(price=pivot, level_type='pivot',      strength=0.8, label='Pivot'),
            SRLevel(price=r1,    level_type='resistance', strength=0.7, label='R1'),
            SRLevel(price=r2,    level_type='resistance', strength=0.6, label='R2'),
            SRLevel(price=r3,    level_type='resistance', strength=0.5, label='R3'),
            SRLevel(price=s1,    level_type='support',    strength=0.7, label='S1'),
            SRLevel(price=s2,    level_type='support',    strength=0.6, label='S2'),
            SRLevel(price=s3,    level_type='support',    strength=0.5, label='S3'),
        ]
        return levels

    # ── FIBONACCI ─────────────────────────────────────────

    def _calculate_fibonacci(self, df: pd.DataFrame) -> dict:
        """Fibonacci retracement from recent swing high to swing low"""
        high = float(df['high'].max())
        low  = float(df['low'].min())
        diff = high - low

        return {
            '0.0':   high,
            '23.6':  high - diff * 0.236,
            '38.2':  high - diff * 0.382,
            '50.0':  high - diff * 0.500,
            '61.8':  high - diff * 0.618,
            '70.5':  high - diff * 0.705,
            '78.6':  high - diff * 0.786,
            '88.6':  high - diff * 0.886,
            '100.0': low,
            '127.2': low  - diff * 0.272,
            '161.8': low  - diff * 0.618,
        }

    # ── SWING HIGH / LOW S/R ──────────────────────────────

    def _find_swing_sr(self, df: pd.DataFrame,
                       window: int = 8) -> list[SRLevel]:
        levels = []
        highs  = df['high'].values
        lows   = df['low'].values
        closes = df['close'].values
        n = len(df)
        tol = self.TOUCH_TOLERANCE
        price_map = {}

        for i in range(window, n - window):
            # Swing High
            if highs[i] == max(highs[i-window:i+window+1]):
                p = round(highs[i], 2)
                key = round(p / (p * tol + 1e-9))
                if key not in price_map:
                    price_map[key] = {'price': p, 'type': 'resistance', 'touches': 0}
                price_map[key]['touches'] += 1

            # Swing Low
            if lows[i] == min(lows[i-window:i+window+1]):
                p = round(lows[i], 2)
                key = round(p / (p * tol + 1e-9))
                if key not in price_map:
                    price_map[key] = {'price': p, 'type': 'support', 'touches': 0}
                price_map[key]['touches'] += 1

        # Count touches for each level
        for data in price_map.values():
            touches = data['touches']
            strength = min(1.0, touches / 5.0)
            if touches >= 2:
                levels.append(SRLevel(
                    price=data['price'],
                    level_type=data['type'],
                    strength=round(strength, 3),
                    label=f"Swing {'H' if data['type']=='resistance' else 'L'} ({touches}x)",
                    touches=touches,
                ))

        return sorted(levels, key=lambda x: x.strength, reverse=True)[:20]

    # ── MERGE CLOSE LEVELS ────────────────────────────────

    def _merge_levels(self, levels: list[SRLevel],
                      current_price: float) -> list[SRLevel]:
        """Merge levels within 0.5% of each other"""
        if not levels: return []
        merged = []
        used   = set()
        tol    = 0.005

        for i, l in enumerate(levels):
            if i in used: continue
            group = [l]
            for j, l2 in enumerate(levels):
                if j != i and j not in used:
                    if abs(l.price - l2.price) / l.price < tol:
                        group.append(l2)
                        used.add(j)
            used.add(i)
            # Average the group
            avg_price = np.mean([x.price for x in group])
            avg_str   = max(x.strength for x in group)
            types     = [x.level_type for x in group]
            level_type = 'support' if types.count('support') > types.count('resistance') else 'resistance'
            labels    = ' / '.join(set(x.label for x in group))
            merged.append(SRLevel(
                price=round(avg_price, 4),
                level_type=level_type,
                strength=avg_str,
                label=labels,
                touches=sum(x.touches for x in group),
            ))

        return merged

    # ── PARTIAL EXIT LEVELS ───────────────────────────────

    def _partial_exit_levels(self, resistances: list[SRLevel],
                             current_price: float) -> list[dict]:
        """
        Calculate partial exit strategy:
        - First resistance  → exit 30% of position
        - Second resistance → exit 30% more
        - Third resistance  → exit remaining 40%
        """
        exit_pcts = [0.30, 0.30, 0.40]
        exits = []
        for i, res in enumerate(resistances[:3]):
            if i < len(exit_pcts):
                exits.append({
                    'price':       res.price,
                    'exit_pct':    exit_pcts[i],
                    'label':       res.label,
                    'dist_pct':    round((res.price - current_price) / current_price * 100, 2),
                })
        return exits

    # ── QUICK SUMMARY ─────────────────────────────────────

    def summary_dict(self, analysis: SRAnalysis) -> dict:
        return {
            'nearest_support':     analysis.nearest_support,
            'nearest_resistance':  analysis.nearest_resistance,
            'pivot':               analysis.pivot,
            'near_resistance':     analysis.near_resistance,
            'near_support':        analysis.near_support,
            'dist_to_res_pct':     analysis.distance_to_resistance_pct,
            'dist_to_sup_pct':     analysis.distance_to_support_pct,
            'partial_exits':       analysis.partial_exit_levels,
            'fib_618':             analysis.fib_levels.get('61.8', 0),
            'fib_786':             analysis.fib_levels.get('78.6', 0),
            'fib_382':             analysis.fib_levels.get('38.2', 0),
        }
