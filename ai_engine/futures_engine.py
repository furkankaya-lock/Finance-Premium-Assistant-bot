"""
ai_engine/futures_engine.py
============================
Futures Trading Engine
Features: Real Long/Short · Position Reversal · Partial Exit
          Hold Until Opposite Signal · ICT + SR Integration
"""

import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from binance.client import Client
from binance.exceptions import BinanceAPIException

log = logging.getLogger("CryptoBot.Futures")


# ─────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class FuturesPosition:
    symbol: str
    side: str                    # 'LONG' | 'SHORT'
    entry_price: float
    quantity: float
    leverage: int
    sl_price: float
    tp1_price: float
    tp2_price: float
    tp3_price: float
    opened_at: str = field(default_factory=lambda: datetime.now().isoformat())
    partial_exits_done: list = field(default_factory=list)
    remaining_qty: float = 0.0
    unrealized_pnl: float = 0.0
    peak_price: float = 0.0      # For trailing stop
    mode: str = 'futures'        # 'futures' | 'demo'


@dataclass
class TradeResult:
    success: bool
    action: str
    symbol: str
    price: float
    quantity: float
    pnl: float = 0.0
    reason: str = ''
    order_id: str = ''


# ─────────────────────────────────────────────────────────────
# FUTURES ENGINE
# ─────────────────────────────────────────────────────────────

class FuturesEngine:
    """
    Manages Binance Futures positions with full ICT/SR integration.
    Supports: Long, Short, Reversal, Partial Exit, Trailing Stop.
    """

    def __init__(self,
                 client: Client,
                 default_leverage: int = 10,
                 demo_mode: bool = False):
        self.client           = client
        self.default_leverage = default_leverage
        self.demo_mode        = demo_mode
        self.positions: dict[str, FuturesPosition] = {}
        self.trade_history: list = []

    # ── POSITION MANAGEMENT ───────────────────────────────

    def open_long(self, symbol: str, usdt_amount: float,
                  sl: float, tp1: float, tp2: float, tp3: float,
                  leverage: int = None) -> TradeResult:
        """Open a LONG futures position."""
        lev = leverage or self.default_leverage
        return self._open_position(symbol, 'LONG', usdt_amount, sl, tp1, tp2, tp3, lev)

    def open_short(self, symbol: str, usdt_amount: float,
                   sl: float, tp1: float, tp2: float, tp3: float,
                   leverage: int = None) -> TradeResult:
        """Open a SHORT futures position."""
        lev = leverage or self.default_leverage
        return self._open_position(symbol, 'SHORT', usdt_amount, sl, tp1, tp2, tp3, lev)

    def reverse_position(self, symbol: str, new_side: str,
                         usdt_amount: float,
                         sl: float, tp1: float, tp2: float, tp3: float,
                         current_price: float) -> TradeResult:
        """
        Reverse: close existing position, open opposite.
        LONG → SHORT or SHORT → LONG
        """
        if symbol in self.positions:
            log.info(f"[Futures] Reversing {symbol}: {self.positions[symbol].side} → {new_side}")
            close_result = self.close_position(symbol, current_price, reason=f'Reversed to {new_side}')
            if not close_result.success:
                return close_result

        return self._open_position(symbol, new_side, usdt_amount, sl, tp1, tp2, tp3,
                                   self.default_leverage)

    def partial_exit(self, symbol: str, exit_pct: float,
                     current_price: float, reason: str = '') -> TradeResult:
        """
        Exit a percentage of the position.
        exit_pct: 0.0 – 1.0 (e.g., 0.30 = exit 30%)
        """
        if symbol not in self.positions:
            return TradeResult(success=False, action='PARTIAL_EXIT',
                               symbol=symbol, price=current_price,
                               quantity=0, reason='No open position')

        pos   = self.positions[symbol]
        qty   = round(pos.remaining_qty * exit_pct, 6)
        if qty <= 0:
            return TradeResult(success=False, action='PARTIAL_EXIT',
                               symbol=symbol, price=current_price,
                               quantity=0, reason='Zero quantity')

        pnl = self._calc_pnl(pos, current_price, qty)

        if not self.demo_mode:
            try:
                side = 'SELL' if pos.side == 'LONG' else 'BUY'
                self.client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type='MARKET',
                    quantity=qty,
                    reduceOnly=True,
                )
            except BinanceAPIException as e:
                log.error(f"[Futures] Partial exit error: {e}")
                return TradeResult(success=False, action='PARTIAL_EXIT',
                                   symbol=symbol, price=current_price,
                                   quantity=qty, reason=str(e))

        pos.remaining_qty    -= qty
        pos.partial_exits_done.append({
            'price': current_price, 'qty': qty,
            'pnl': pnl, 'reason': reason,
            'time': datetime.now().isoformat(),
        })

        log.info(f"[Futures] Partial exit {symbol} | {exit_pct:.0%} | "
                 f"qty:{qty} | pnl:${pnl:.4f} | {reason}")

        result = TradeResult(success=True, action='PARTIAL_EXIT',
                             symbol=symbol, price=current_price,
                             quantity=qty, pnl=pnl, reason=reason)
        self.trade_history.append(vars(result))
        return result

    def close_position(self, symbol: str, current_price: float,
                       reason: str = '') -> TradeResult:
        """Close entire position."""
        if symbol not in self.positions:
            return TradeResult(success=False, action='CLOSE',
                               symbol=symbol, price=current_price,
                               quantity=0, reason='No position')

        pos = self.positions[symbol]
        qty = pos.remaining_qty
        pnl = self._calc_pnl(pos, current_price, qty)
        pnl += sum(e['pnl'] for e in pos.partial_exits_done)

        if not self.demo_mode:
            try:
                side = 'SELL' if pos.side == 'LONG' else 'BUY'
                self.client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type='MARKET',
                    quantity=round(qty, 6),
                    reduceOnly=True,
                )
            except BinanceAPIException as e:
                log.error(f"[Futures] Close error: {e}")
                return TradeResult(success=False, action='CLOSE',
                                   symbol=symbol, price=current_price,
                                   quantity=qty, reason=str(e), pnl=pnl)

        log.info(f"[Futures] CLOSED {symbol} | {pos.side} | "
                 f"entry:${pos.entry_price:.4f} exit:${current_price:.4f} | "
                 f"pnl:${pnl:.4f} | {reason}")

        del self.positions[symbol]
        result = TradeResult(success=True, action='CLOSE',
                             symbol=symbol, price=current_price,
                             quantity=qty, pnl=pnl, reason=reason)
        self.trade_history.append(vars(result))
        return result

    # ── POSITION MONITORING ───────────────────────────────

    def monitor_position(self, symbol: str, current_price: float,
                         partial_exit_levels: list,
                         atr: float,
                         opposite_signal: bool = False) -> Optional[str]:
        """
        Called every scan. Returns action taken or None.
        Handles: SL, TP1/2/3, Partial Exit, Trailing Stop, Reversal signal.

        opposite_signal: True if short signal while long (or vice versa)
        """
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        self._update_unrealized_pnl(pos, current_price)

        action_taken = None

        # 1. Stop Loss
        if pos.side == 'LONG' and current_price <= pos.sl_price:
            self.close_position(symbol, current_price, 'SL_HIT')
            return 'SL_HIT'
        if pos.side == 'SHORT' and current_price >= pos.sl_price:
            self.close_position(symbol, current_price, 'SL_HIT')
            return 'SL_HIT'

        # 2. Opposite signal → reverse
        if opposite_signal:
            new_side = 'SHORT' if pos.side == 'LONG' else 'LONG'
            log.info(f"[Futures] Opposite signal — will reverse {symbol} to {new_side}")
            return f'REVERSE_TO_{new_side}'

        # 3. Partial exits at resistance/support levels
        for level in partial_exit_levels:
            level_price = level['price']
            exit_pct    = level['exit_pct']
            label       = level['label']

            if level_price in pos.partial_exits_done:
                continue

            triggered = False
            if pos.side == 'LONG'  and current_price >= level_price:
                triggered = True
            if pos.side == 'SHORT' and current_price <= level_price:
                triggered = True

            if triggered and pos.remaining_qty > 0:
                result = self.partial_exit(symbol, exit_pct, current_price,
                                           reason=f'Partial exit at {label}')
                if result.success:
                    pos.partial_exits_done.append(level_price)
                    action_taken = f'PARTIAL_{int(exit_pct*100)}pct_AT_{label}'

        # 4. TP levels (full close at TP3)
        if pos.side == 'LONG':
            if current_price >= pos.tp3_price:
                self.close_position(symbol, current_price, 'TP3_HIT')
                return 'TP3_HIT'
            if current_price >= pos.tp2_price and 'tp2' not in pos.partial_exits_done:
                self.partial_exit(symbol, 0.30, current_price, 'TP2_HIT')
                pos.partial_exits_done.append('tp2')
            if current_price >= pos.tp1_price and 'tp1' not in pos.partial_exits_done:
                self.partial_exit(symbol, 0.30, current_price, 'TP1_HIT')
                pos.partial_exits_done.append('tp1')
                # Move SL to breakeven
                pos.sl_price = pos.entry_price
                log.info(f"[Futures] SL moved to breakeven: ${pos.sl_price:.4f}")

        elif pos.side == 'SHORT':
            if current_price <= pos.tp3_price:
                self.close_position(symbol, current_price, 'TP3_HIT')
                return 'TP3_HIT'
            if current_price <= pos.tp2_price and 'tp2' not in pos.partial_exits_done:
                self.partial_exit(symbol, 0.30, current_price, 'TP2_HIT')
                pos.partial_exits_done.append('tp2')
            if current_price <= pos.tp1_price and 'tp1' not in pos.partial_exits_done:
                self.partial_exit(symbol, 0.30, current_price, 'TP1_HIT')
                pos.partial_exits_done.append('tp1')
                pos.sl_price = pos.entry_price

        # 5. Trailing stop update
        self._update_trailing_stop(pos, current_price, atr)

        return action_taken

    # ── LEVERAGE ──────────────────────────────────────────

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        if self.demo_mode:
            self.default_leverage = leverage
            return True
        try:
            self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
            self.default_leverage = leverage
            log.info(f"[Futures] Leverage set: {symbol} × {leverage}")
            return True
        except BinanceAPIException as e:
            log.error(f"[Futures] Leverage error: {e}")
            return False

    def get_position_info(self, symbol: str) -> Optional[dict]:
        """Get real-time position info from Binance."""
        if self.demo_mode or symbol not in self.positions:
            return None
        try:
            positions = self.client.futures_position_information(symbol=symbol)
            for p in positions:
                if float(p['positionAmt']) != 0:
                    return {
                        'side':             'LONG' if float(p['positionAmt'])>0 else 'SHORT',
                        'entry_price':      float(p['entryPrice']),
                        'unrealized_pnl':   float(p['unRealizedProfit']),
                        'liquidation_price':float(p['liquidationPrice']),
                        'leverage':         int(p['leverage']),
                        'quantity':         abs(float(p['positionAmt'])),
                    }
        except Exception as e:
            log.warning(f"[Futures] Position info error: {e}")
        return None

    # ── INTERNAL ──────────────────────────────────────────

    def _open_position(self, symbol: str, side: str,
                       usdt_amount: float,
                       sl: float, tp1: float, tp2: float, tp3: float,
                       leverage: int) -> TradeResult:

        if symbol in self.positions:
            return TradeResult(success=False, action=f'OPEN_{side}',
                               symbol=symbol, price=0,
                               quantity=0, reason='Position already open')

        # Set leverage
        if not self.demo_mode:
            self.set_leverage(symbol, leverage)

        # Get current price
        try:
            if not self.demo_mode:
                ticker = self.client.futures_symbol_ticker(symbol=symbol)
                price  = float(ticker['price'])
            else:
                price = sl / 0.97 if side == 'LONG' else sl / 1.03
        except Exception as e:
            return TradeResult(success=False, action=f'OPEN_{side}',
                               symbol=symbol, price=0, quantity=0, reason=str(e))

        qty = round((usdt_amount * leverage) / price, 6)

        if not self.demo_mode:
            try:
                binance_side = 'BUY' if side == 'LONG' else 'SELL'
                order = self.client.futures_create_order(
                    symbol=symbol,
                    side=binance_side,
                    type='MARKET',
                    quantity=qty,
                )
                price = float(order.get('avgPrice', price))
            except BinanceAPIException as e:
                log.error(f"[Futures] Open error: {e}")
                return TradeResult(success=False, action=f'OPEN_{side}',
                                   symbol=symbol, price=price,
                                   quantity=qty, reason=str(e))

        pos = FuturesPosition(
            symbol=symbol, side=side,
            entry_price=price, quantity=qty,
            leverage=leverage,
            sl_price=sl, tp1_price=tp1, tp2_price=tp2, tp3_price=tp3,
            remaining_qty=qty, peak_price=price,
            mode='demo' if self.demo_mode else 'futures',
        )
        self.positions[symbol] = pos

        log.info(f"[Futures] OPENED {side} {symbol} | ${price:.4f} × {qty:.6f} | "
                 f"×{leverage} | SL:${sl:.4f} TP1:${tp1:.4f} TP3:${tp3:.4f}")

        result = TradeResult(success=True, action=f'OPEN_{side}',
                             symbol=symbol, price=price, quantity=qty)
        self.trade_history.append(vars(result))
        return result

    def _calc_pnl(self, pos: FuturesPosition, exit_price: float, qty: float) -> float:
        if pos.side == 'LONG':
            pnl = (exit_price - pos.entry_price) * qty * pos.leverage
        else:
            pnl = (pos.entry_price - exit_price) * qty * pos.leverage
        return round(pnl, 6)

    def _update_unrealized_pnl(self, pos: FuturesPosition, price: float):
        pos.unrealized_pnl = self._calc_pnl(pos, price, pos.remaining_qty)
        if pos.side == 'LONG'  and price > pos.peak_price:
            pos.peak_price = price
        if pos.side == 'SHORT' and price < pos.peak_price:
            pos.peak_price = price

    def _update_trailing_stop(self, pos: FuturesPosition,
                              price: float, atr: float):
        """Trail SL by ATR × 1.5 from peak price."""
        trail_dist = atr * 1.5
        if pos.side == 'LONG':
            new_sl = pos.peak_price - trail_dist
            if new_sl > pos.sl_price:
                pos.sl_price = round(new_sl, 6)
        elif pos.side == 'SHORT':
            new_sl = pos.peak_price + trail_dist
            if new_sl < pos.sl_price:
                pos.sl_price = round(new_sl, 6)

    def stats(self) -> dict:
        trades = self.trade_history
        if not trades: return {'total': 0}
        closes = [t for t in trades if t.get('action') in ('CLOSE','PARTIAL_EXIT')]
        wins   = [t for t in closes if t.get('pnl', 0) > 0]
        total_pnl = sum(t.get('pnl', 0) for t in closes)
        return {
            'total':      len(closes),
            'wins':       len(wins),
            'losses':     len(closes)-len(wins),
            'win_rate':   round(len(wins)/len(closes), 3) if closes else 0,
            'total_pnl':  round(total_pnl, 4),
            'open_count': len(self.positions),
        }
