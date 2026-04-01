"""
memory/memory_manager.py
========================
Long-term memory system for the trading bot.
Stores market summaries, strategy performance, important events,
and regime history. Claude reads this on every analysis.
"""

import os
import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger("CryptoBot.Memory")

MEMORY_DIR  = "memory"
MEMORY_FILE = os.path.join(MEMORY_DIR, "bot_memory.json")
os.makedirs(MEMORY_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# DEFAULT MEMORY STRUCTURE
# ─────────────────────────────────────────────────────────────

DEFAULT_MEMORY = {
    "version":          "1.0",
    "created_at":       datetime.now().isoformat(),
    "last_updated":     datetime.now().isoformat(),

    # Market context
    "market_regime": {
        "current":       "unknown",        # bullish | bearish | ranging | volatile | accumulation
        "since":         None,
        "previous":      None,
        "regime_history": [],              # last 10 regimes
    },

    # Performance memory
    "strategy_performance": {
        "all_time": {
            "total_trades":  0,
            "wins":          0,
            "losses":        0,
            "total_pnl":     0.0,
            "win_rate":      0.0,
            "profit_factor": 0.0,
            "best_trade":    0.0,
            "worst_trade":   0.0,
        },
        "by_symbol": {},      # {BTCUSDT: {wins, losses, pnl}}
        "by_strategy": {},    # {rsi: {wins, losses}, ema: {...}}
        "monthly_pnl": [],    # last 12 months [{month, pnl, trades}]
    },

    # Important events log
    "important_events": [],   # last 30 events [{date, type, description, impact}]

    # Market summaries
    "daily_summaries": [],    # last 30 days [{date, btc_change, eth_change, regime, notes}]
    "weekly_summary":  "",
    "monthly_summary": "",

    # Bot learnings
    "learnings": [],          # [{date, learning, confidence}]

    # Risk notes
    "risk_notes": {
        "current_risk_level": "medium",
        "max_drawdown_ever":  0.0,
        "worst_period":       None,
        "best_period":        None,
    },

    # Macro snapshot
    "macro_snapshot": {
        "last_updated":   None,
        "btc_dominance":  0.0,
        "total_mcap":     0.0,
        "fear_greed":     50,
        "fg_trend":       "neutral",
        "funding_rate":   0.0,
        "btc_halving_days": None,
    },
}


# ─────────────────────────────────────────────────────────────
# MEMORY MANAGER
# ─────────────────────────────────────────────────────────────

class MemoryManager:
    """
    Thread-safe long-term memory for the trading bot.
    Persists to disk as JSON. Claude reads a summary on every analysis.
    """

    def __init__(self):
        self._lock  = threading.Lock()
        self.memory = self._load()

    # ── LOAD / SAVE ───────────────────────────────────────────

    def _load(self) -> dict:
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                log.info("[Memory] Loaded from disk")
                # Merge with defaults to handle new fields
                return self._merge_defaults(data)
            except Exception as e:
                log.warning(f"[Memory] Load failed: {e} — starting fresh")
        return DEFAULT_MEMORY.copy()

    def _save(self):
        try:
            self.memory["last_updated"] = datetime.now().isoformat()
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log.error(f"[Memory] Save failed: {e}")

    def _merge_defaults(self, data: dict) -> dict:
        """Add any missing keys from DEFAULT_MEMORY."""
        for key, val in DEFAULT_MEMORY.items():
            if key not in data:
                data[key] = val
        return data

    # ── TRADE RECORDING ───────────────────────────────────────

    def record_trade(self, symbol: str, pnl: float,
                     strategy: str = "unknown",
                     side: str = "long") -> None:
        """Record a completed trade into memory."""
        with self._lock:
            perf = self.memory["strategy_performance"]

            # All-time stats
            at = perf["all_time"]
            at["total_trades"] += 1
            at["total_pnl"] = round(at["total_pnl"] + pnl, 4)
            if pnl > 0:
                at["wins"] += 1
                at["best_trade"] = max(at["best_trade"], pnl)
            else:
                at["losses"] += 1
                at["worst_trade"] = min(at["worst_trade"], pnl)
            total = at["total_trades"]
            at["win_rate"] = round(at["wins"] / total, 4) if total > 0 else 0
            gp = sum(t for t in [at["best_trade"]] if t > 0)
            gl = abs(at["worst_trade"]) if at["worst_trade"] < 0 else 1
            at["profit_factor"] = round(at["wins"] * abs(at.get("avg_win", 1)) /
                                        max(at["losses"] * abs(at.get("avg_loss", 1)), 0.001), 3)

            # By symbol
            sym = perf["by_symbol"].setdefault(symbol, {"wins": 0, "losses": 0, "pnl": 0.0})
            sym["pnl"] = round(sym["pnl"] + pnl, 4)
            if pnl > 0: sym["wins"] += 1
            else:       sym["losses"] += 1

            # By strategy
            strat = perf["by_strategy"].setdefault(strategy, {"wins": 0, "losses": 0, "pnl": 0.0})
            strat["pnl"] = round(strat["pnl"] + pnl, 4)
            if pnl > 0: strat["wins"] += 1
            else:       strat["losses"] += 1

            self._save()
        log.debug(f"[Memory] Trade recorded: {symbol} {pnl:+.4f}")

    # ── MARKET REGIME ─────────────────────────────────────────

    def update_regime(self, regime: str) -> None:
        """Update current market regime."""
        with self._lock:
            reg = self.memory["market_regime"]
            if reg["current"] != regime:
                # Regime changed
                reg["previous"] = reg["current"]
                reg["current"]  = regime
                reg["since"]    = datetime.now().isoformat()
                history = reg["regime_history"]
                history.append({
                    "regime": regime,
                    "date":   datetime.now().strftime("%Y-%m-%d"),
                })
                reg["regime_history"] = history[-10:]  # Keep last 10
            self._save()

    # ── EVENTS ────────────────────────────────────────────────

    def add_event(self, event_type: str, description: str,
                  impact: str = "neutral") -> None:
        """
        Log an important market event.
        event_type: 'fed_decision' | 'halving' | 'crash' | 'rally' | 'news' | 'signal'
        impact: 'bullish' | 'bearish' | 'neutral'
        """
        with self._lock:
            events = self.memory["important_events"]
            events.append({
                "date":        datetime.now().strftime("%Y-%m-%d %H:%M"),
                "type":        event_type,
                "description": description,
                "impact":      impact,
            })
            self.memory["important_events"] = events[-30:]
            self._save()

    # ── DAILY SUMMARY ─────────────────────────────────────────

    def add_daily_summary(self, btc_change: float, eth_change: float,
                          regime: str, notes: str = "") -> None:
        """Add today's market summary."""
        with self._lock:
            summaries = self.memory["daily_summaries"]
            today = datetime.now().strftime("%Y-%m-%d")

            # Update if today already exists
            for s in summaries:
                if s["date"] == today:
                    s.update({"btc_change": round(btc_change, 2),
                              "eth_change": round(eth_change, 2),
                              "regime": regime, "notes": notes})
                    self._save()
                    return

            summaries.append({
                "date":       today,
                "btc_change": round(btc_change, 2),
                "eth_change": round(eth_change, 2),
                "regime":     regime,
                "notes":      notes,
            })
            self.memory["daily_summaries"] = summaries[-30:]
            self._save()

    # ── LEARNINGS ─────────────────────────────────────────────

    def add_learning(self, learning: str, confidence: float = 0.7) -> None:
        """Store a bot learning/observation."""
        with self._lock:
            learnings = self.memory["learnings"]
            learnings.append({
                "date":       datetime.now().strftime("%Y-%m-%d"),
                "learning":   learning,
                "confidence": round(confidence, 2),
            })
            self.memory["learnings"] = learnings[-20:]
            self._save()

    # ── MACRO UPDATE ──────────────────────────────────────────

    def update_macro(self, data: dict) -> None:
        """Update macro snapshot."""
        with self._lock:
            snap = self.memory["macro_snapshot"]
            snap.update(data)
            snap["last_updated"] = datetime.now().isoformat()
            self._save()

    # ── MONTHLY PNL ───────────────────────────────────────────

    def update_monthly_pnl(self) -> None:
        """Recalculate this month's P&L from trade history."""
        with self._lock:
            now   = datetime.now()
            month = now.strftime("%Y-%m")
            perf  = self.memory["strategy_performance"]
            monthly = perf["monthly_pnl"]

            # Find or create this month
            entry = next((m for m in monthly if m["month"] == month), None)
            if not entry:
                entry = {"month": month, "pnl": 0.0, "trades": 0}
                monthly.append(entry)
                perf["monthly_pnl"] = monthly[-12:]
            self._save()

    # ── CLAUDE CONTEXT BUILDER ────────────────────────────────

    def build_context(self) -> str:
        """
        Build a concise memory context string for Claude.
        This is injected into every Claude analysis prompt.
        Max ~400 tokens to keep API costs low.
        """
        with self._lock:
            m   = self.memory
            reg = m["market_regime"]
            at  = m["strategy_performance"]["all_time"]
            snp = m["macro_snapshot"]
            events = m["important_events"][-5:]
            summaries = m["daily_summaries"][-7:]
            learnings = m["learnings"][-3:]

            # Recent events string
            events_str = ""
            if events:
                events_str = " | ".join(
                    f"{e['date'][-5:]}: {e['description'][:50]} ({e['impact']})"
                    for e in events[-3:]
                )

            # Last 7 days market summary
            days_str = ""
            if summaries:
                days_str = ", ".join(
                    f"{s['date'][-5:]} BTC:{s['btc_change']:+.1f}%"
                    for s in summaries[-5:]
                )

            # Best/worst strategies
            by_strat = m["strategy_performance"]["by_strategy"]
            best_strat = ""
            if by_strat:
                sorted_strats = sorted(
                    by_strat.items(),
                    key=lambda x: x[1].get("pnl", 0),
                    reverse=True
                )
                if sorted_strats:
                    best = sorted_strats[0]
                    best_strat = f"{best[0].upper()} (${best[1]['pnl']:+.2f})"

            # Learnings
            learn_str = ""
            if learnings:
                learn_str = " | ".join(l["learning"][:60] for l in learnings[-2:])

            context = f"""
=== BOT MEMORY CONTEXT ===
Market Regime: {reg['current']} (since {str(reg.get('since','?'))[:10]})
Previous Regime: {reg.get('previous', 'unknown')}

Performance (All-time):
  Trades: {at['total_trades']} | Win Rate: {at['win_rate']:.1%} | Total PnL: ${at['total_pnl']:.2f}
  Best Trade: +${at['best_trade']:.2f} | Worst Trade: ${at['worst_trade']:.2f}
  Best Strategy: {best_strat or 'N/A'}

Last 5 Events: {events_str or 'None'}
Last 7 Days: {days_str or 'No data'}

Macro Snapshot:
  BTC Dominance: {snp.get('btc_dominance', 0):.1f}%
  Fear & Greed: {snp.get('fear_greed', 50)}/100 ({snp.get('fg_trend', 'neutral')})
  Funding Rate: {snp.get('funding_rate', 0):.4f}%

Bot Learnings: {learn_str or 'None yet'}
=== END MEMORY ===
""".strip()

            return context

    # ── STATS ─────────────────────────────────────────────────

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "all_time":     self.memory["strategy_performance"]["all_time"],
                "regime":       self.memory["market_regime"]["current"],
                "events_count": len(self.memory["important_events"]),
                "learnings":    len(self.memory["learnings"]),
                "last_updated": self.memory["last_updated"],
            }

    def reset(self) -> None:
        """Full memory reset (use with caution)."""
        with self._lock:
            self.memory = DEFAULT_MEMORY.copy()
            self.memory["created_at"] = datetime.now().isoformat()
            self._save()
        log.warning("[Memory] Memory reset to defaults")
