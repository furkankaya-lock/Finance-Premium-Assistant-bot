"""
ai_engine/orchestrator.py
==========================
Multi-Agent Orchestrator
Coordinates: MarketAgent · MLAgent · NewsAgent · PortfolioAgent · ExecutionAgent
Event-driven, non-blocking, consensus-based final decision.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, Callable

log = logging.getLogger("CryptoBot.Orchestrator")


# ─────────────────────────────────────────────────────────────
# AGENT RESULTS
# ─────────────────────────────────────────────────────────────

@dataclass
class AgentSignal:
    agent:      str         # agent name
    action:     str         # 'BUY' | 'SELL' | 'HOLD' | 'REDUCE' | 'CLOSE'
    confidence: float       # 0.0 – 1.0
    reasoning:  str
    data:       dict = field(default_factory=dict)
    timestamp:  str  = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class OrchestratorDecision:
    final_action:   str         # consensus action
    confidence:     float
    consensus_pct:  float       # e.g. 0.75 = 3/4 agents agree
    agent_votes:    Dict[str, AgentSignal]
    reasoning:      str
    sl_price:       Optional[float] = None
    tp_price:       Optional[float] = None
    position_size:  Optional[float] = None
    timestamp:      str = field(default_factory=lambda: datetime.now().isoformat())


# ─────────────────────────────────────────────────────────────
# BASE AGENT
# ─────────────────────────────────────────────────────────────

class BaseAgent:
    def __init__(self, name: str):
        self.name    = name
        self._lock   = threading.Lock()
        self._result: Optional[AgentSignal] = None

    def analyze(self, context: dict) -> AgentSignal:
        raise NotImplementedError

    def run_async(self, context: dict,
                  callback: Callable[[AgentSignal], None]) -> None:
        def _run():
            try:
                signal = self.analyze(context)
                with self._lock:
                    self._result = signal
                if callback:
                    callback(signal)
            except Exception as e:
                log.error(f"[{self.name}] Error: {e}")
                fallback = AgentSignal(agent=self.name, action="HOLD",
                                       confidence=0.0, reasoning=f"Error: {e}")
                with self._lock:
                    self._result = fallback
                if callback:
                    callback(fallback)
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t

    @property
    def result(self) -> Optional[AgentSignal]:
        with self._lock:
            return self._result


# ─────────────────────────────────────────────────────────────
# MARKET ANALYST AGENT
# ─────────────────────────────────────────────────────────────

class MarketAnalystAgent(BaseAgent):
    """Analyzes technical indicators + ICT signals."""

    def __init__(self, ict_engine, sr_engine):
        super().__init__("MarketAnalyst")
        self.ict = ict_engine
        self.sr  = sr_engine

    def analyze(self, context: dict) -> AgentSignal:
        df    = context.get("df")
        price = context.get("price", 0)
        feats = context.get("features", {})

        if df is None or len(df) < 50:
            return AgentSignal(agent=self.name, action="HOLD",
                               confidence=0.3, reasoning="Insufficient data")

        # ICT analysis
        ict_signal = self.ict.analyze(df, price)
        sr_analysis = self.sr.analyze(df, price)

        # RSI signal
        rsi = feats.get("rsi", 50)
        rsi_signal = "BUY" if rsi < 30 else "SELL" if rsi > 70 else "HOLD"

        # Combine signals
        signals = []
        if ict_signal.action == "LONG":  signals.append("BUY")
        elif ict_signal.action == "SHORT": signals.append("SELL")
        else:                              signals.append("HOLD")
        signals.append(rsi_signal)

        # MACD
        macd_hist = feats.get("macd_hist", 0)
        signals.append("BUY" if macd_hist > 0 else "SELL" if macd_hist < -0.001 else "HOLD")

        buy_count  = signals.count("BUY")
        sell_count = signals.count("SELL")

        if buy_count >= 2:
            action = "BUY"; confidence = buy_count / len(signals)
        elif sell_count >= 2:
            action = "SELL"; confidence = sell_count / len(signals)
        else:
            action = "HOLD"; confidence = 0.4

        near_res = sr_analysis.near_resistance
        if near_res and action == "BUY":
            confidence *= 0.75

        reasoning = (
            f"ICT:{ict_signal.action}({ict_signal.confidence:.0%}) | "
            f"RSI:{rsi:.1f}({rsi_signal}) | "
            f"MACD:{'POS' if macd_hist > 0 else 'NEG'} | "
            f"NearRes:{near_res} | "
            f"SR: sup=${sr_analysis.nearest_support:.0f} res=${sr_analysis.nearest_resistance:.0f}"
        )

        return AgentSignal(
            agent=self.name, action=action,
            confidence=round(confidence, 3),
            reasoning=reasoning,
            data={
                "ict":    {"action": ict_signal.action, "conf": ict_signal.confidence},
                "sr":     {"support": sr_analysis.nearest_support,
                           "resistance": sr_analysis.nearest_resistance},
                "rsi":    rsi,
                "partial_exits": sr_analysis.partial_exit_levels,
            }
        )


# ─────────────────────────────────────────────────────────────
# ML PREDICTION AGENT
# ─────────────────────────────────────────────────────────────

class MLPredictionAgent(BaseAgent):
    """Runs LSTM + RF models and XGBoost ensemble."""

    def __init__(self, lstm_model, rf_model):
        super().__init__("MLPrediction")
        self.lstm = lstm_model
        self.rf   = rf_model

    def analyze(self, context: dict) -> AgentSignal:
        df    = context.get("df")
        if df is None:
            return AgentSignal(agent=self.name, action="HOLD",
                               confidence=0.3, reasoning="No data")

        lstm_result = self.lstm.tahmin(df)
        rf_result   = self.rf.tahmin(df)

        # Vote
        lstm_action = ("BUY" if lstm_result["yon"] == "yukari" else
                       "SELL" if lstm_result["yon"] == "asagi" else "HOLD")
        rf_action   = rf_result["sinyal"]

        votes = [lstm_action, rf_action]
        buy_v = votes.count("BUY"); sell_v = votes.count("SELL")

        avg_conf = (lstm_result["guven"] + rf_result["guven"]) / 2

        if buy_v == 2:    action = "BUY";  confidence = avg_conf
        elif sell_v == 2: action = "SELL"; confidence = avg_conf
        elif buy_v > sell_v: action = "BUY"; confidence = avg_conf * 0.7
        elif sell_v > buy_v: action = "SELL"; confidence = avg_conf * 0.7
        else:              action = "HOLD"; confidence = 0.35

        reasoning = (
            f"LSTM:{lstm_action}({lstm_result['guven']:.0%}) "
            f"backend:{lstm_result.get('backend','?')} | "
            f"RF:{rf_action}({rf_result['guven']:.0%})"
        )

        return AgentSignal(
            agent=self.name, action=action,
            confidence=round(confidence, 3),
            reasoning=reasoning,
            data={"lstm": lstm_result, "rf": rf_result}
        )


# ─────────────────────────────────────────────────────────────
# NEWS & SENTIMENT AGENT
# ─────────────────────────────────────────────────────────────

class NewsSentimentAgent(BaseAgent):
    """Analyzes news sentiment + macro context."""

    def __init__(self, memory_manager=None, macro_engine=None):
        super().__init__("NewsSentiment")
        self.memory = memory_manager
        self.macro  = macro_engine

    def analyze(self, context: dict) -> AgentSignal:
        fear_greed = context.get("fear_greed", {})
        news       = context.get("news", {})
        macro      = {}

        if self.macro:
            try:
                macro = self.macro.fetch_all()
            except Exception:
                pass

        fg_score    = fear_greed.get("score", 50)
        news_score  = news.get("score", 0)
        fg_trend    = macro.get("fear_greed_trend", "stable")
        btc_7d      = macro.get("btc_change_7d", 0)
        funding     = macro.get("funding_sentiment", "neutral")

        score = 0.0
        reasons = []

        # Fear & Greed contrarian
        if fg_score < 25:
            score += 0.25; reasons.append(f"Extreme Fear({fg_score}) → contrarian BUY")
        elif fg_score > 75:
            score -= 0.25; reasons.append(f"Extreme Greed({fg_score}) → caution")
        elif fg_score < 40:
            score += 0.10; reasons.append(f"Fear({fg_score}) → mild bullish")

        # News sentiment
        if news_score > 0.3:
            score += 0.15; reasons.append(f"News positive({news_score:.2f})")
        elif news_score < -0.3:
            score -= 0.15; reasons.append(f"News negative({news_score:.2f})")

        # Funding rate
        if funding == "extremely_long":
            score -= 0.15; reasons.append("Funding: longs crowded → risk")
        elif funding == "short_heavy":
            score += 0.10; reasons.append("Funding: shorts heavy → squeeze risk")

        # F&G trend
        if fg_trend == "improving_fast":
            score += 0.10; reasons.append("F&G improving fast")
        elif fg_trend == "deteriorating_fast":
            score -= 0.10; reasons.append("F&G deteriorating fast")

        confidence = min(0.85, abs(score) * 2 + 0.3)

        if score >= 0.20:    action = "BUY"
        elif score <= -0.20: action = "SELL"
        else:                action = "HOLD"

        return AgentSignal(
            agent=self.name, action=action,
            confidence=round(confidence, 3),
            reasoning=" | ".join(reasons) or "Neutral sentiment",
            data={"fear_greed": fg_score, "news_score": news_score,
                  "funding": funding, "macro_bias": macro.get("macro_summary","")}
        )


# ─────────────────────────────────────────────────────────────
# PORTFOLIO MANAGER AGENT
# ─────────────────────────────────────────────────────────────

class PortfolioManagerAgent(BaseAgent):
    """Manages risk, position sizing, drawdown."""

    def __init__(self, risk_manager, memory_manager=None):
        super().__init__("PortfolioManager")
        self.risk   = risk_manager
        self.memory = memory_manager

    def analyze(self, context: dict) -> AgentSignal:
        balance        = context.get("balance", 0)
        open_positions = context.get("open_positions", [])
        daily_pnl_pct  = context.get("daily_pnl_pct", 0)
        symbol         = context.get("symbol", "")

        reasons = []
        action  = "BUY"  # default allow
        confidence = 0.75

        # Daily loss check
        if daily_pnl_pct < -0.04:
            action = "HOLD"; confidence = 0.9
            reasons.append(f"Daily loss {daily_pnl_pct:.1%} approaching limit → HOLD")
            return AgentSignal(agent=self.name, action=action,
                               confidence=confidence, reasoning=" | ".join(reasons))

        # Max positions
        if len(open_positions) >= 2:
            action = "HOLD"; confidence = 0.85
            reasons.append(f"Max positions reached ({len(open_positions)})")
            return AgentSignal(agent=self.name, action=action,
                               confidence=confidence, reasoning=" | ".join(reasons))

        # Correlation check (avoid BTC+ETH simultaneously)
        open_syms = [p.get("symbol","") for p in open_positions]
        if symbol in ("BTCUSDT","ETHUSDT") and any(s in ("BTCUSDT","ETHUSDT") for s in open_syms):
            action = "HOLD"; confidence = 0.7
            reasons.append("Correlation filter: BTC/ETH already open")
            return AgentSignal(agent=self.name, action=action,
                               confidence=confidence, reasoning=" | ".join(reasons))

        # Memory win rate check
        if self.memory:
            try:
                stats = self.memory.get_stats()
                wr = stats["all_time"]["win_rate"]
                if wr < 0.40 and stats["all_time"]["total_trades"] > 10:
                    confidence *= 0.7
                    reasons.append(f"Win rate low ({wr:.0%}) → reducing confidence")
            except Exception:
                pass

        # Position sizing
        from risk.manager import RiskYoneticisi
        try:
            size = self.risk.pozisyon_boyutu(0.7, 0.4, balance)
        except Exception:
            size = balance * 0.05

        reasons.append(f"Portfolio OK | Size: ${size:.0f} | Balance: ${balance:.0f}")

        return AgentSignal(
            agent=self.name, action=action,
            confidence=confidence,
            reasoning=" | ".join(reasons),
            data={"position_size": size, "daily_pnl_pct": daily_pnl_pct}
        )


# ─────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Coordinates all agents, collects votes, reaches consensus.
    Timeout: 15s per analysis cycle.
    """

    TIMEOUT = 15.0

    def __init__(self,
                 market_agent:    MarketAnalystAgent,
                 ml_agent:        MLPredictionAgent,
                 news_agent:      NewsSentimentAgent,
                 portfolio_agent: PortfolioManagerAgent,
                 claude_advisor=None,
                 on_thought:      Callable = None):
        self.agents = {
            "market":    market_agent,
            "ml":        ml_agent,
            "news":      news_agent,
            "portfolio": portfolio_agent,
        }
        self.claude   = claude_advisor
        self.on_thought = on_thought
        self._thought_log = []

    def _log(self, icon: str, text: str):
        if self.on_thought:
            try:
                self.on_thought(icon, text)
            except Exception:
                pass
        log.info(f"[Orchestrator] {icon} {text}")

    def analyze(self, symbol: str, context: dict) -> OrchestratorDecision:
        """
        Run all agents in parallel, collect results, reach consensus.
        """
        self._log("🧠", f"Orchestrator analysis: {symbol}")
        start = time.time()

        results: Dict[str, Optional[AgentSignal]] = {k: None for k in self.agents}
        events = {k: threading.Event() for k in self.agents}

        def make_callback(name):
            def cb(signal: AgentSignal):
                results[name] = signal
                events[name].set()
                self._log(
                    "✅" if signal.action != "HOLD" else "💭",
                    f"{name}: {signal.action} ({signal.confidence:.0%}) — {signal.reasoning[:80]}"
                )
            return cb

        # Launch all agents in parallel
        threads = []
        for name, agent in self.agents.items():
            t = agent.run_async(context, make_callback(name))
            threads.append(t)

        # Wait for all with timeout
        remaining = self.TIMEOUT
        for name, evt in events.items():
            t_wait = max(0.1, remaining - (time.time() - start))
            evt.wait(timeout=t_wait)
            if results[name] is None:
                results[name] = AgentSignal(
                    agent=name, action="HOLD",
                    confidence=0.3, reasoning="Timeout"
                )
                self._log("⚠️", f"{name} timed out — defaulting to HOLD")

        # Claude final layer (if available)
        claude_signal = None
        if self.claude:
            try:
                claude_dec = self.claude(
                    symbol=symbol,
                    fiyat=context.get("price", 0),
                    ozellikler=context.get("features", {}),
                    lstm_sonuc=results["ml"].data.get("lstm", {}) if results["ml"] else {},
                    rf_sonuc=results["ml"].data.get("rf", {}) if results["ml"] else {},
                    ob_analiz=context.get("order_book", {}),
                    fear_greed=context.get("fear_greed", {}),
                    haber=context.get("news", {}),
                )
                claude_action = claude_dec.get("karar", "BEKLE")
                if claude_action == "AL":   claude_action = "BUY"
                elif claude_action == "SAT": claude_action = "SELL"
                else:                       claude_action = "HOLD"
                claude_signal = AgentSignal(
                    agent="claude", action=claude_action,
                    confidence=claude_dec.get("guven", 0.5),
                    reasoning=claude_dec.get("gerekceler",""),
                )
                results["claude"] = claude_signal
                self._log("🤖", f"Claude: {claude_action} ({claude_dec.get('guven',0):.0%})")
            except Exception as e:
                log.warning(f"[Orchestrator] Claude failed: {e}")

        # Consensus
        decision = self._consensus(symbol, results, context)
        elapsed = time.time() - start
        self._log("⚡", f"Decision: {decision.final_action} ({decision.confidence:.0%}) | {elapsed:.1f}s | Consensus: {decision.consensus_pct:.0%}")
        return decision

    def _consensus(self, symbol: str, results: dict,
                   context: dict) -> OrchestratorDecision:
        """Weighted voting consensus."""
        weights = {"market": 0.30, "ml": 0.25, "news": 0.20,
                   "portfolio": 0.15, "claude": 0.10}

        buy_weight  = 0.0
        sell_weight = 0.0
        hold_weight = 0.0

        # Portfolio agent veto
        pm = results.get("portfolio")
        if pm and pm.action == "HOLD" and pm.confidence >= 0.85:
            reasoning = f"Portfolio veto: {pm.reasoning}"
            return OrchestratorDecision(
                final_action="HOLD", confidence=pm.confidence,
                consensus_pct=0.0, agent_votes=results,
                reasoning=reasoning,
                position_size=pm.data.get("position_size", 50),
            )

        total_weight = 0.0
        for name, signal in results.items():
            if signal is None: continue
            w = weights.get(name, 0.10) * signal.confidence
            if signal.action == "BUY":   buy_weight  += w
            elif signal.action == "SELL": sell_weight += w
            else:                         hold_weight += w
            total_weight += w

        if total_weight == 0:
            return OrchestratorDecision(
                final_action="HOLD", confidence=0.3,
                consensus_pct=0.0, agent_votes=results,
                reasoning="No valid signals"
            )

        buy_pct  = buy_weight  / total_weight
        sell_pct = sell_weight / total_weight

        if buy_pct >= 0.50:
            action = "BUY"; confidence = buy_pct; consensus = buy_pct
        elif sell_pct >= 0.50:
            action = "SELL"; confidence = sell_pct; consensus = sell_pct
        else:
            action = "HOLD"; confidence = 0.4; consensus = 0.0

        # Get SL/TP from market agent
        ma = results.get("market")
        sl_price = tp_price = None
        if ma and action != "HOLD":
            price = context.get("price", 0)
            atr   = context.get("features", {}).get("atr_14", price * 0.005)
            if action == "BUY":
                sl_price = price - atr * 2.0
                tp_price = price + atr * 3.5
            else:
                sl_price = price + atr * 2.0
                tp_price = price - atr * 3.5

        pm_size = results.get("portfolio")
        pos_size = pm_size.data.get("position_size", 50) if pm_size and pm_size.data else 50

        reasoning_parts = []
        for name, sig in results.items():
            if sig:
                reasoning_parts.append(f"{name}:{sig.action}({sig.confidence:.0%})")

        return OrchestratorDecision(
            final_action=action,
            confidence=round(confidence, 3),
            consensus_pct=round(consensus, 3),
            agent_votes=results,
            reasoning=" | ".join(reasoning_parts),
            sl_price=round(sl_price, 4) if sl_price else None,
            tp_price=round(tp_price, 4) if tp_price else None,
            position_size=round(pos_size, 2),
        )
