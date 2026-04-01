"""
ai_engine/xai_explainer.py
==========================
Explainable AI (XAI) Module
Answers: "Why did the bot decide to BUY/SELL?"
Methods: Feature Importance · Decision Breakdown · Confidence Attribution
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional

log = logging.getLogger("CryptoBot.XAI")


@dataclass
class FeatureContribution:
    feature:      str
    value:        float
    contribution: float     # -1.0 (bearish) to +1.0 (bullish)
    direction:    str       # 'bullish' | 'bearish' | 'neutral'
    weight:       float     # relative importance 0-1


@dataclass
class DecisionExplanation:
    action:           str
    confidence:       float
    main_reason:      str               # Primary reason in plain English/Turkish
    supporting:       List[str]         # Supporting factors
    against:          List[str]         # Opposing factors
    feature_breakdown: List[FeatureContribution]
    risk_factors:     List[str]
    score_breakdown:  Dict[str, float]  # each component's contribution


class XAIExplainer:
    """
    Generates human-readable explanations for bot decisions.
    Works by analyzing feature contributions and agent votes.
    """

    # Feature display names and their directional rules
    FEATURE_RULES = {
        "rsi":           {"name": "RSI",        "low": 30, "high": 70,  "low_dir": "bullish", "high_dir": "bearish"},
        "macd_hist":     {"name": "MACD Hist",  "low": 0,  "high": 0,   "low_dir": "bearish", "high_dir": "bullish"},
        "bb_pos":        {"name": "BB Position","low": 0.2,"high": 0.8, "low_dir": "bullish", "high_dir": "bearish"},
        "stoch_k":       {"name": "Stochastic", "low": 20, "high": 80,  "low_dir": "bullish", "high_dir": "bearish"},
        "volume_ratio":  {"name": "Vol Ratio",  "low": 0.8,"high": 2.0, "low_dir": "bearish", "high_dir": "bullish"},
        "volatility":    {"name": "Volatility", "low": 0,  "high": 5.0, "low_dir": "neutral", "high_dir": "bearish"},
        "ema_trend":     {"name": "EMA Trend",  "low": 0,  "high": 1,   "low_dir": "bearish", "high_dir": "bullish"},
        "momentum":      {"name": "Momentum",   "low": 0,  "high": 0,   "low_dir": "bearish", "high_dir": "bullish"},
        "williams_r":    {"name": "Williams %R","low": -80,"high": -20, "low_dir": "bullish", "high_dir": "bearish"},
    }

    def explain(self, decision_action: str, decision_confidence: float,
                features: dict, agent_votes: dict,
                ict_signal=None, sr_analysis=None,
                lang: str = "en") -> DecisionExplanation:
        """
        Generate full explanation for a trading decision.
        """

        contributions = self._analyze_features(features)
        score_breakdown = self._score_breakdown(contributions)
        main_reason, supporting, against = self._build_narrative(
            decision_action, contributions, agent_votes, ict_signal, lang
        )
        risk_factors = self._identify_risks(features, sr_analysis, lang)

        return DecisionExplanation(
            action=decision_action,
            confidence=decision_confidence,
            main_reason=main_reason,
            supporting=supporting,
            against=against,
            feature_breakdown=sorted(contributions,
                                      key=lambda x: abs(x.contribution),
                                      reverse=True)[:8],
            risk_factors=risk_factors,
            score_breakdown=score_breakdown,
        )

    def _analyze_features(self, features: dict) -> List[FeatureContribution]:
        contributions = []

        for feat_key, rules in self.FEATURE_RULES.items():
            val = features.get(feat_key)
            if val is None:
                continue

            val = float(val)
            low   = rules["low"]
            high  = rules["high"]
            range_size = abs(high - low) if high != low else 1.0

            # Normalize contribution
            if high > low:
                if val <= low:
                    contrib = -0.8 if rules["low_dir"] == "bearish" else 0.8
                    direction = rules["low_dir"]
                elif val >= high:
                    contrib = 0.8 if rules["high_dir"] == "bullish" else -0.8
                    direction = rules["high_dir"]
                else:
                    normalized = (val - low) / range_size
                    if rules["high_dir"] == "bullish":
                        contrib = (normalized - 0.5) * 1.6
                    else:
                        contrib = (0.5 - normalized) * 1.6
                    direction = "bullish" if contrib > 0 else "bearish" if contrib < 0 else "neutral"
            else:
                contrib = 0.1 if val > 0 else -0.1
                direction = "bullish" if contrib > 0 else "bearish"

            contrib = max(-1.0, min(1.0, contrib))

            contributions.append(FeatureContribution(
                feature=rules["name"],
                value=round(val, 4),
                contribution=round(contrib, 3),
                direction=direction,
                weight=abs(contrib),
            ))

        return contributions

    def _score_breakdown(self, contributions: List[FeatureContribution]) -> Dict[str, float]:
        total_bull = sum(c.contribution for c in contributions if c.contribution > 0)
        total_bear = abs(sum(c.contribution for c in contributions if c.contribution < 0))
        net = total_bull - total_bear
        return {
            "bullish_score": round(total_bull, 3),
            "bearish_score": round(total_bear, 3),
            "net_score":     round(net, 3),
            "signal_count":  len(contributions),
        }

    def _build_narrative(self, action: str, contributions: list,
                          agent_votes: dict, ict_signal,
                          lang: str) -> tuple:
        # Sort by magnitude
        bullish = [c for c in contributions if c.direction == "bullish"]
        bearish = [c for c in contributions if c.direction == "bearish"]

        if lang == "tr":
            if action == "BUY":
                main = self._build_main_tr_buy(bullish, agent_votes, ict_signal)
            elif action == "SELL":
                main = self._build_main_tr_sell(bearish, agent_votes, ict_signal)
            else:
                main = "Birden fazla çelişkili sinyal — bekle sinyali oluştu."
            supporting = [f"{c.feature}: {c.value:.2f} → Yükseliş desteği" for c in bullish[:3]]
            against    = [f"{c.feature}: {c.value:.2f} → Düşüş baskısı" for c in bearish[:2]]
        else:
            if action == "BUY":
                main = self._build_main_en_buy(bullish, agent_votes, ict_signal)
            elif action == "SELL":
                main = self._build_main_en_sell(bearish, agent_votes, ict_signal)
            else:
                main = "Conflicting signals from multiple sources — HOLD decision reached."
            supporting = [f"{c.feature}: {c.value:.2f} → Bullish support" for c in bullish[:3]]
            against    = [f"{c.feature}: {c.value:.2f} → Bearish pressure" for c in bearish[:2]]

        return main, supporting, against

    def _build_main_tr_buy(self, bullish: list, agent_votes: dict, ict) -> str:
        parts = []
        if bullish:
            top = bullish[0]
            parts.append(f"{top.feature} ({top.value:.1f}) aşırı satılmış bölgede")
        if ict and ict.action == "LONG":
            parts.append(f"ICT Bullish OB/FVG teyidi")
        buy_agents = [k for k,v in agent_votes.items() if v and v.action == "BUY"]
        if len(buy_agents) >= 2:
            parts.append(f"{len(buy_agents)} ajan AL oyladı ({', '.join(buy_agents)})")
        return " · ".join(parts) if parts else "Teknik göstergeler AL sinyali üretiyor"

    def _build_main_tr_sell(self, bearish: list, agent_votes: dict, ict) -> str:
        parts = []
        if bearish:
            top = bearish[0]
            parts.append(f"{top.feature} ({top.value:.1f}) aşırı alınmış bölgede")
        if ict and ict.action == "SHORT":
            parts.append("ICT Bearish OB/FVG teyidi")
        sell_agents = [k for k,v in agent_votes.items() if v and v.action == "SELL"]
        if len(sell_agents) >= 2:
            parts.append(f"{len(sell_agents)} ajan SAT oyladı")
        return " · ".join(parts) if parts else "Teknik göstergeler SAT sinyali üretiyor"

    def _build_main_en_buy(self, bullish: list, agent_votes: dict, ict) -> str:
        parts = []
        if bullish:
            top = bullish[0]
            parts.append(f"{top.feature} ({top.value:.1f}) in oversold territory")
        if ict and ict.action == "LONG":
            parts.append("ICT Bullish OB/FVG confirmed")
        buy_agents = [k for k,v in agent_votes.items() if v and v.action == "BUY"]
        if len(buy_agents) >= 2:
            parts.append(f"{len(buy_agents)} agents voted BUY ({', '.join(buy_agents)})")
        return " · ".join(parts) if parts else "Technical indicators generating BUY signal"

    def _build_main_en_sell(self, bearish: list, agent_votes: dict, ict) -> str:
        parts = []
        if bearish:
            top = bearish[0]
            parts.append(f"{top.feature} ({top.value:.1f}) in overbought territory")
        if ict and ict.action == "SHORT":
            parts.append("ICT Bearish OB/FVG confirmed")
        sell_agents = [k for k,v in agent_votes.items() if v and v.action == "SELL"]
        if len(sell_agents) >= 2:
            parts.append(f"{len(sell_agents)} agents voted SELL")
        return " · ".join(parts) if parts else "Technical indicators generating SELL signal"

    def _identify_risks(self, features: dict, sr_analysis, lang: str) -> List[str]:
        risks = []
        is_tr = lang == "tr"

        vol = features.get("volatility", 0)
        if vol > 4:
            risks.append(("Yüksek volatilite — pozisyon boyutunu küçült" if is_tr
                          else "High volatility — reduce position size"))

        vr = features.get("volume_ratio", 1)
        if vr < 0.5:
            risks.append(("Düşük hacim — manipülasyon riski var" if is_tr
                          else "Low volume — manipulation risk present"))

        if sr_analysis and sr_analysis.near_resistance:
            risks.append(("Fiyat dirençe yakın — kısmi çıkış düşünülebilir" if is_tr
                          else "Price near resistance — consider partial exit"))

        rsi = features.get("rsi", 50)
        if 45 < rsi < 55:
            risks.append(("RSI orta bölge — sinyal zayıf" if is_tr
                          else "RSI in midrange — weak signal"))

        return risks

    def to_dict(self, explanation: DecisionExplanation) -> dict:
        return {
            "action":       explanation.action,
            "confidence":   explanation.confidence,
            "main_reason":  explanation.main_reason,
            "supporting":   explanation.supporting,
            "against":      explanation.against,
            "risk_factors": explanation.risk_factors,
            "score":        explanation.score_breakdown,
            "features":     [
                {
                    "name":         f.feature,
                    "value":        f.value,
                    "direction":    f.direction,
                    "contribution": f.contribution,
                }
                for f in explanation.feature_breakdown
            ],
        }

    def format_short(self, explanation: DecisionExplanation, lang: str = "en") -> str:
        """One-line summary for dashboard log."""
        conf_str = f"{explanation.confidence:.0%}"
        if lang == "tr":
            return (f"{explanation.action} | {conf_str} | {explanation.main_reason[:80]}")
        return (f"{explanation.action} | {conf_str} | {explanation.main_reason[:80]}")
