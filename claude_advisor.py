"""
ai_engine/claude_advisor.py
============================
Claude API (claude-sonnet-4-6) — Final Decision Maker
Now enriched with:
  - Long-term bot memory (strategy performance, events, learnings)
  - Macro context (BTC dominance, F&G trend, halving phase, funding)
  - Periodic regime analysis
"""

import json
import logging
import anthropic
from typing import Optional

log = logging.getLogger("CryptoBot.Claude")

_client: Optional[anthropic.Anthropic] = None

# Memory + macro injected at runtime (set by crypto_bot.py)
_memory_manager = None
_macro_engine   = None


def istemci_baslat(api_key: str) -> None:
    global _client
    _client = anthropic.Anthropic(api_key=api_key)
    log.info("✅ Claude API client initialized")


def set_memory(memory_manager, macro_engine) -> None:
    """Inject memory + macro engines (called once at bot startup)."""
    global _memory_manager, _macro_engine
    _memory_manager = memory_manager
    _macro_engine   = macro_engine
    log.info("✅ Claude advisor: Memory + Macro context enabled")


# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────

_SISTEM_PROMPT = """You are a professional crypto trading AI advisor.
You receive technical indicators, AI model outputs, market data,
long-term bot memory, AND current macro context.

Your job: Synthesize ALL of this into the best possible trade decision.

DECISION FRAMEWORK:
1. Read MEMORY CONTEXT first — understand recent performance and market history
2. Check MACRO CONTEXT — is the broader environment bullish or bearish?
3. Analyze TECHNICAL signals — RSI, MACD, EMA, ICT, order flow
4. Weigh MODEL OUTPUTS — LSTM probability, RF signal, confidence levels
5. Cross-reference with NEWS sentiment and Fear & Greed
6. Make FINAL DECISION

MEMORY-AWARE RULES:
- If win rate dropped below 50% recently → be more conservative (WAIT more)
- If a specific strategy is consistently losing → reduce its weight
- If macro is extreme fear + RSI oversold → stronger BUY bias (contrarian)
- If halving approaching (< 180 days) → be slightly more bullish long-term
- If funding rate extremely positive → longs are crowded, be careful with BUY
- If bot has been on a losing streak → reduce position size suggestion

CONFIDENCE RULES:
- All signals agree → high confidence (0.75+)
- Mixed signals → medium confidence (0.50–0.65)
- Conflicting signals → low confidence (< 0.50) → force WAIT
- Risk score > 0.80 → always return WAIT regardless

Respond ONLY in this exact JSON format, nothing else:
{
  "karar": "AL" or "SAT" or "BEKLE",
  "guven": 0.0 to 1.0,
  "risk_skoru": 0.0 to 1.0,
  "gerekceler": "Turkish reasoning (max 200 chars)",
  "onerilen_sl_pct": number or null,
  "onerilen_tp_pct": number or null,
  "regime_gozlem": "Brief market regime observation in Turkish (max 100 chars)",
  "ogrenme": "Any new learning to store in memory (max 100 chars) or null"
}"""


# ─────────────────────────────────────────────────────────────
# MAIN DECISION FUNCTION
# ─────────────────────────────────────────────────────────────

def karar_al(
    symbol:        str,
    fiyat:         float,
    ozellikler:    dict,
    lstm_sonuc:    dict,
    rf_sonuc:      dict,
    ob_analiz:     dict,
    fear_greed:    dict,
    haber:         dict,
    acik_pozisyon: Optional[dict] = None,
) -> dict:
    """
    Final trade decision with memory + macro awareness.
    Returns: {karar, guven, risk_skoru, gerekceler, onerilen_sl, onerilen_tp, ham_yanit}
    """
    if _client is None:
        log.warning("Claude API not initialized — returning WAIT")
        return _empty_decision("API_NOT_INITIALIZED")

    # Build memory context (empty string if memory not available)
    memory_ctx = ""
    if _memory_manager:
        try:
            memory_ctx = _memory_manager.build_context()
        except Exception as e:
            log.warning(f"Memory context failed: {e}")

    # Build macro context
    macro_ctx = ""
    if _macro_engine:
        try:
            macro_data = _macro_engine.fetch_all()
            macro_ctx  = macro_data.get("macro_summary", "")
        except Exception as e:
            log.warning(f"Macro context failed: {e}")

    # Build full prompt
    prompt = _build_prompt(
        symbol, fiyat, ozellikler, lstm_sonuc, rf_sonuc,
        ob_analiz, fear_greed, haber, acik_pozisyon,
        memory_ctx, macro_ctx,
    )

    try:
        msg = _client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            system=_SISTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = msg.content[0].text.strip()
        log.debug(f"[Claude] Response: {response_text[:200]}")

        result = _parse_response(response_text, fiyat)

        # Store learning in memory if provided
        if _memory_manager and result.get("ogrenme"):
            try:
                _memory_manager.add_learning(
                    result["ogrenme"],
                    confidence=result.get("guven", 0.6),
                )
            except Exception:
                pass

        return result

    except anthropic.RateLimitError:
        log.warning("Claude rate limit — returning WAIT")
        return _empty_decision("RATE_LIMIT")
    except anthropic.APIStatusError as e:
        log.error(f"Claude API error: {e}")
        return _empty_decision(f"API_ERROR_{e.status_code}")
    except Exception as e:
        log.error(f"Claude unexpected error: {e}")
        return _empty_decision("UNKNOWN_ERROR")


# ─────────────────────────────────────────────────────────────
# REGIME ANALYSIS (periodic — every N hours)
# ─────────────────────────────────────────────────────────────

def analiz_rejim(piyasa_ozeti: dict) -> str:
    """
    Ask Claude to assess current market regime.
    Called periodically (every 6h) to update memory.
    Returns regime string.
    """
    if _client is None:
        return "unknown"

    prompt = f"""Based on this market data, what is the current market regime?
Data: {json.dumps(piyasa_ozeti, ensure_ascii=False)[:1000]}

Respond with ONLY one of these words:
bullish | bearish | ranging | volatile | accumulation | distribution"""

    try:
        msg = _client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=20,
            messages=[{"role": "user", "content": prompt}],
        )
        regime = msg.content[0].text.strip().lower()
        valid  = {"bullish","bearish","ranging","volatile","accumulation","distribution"}
        return regime if regime in valid else "unknown"
    except Exception as e:
        log.warning(f"Regime analysis failed: {e}")
        return "unknown"


# ─────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────

def _build_prompt(symbol, fiyat, ozellik, lstm, rf, ob, fg, haber,
                  acik_poz, memory_ctx, macro_ctx) -> str:

    acik_str = "NONE"
    if acik_poz:
        kar = (fiyat - acik_poz["giris_fiyati"]) / acik_poz["giris_fiyati"] * 100
        acik_str = (f"Entry: ${acik_poz['giris_fiyati']:.2f} | "
                    f"Qty: {acik_poz['miktar']} | "
                    f"Current PnL: {kar:+.2f}%")

    return f"""
{memory_ctx}

MACRO CONTEXT: {macro_ctx or 'Not available'}

## {symbol} — Trade Analysis

### Price: ${fiyat:.4f} USDT

### Technical Indicators
- RSI(14): {ozellik.get('rsi', 'N/A')} | RSI(21): {ozellik.get('rsi_21', 'N/A')}
- MACD: {ozellik.get('macd', 'N/A')} | Signal: {ozellik.get('macd_signal', 'N/A')} | Hist: {ozellik.get('macd_hist', 'N/A')}
- EMA Trend: {ozellik.get('ema_trend', 'N/A')} | Golden Cross: {ozellik.get('golden_cross', 'N/A')}
- Bollinger Pos: {ozellik.get('bb_pos', 'N/A')} | Width: {ozellik.get('bb_width', 'N/A')}
- Stochastic %K: {ozellik.get('stoch_k', 'N/A')} | Williams %R: {ozellik.get('williams_r', 'N/A')}
- Momentum: {ozellik.get('momentum', 'N/A')} | Volume Ratio: {ozellik.get('volume_ratio', 'N/A')}x
- ATR(14): {ozellik.get('atr_14', 'N/A')} | Volatility: {ozellik.get('volatility', 'N/A')}%
- VWAP: {ozellik.get('vwap', 'N/A')} | OBV: {ozellik.get('obv', 'N/A')}

### LSTM Model
- Direction: {lstm.get('yon', 'N/A')} | Probability: {lstm.get('olasilik', 'N/A')}
- Confidence: {lstm.get('guven', 'N/A')} | Backend: {lstm.get('backend', 'N/A')}

### Random Forest
- Signal: {rf.get('sinyal', 'N/A')} | Confidence: {rf.get('guven', 'N/A')}
- Probabilities → BUY:{rf.get('olasiliklar',{}).get('AL','N/A')} SELL:{rf.get('olasiliklar',{}).get('SAT','N/A')} HOLD:{rf.get('olasiliklar',{}).get('BEKLE','N/A')}

### Order Book
- Bid/Ask Ratio: {ob.get('bid_ask_ratio', 'N/A')} | Buy Pressure: {ob.get('buy_pressure', 'N/A')}%
- Spread: {ob.get('spread_pct', 'N/A')}%

### Fear & Greed
- Score: {fg.get('score', 'N/A')}/100 | Label: {fg.get('label', 'N/A')}

### News Sentiment
- Score: {haber.get('score', 'N/A')} | Label: {haber.get('label', 'N/A')}

### Open Position
{acik_str}

Synthesize all data including MEMORY and MACRO context. Return JSON decision.
""".strip()


# ─────────────────────────────────────────────────────────────
# RESPONSE PARSER
# ─────────────────────────────────────────────────────────────

def _parse_response(text: str, fiyat: float) -> dict:
    try:
        clean = text.strip()
        if clean.startswith("```"):
            parts = clean.split("```")
            clean = parts[1] if len(parts) > 1 else clean
            if clean.startswith("json"):
                clean = clean[4:]
        clean = clean.strip()

        data = json.loads(clean)
        karar = str(data.get("karar", "BEKLE")).upper()
        if karar not in ("AL", "SAT", "BEKLE"):
            karar = "BEKLE"

        guven      = float(data.get("guven", 0.5))
        risk_skoru = float(data.get("risk_skoru", 0.5))
        gerekceler = str(data.get("gerekceler", ""))
        ogrenme    = data.get("ogrenme")
        regime_obs = data.get("regime_gozlem", "")

        sl_pct = data.get("onerilen_sl_pct")
        tp_pct = data.get("onerilen_tp_pct")
        onerilen_sl = fiyat * (1 - float(sl_pct) / 100) if sl_pct else None
        onerilen_tp = fiyat * (1 + float(tp_pct) / 100) if tp_pct else None

        # Safety filters
        if guven < 0.50 or risk_skoru > 0.80:
            karar = "BEKLE"

        return {
            "karar":       karar,
            "guven":       round(guven, 4),
            "risk_skoru":  round(risk_skoru, 4),
            "gerekceler":  gerekceler,
            "regime_gozlem": regime_obs,
            "ogrenme":     ogrenme,
            "onerilen_sl": onerilen_sl,
            "onerilen_tp": onerilen_tp,
            "ham_yanit":   text,
        }

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        log.warning(f"Response parse error: {e} | Text: {text[:200]}")
        return _empty_decision("PARSE_ERROR")


def _empty_decision(reason: str) -> dict:
    return {
        "karar": "BEKLE", "guven": 0.0, "risk_skoru": 1.0,
        "gerekceler": f"Decision failed: {reason}",
        "regime_gozlem": "", "ogrenme": None,
        "onerilen_sl": None, "onerilen_tp": None, "ham_yanit": "",
    }


# ─────────────────────────────────────────────────────────────
# VOTING SYSTEM (unchanged — backward compatible)
# ─────────────────────────────────────────────────────────────

def oylama_sistemi(lstm_yon: str, rf_sinyal: str, claude_karar: str) -> dict:
    """3-model voting. 2/3 majority → decision valid."""
    lstm_karar = "AL" if lstm_yon == "yukari" else "SAT" if lstm_yon == "asagi" else "BEKLE"
    kararlar   = [lstm_karar, rf_sinyal, claude_karar]
    al_oylari  = kararlar.count("AL")
    sat_oylari = kararlar.count("SAT")

    if al_oylari >= 2:   final, oy = "AL",    al_oylari
    elif sat_oylari >= 2: final, oy = "SAT",   sat_oylari
    else:                 final, oy = "BEKLE", 1

    return {
        "final_karar":   final,
        "oylama":        {"AL": al_oylari, "SAT": sat_oylari,
                          "BEKLE": 3 - al_oylari - sat_oylari},
        "mutabakat_pct": round(oy / 3, 4),
        "bireysel":      {"lstm": lstm_karar, "rf": rf_sinyal, "claude": claude_karar},
    }
