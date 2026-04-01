"""
memory/macro_context.py
=======================
Macro Context Engine — Free APIs only
Sources: CoinGecko · Alternative.me · Binance (funding) · FRED (optional)
Provides: BTC dominance · MCap · Fear&Greed · Funding Rate · Halving countdown
"""

import logging
import requests
import time
from datetime import datetime, timedelta
from typing import Optional

log = logging.getLogger("CryptoBot.Macro")

# Request timeout
TIMEOUT = 8

# BTC next halving (approximate — April 2028)
NEXT_HALVING = datetime(2028, 4, 20)


# ─────────────────────────────────────────────────────────────
# MACRO CONTEXT ENGINE
# ─────────────────────────────────────────────────────────────

class MacroContextEngine:
    """
    Fetches and caches macro-level market data from free APIs.
    Cache duration: 15 minutes (macro data doesn't change by the second)
    """

    CACHE_MINUTES = 15

    def __init__(self, binance_client=None):
        self.client    = binance_client
        self._cache    = {}
        self._cache_ts = {}

    # ── MASTER FETCH ──────────────────────────────────────────

    def fetch_all(self) -> dict:
        """
        Fetch all macro data. Uses cache if fresh enough.
        Returns a dict with all macro metrics.
        """
        result = {}

        result.update(self._fetch_coingecko_global())
        result.update(self._fetch_fear_greed())
        result.update(self._fetch_funding_rate())
        result.update(self._fetch_halving_countdown())
        result.update(self._fetch_bitcoin_metrics())

        result["last_updated"]   = datetime.now().isoformat()
        result["macro_summary"]  = self._build_summary(result)

        return result

    # ── COINGECKO GLOBAL ──────────────────────────────────────

    def _fetch_coingecko_global(self) -> dict:
        """BTC dominance, total market cap, 24h change. Free, no key needed."""
        cache_key = "cg_global"
        cached = self._get_cache(cache_key)
        if cached: return cached

        try:
            r = requests.get(
                "https://api.coingecko.com/api/v3/global",
                timeout=TIMEOUT,
                headers={"User-Agent": "CryptoBot/4.0"},
            )
            r.raise_for_status()
            d = r.json().get("data", {})

            result = {
                "btc_dominance":      round(d.get("market_cap_percentage", {}).get("btc", 0), 2),
                "eth_dominance":      round(d.get("market_cap_percentage", {}).get("eth", 0), 2),
                "total_mcap_usd":     d.get("total_market_cap", {}).get("usd", 0),
                "total_volume_24h":   d.get("total_volume", {}).get("usd", 0),
                "mcap_change_24h":    round(d.get("market_cap_change_percentage_24h_usd", 0), 2),
                "active_coins":       d.get("active_cryptocurrencies", 0),
                "markets":            d.get("markets", 0),
            }
            self._set_cache(cache_key, result)
            log.debug(f"[Macro] CoinGecko: BTC dom {result['btc_dominance']}%")
            return result

        except Exception as e:
            log.warning(f"[Macro] CoinGecko failed: {e}")
            return {
                "btc_dominance": 52.0, "eth_dominance": 17.0,
                "total_mcap_usd": 0, "total_volume_24h": 0,
                "mcap_change_24h": 0, "active_coins": 0, "markets": 0,
            }

    # ── FEAR & GREED ──────────────────────────────────────────

    def _fetch_fear_greed(self) -> dict:
        """Fear & Greed Index + 7-day trend. Free."""
        cache_key = "fear_greed"
        cached = self._get_cache(cache_key)
        if cached: return cached

        try:
            r = requests.get(
                "https://api.alternative.me/fng/?limit=7",
                timeout=TIMEOUT,
            )
            r.raise_for_status()
            data = r.json().get("data", [])

            if not data:
                return self._fg_fallback()

            current  = int(data[0]["value"])
            label    = data[0]["value_classification"]
            week_avg = round(sum(int(d["value"]) for d in data) / len(data), 1)
            trend_7d = current - int(data[-1]["value"]) if len(data) >= 7 else 0

            # Classify trend
            if trend_7d > 10:    fg_trend = "improving_fast"
            elif trend_7d > 3:   fg_trend = "improving"
            elif trend_7d < -10: fg_trend = "deteriorating_fast"
            elif trend_7d < -3:  fg_trend = "deteriorating"
            else:                fg_trend = "stable"

            result = {
                "fear_greed":       current,
                "fear_greed_label": label,
                "fear_greed_7d_avg":week_avg,
                "fear_greed_trend": fg_trend,
                "fear_greed_trend_val": trend_7d,
            }
            self._set_cache(cache_key, result)
            log.debug(f"[Macro] F&G: {current}/100 ({label}) trend:{fg_trend}")
            return result

        except Exception as e:
            log.warning(f"[Macro] Fear&Greed failed: {e}")
            return self._fg_fallback()

    def _fg_fallback(self) -> dict:
        return {
            "fear_greed": 50, "fear_greed_label": "Neutral",
            "fear_greed_7d_avg": 50, "fear_greed_trend": "stable",
            "fear_greed_trend_val": 0,
        }

    # ── FUNDING RATE ──────────────────────────────────────────

    def _fetch_funding_rate(self) -> dict:
        """BTC + ETH funding rates from Binance Futures."""
        cache_key = "funding"
        cached = self._get_cache(cache_key)
        if cached: return cached

        result = {"btc_funding_rate": 0.0, "eth_funding_rate": 0.0,
                  "funding_sentiment": "neutral"}

        if not self.client:
            return result

        try:
            for symbol, key in [("BTCUSDT", "btc_funding_rate"),
                                  ("ETHUSDT", "eth_funding_rate")]:
                try:
                    fr = self.client.futures_funding_rate(symbol=symbol, limit=1)
                    if fr:
                        result[key] = round(float(fr[0]["fundingRate"]) * 100, 5)
                except Exception:
                    pass

            # Sentiment from funding
            avg_fr = (result["btc_funding_rate"] + result["eth_funding_rate"]) / 2
            if avg_fr > 0.05:   result["funding_sentiment"] = "extremely_long"
            elif avg_fr > 0.02: result["funding_sentiment"] = "long_heavy"
            elif avg_fr < -0.02: result["funding_sentiment"] = "short_heavy"
            elif avg_fr < -0.05: result["funding_sentiment"] = "extremely_short"
            else:                result["funding_sentiment"] = "neutral"

            self._set_cache(cache_key, result)
            log.debug(f"[Macro] Funding BTC:{result['btc_funding_rate']:.4f}%")
            return result

        except Exception as e:
            log.warning(f"[Macro] Funding rate failed: {e}")
            return result

    # ── HALVING COUNTDOWN ─────────────────────────────────────

    def _fetch_halving_countdown(self) -> dict:
        """Days until next BTC halving."""
        now   = datetime.now()
        delta = NEXT_HALVING - now
        days  = max(0, delta.days)

        # Classify halving cycle phase
        if days > 365 * 2:   phase = "pre_halving_early"
        elif days > 180:     phase = "pre_halving"
        elif days > 0:       phase = "halving_imminent"
        elif days == 0:      phase = "halving_day"
        else:                phase = "post_halving"

        return {
            "halving_days_remaining": days,
            "halving_date":           NEXT_HALVING.strftime("%Y-%m-%d"),
            "halving_cycle_phase":    phase,
        }

    # ── BITCOIN METRICS (CoinGecko) ───────────────────────────

    def _fetch_bitcoin_metrics(self) -> dict:
        """BTC price, 24h/7d/30d change, ATH distance."""
        cache_key = "btc_metrics"
        cached = self._get_cache(cache_key)
        if cached: return cached

        try:
            r = requests.get(
                "https://api.coingecko.com/api/v3/coins/bitcoin"
                "?localization=false&tickers=false&community_data=false"
                "&developer_data=false",
                timeout=TIMEOUT,
                headers={"User-Agent": "CryptoBot/4.0"},
            )
            r.raise_for_status()
            d = r.json()
            md = d.get("market_data", {})

            price  = md.get("current_price", {}).get("usd", 0)
            ath    = md.get("ath", {}).get("usd", 1)
            ath_pct = round((price / ath - 1) * 100, 2) if ath > 0 else 0

            result = {
                "btc_price":         price,
                "btc_change_24h":    round(md.get("price_change_percentage_24h", 0), 2),
                "btc_change_7d":     round(md.get("price_change_percentage_7d", 0), 2),
                "btc_change_30d":    round(md.get("price_change_percentage_30d", 0), 2),
                "btc_ath_usd":       ath,
                "btc_ath_pct":       ath_pct,   # negative = % below ATH
                "btc_mcap_rank":     d.get("market_cap_rank", 1),
            }
            self._set_cache(cache_key, result)
            log.debug(f"[Macro] BTC ${price:,.0f} | ATH dist: {ath_pct:.1f}%")
            return result

        except Exception as e:
            log.warning(f"[Macro] BTC metrics failed: {e}")
            return {
                "btc_price": 0, "btc_change_24h": 0, "btc_change_7d": 0,
                "btc_change_30d": 0, "btc_ath_usd": 0, "btc_ath_pct": 0,
                "btc_mcap_rank": 1,
            }

    # ── MACRO SUMMARY ─────────────────────────────────────────

    def _build_summary(self, data: dict) -> str:
        """
        Build a human-readable macro summary for Claude.
        Concise, max ~200 chars.
        """
        fg       = data.get("fear_greed", 50)
        fg_label = data.get("fear_greed_label", "Neutral")
        dom      = data.get("btc_dominance", 52)
        mcap_chg = data.get("mcap_change_24h", 0)
        btc_7d   = data.get("btc_change_7d", 0)
        funding  = data.get("funding_sentiment", "neutral")
        phase    = data.get("halving_cycle_phase", "unknown")
        ath_pct  = data.get("btc_ath_pct", 0)
        fg_trend = data.get("fear_greed_trend", "stable")

        # Macro bias determination
        bullish_signals = 0
        bearish_signals = 0

        if fg < 30:   bullish_signals += 2   # extreme fear = contrarian buy
        elif fg > 75: bearish_signals += 2
        if mcap_chg > 2:   bullish_signals += 1
        elif mcap_chg < -2: bearish_signals += 1
        if btc_7d > 5:    bullish_signals += 1
        elif btc_7d < -5: bearish_signals += 1
        if funding == "extremely_long":  bearish_signals += 1
        if funding == "short_heavy":     bullish_signals += 1
        if fg_trend == "improving_fast": bullish_signals += 1
        if fg_trend == "deteriorating_fast": bearish_signals += 1

        if bullish_signals > bearish_signals + 1:   bias = "BULLISH"
        elif bearish_signals > bullish_signals + 1: bias = "BEARISH"
        else:                                        bias = "NEUTRAL"

        return (
            f"Macro bias:{bias} | "
            f"F&G:{fg}/100({fg_label},{fg_trend}) | "
            f"BTCdom:{dom}% | "
            f"MCap24h:{mcap_chg:+.1f}% | "
            f"BTC7d:{btc_7d:+.1f}% | "
            f"Funding:{funding} | "
            f"Halving:{phase} | "
            f"ATH dist:{ath_pct:.1f}%"
        )

    # ── CACHE ─────────────────────────────────────────────────

    def _get_cache(self, key: str) -> Optional[dict]:
        ts = self._cache_ts.get(key)
        if ts and (datetime.now() - ts).seconds < self.CACHE_MINUTES * 60:
            return self._cache.get(key)
        return None

    def _set_cache(self, key: str, data: dict) -> None:
        self._cache[key]    = data
        self._cache_ts[key] = datetime.now()

    def clear_cache(self) -> None:
        self._cache.clear()
        self._cache_ts.clear()
