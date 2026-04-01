"""
data/news_engine.py
====================
Multi-Source News & Sentiment Engine
Sources: CryptoPanic API · Alternative.me · CoinGecko (trending) · RSS feeds
NLP: Keyword-based sentiment scoring (no external NLP dependency)
Event Detection: Hacks · ETF news · Regulation · Halving · Major listings
"""

import logging
import re
import time
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Optional

log = logging.getLogger("CryptoBot.News")

REQUEST_TIMEOUT = 8
CACHE_MINUTES   = 10


@dataclass
class NewsItem:
    title:      str
    body:       str         = ""
    source:     str         = "Unknown"
    url:        str         = ""
    published:  str         = ""
    category:   str         = "neutral"   # 'bullish'|'bearish'|'neutral'|'macro'|'event'
    impact:     str         = "low"       # 'high'|'medium'|'low'
    sentiment:  float       = 0.0         # -1.0 to +1.0
    coins:      list        = field(default_factory=list)
    event_type: str         = ""          # 'hack'|'etf'|'regulation'|'listing'|''
    ai_analysis: str        = ""


@dataclass
class NewsReport:
    items:          List[NewsItem]
    overall_score:  float           # -1.0 to +1.0
    overall_label:  str             # 'very_positive' | 'positive' | 'neutral' | 'negative'
    btc_score:      float
    eth_score:      float
    events:         List[str]       # detected events
    last_updated:   str


# ─────────────────────────────────────────────────────────────
# SENTIMENT KEYWORDS
# ─────────────────────────────────────────────────────────────

POSITIVE_WORDS = {
    "bullish", "surge", "soar", "rally", "pump", "moon", "ath", "all-time high",
    "breakout", "adoption", "approval", "approved", "etf", "institutional",
    "accumulation", "buy", "upgrade", "partnership", "launch", "integration",
    "milestone", "record", "growth", "positive", "optimistic", "support",
    "upgrade", "listing", "recovery", "rebound", "oversold", "undervalued",
    "inflow", "inflows", "halving", "bullrun", "bull run",
}

NEGATIVE_WORDS = {
    "bearish", "crash", "dump", "plunge", "hack", "exploit", "rug pull", "scam",
    "ban", "banned", "regulation", "crackdown", "lawsuit", "sec", "fraud",
    "liquidation", "delisting", "concern", "fear", "panic", "sell-off",
    "outflow", "outflows", "warning", "risk", "volatile", "uncertainty",
    "rejected", "rejection", "fud", "negative", "decline", "drop", "loss",
}

HIGH_IMPACT_WORDS = {
    "etf", "approval", "ban", "hack", "exploit", "fed", "halving", "blackrock",
    "sec", "institutional", "billion", "trillion", "ath", "record",
}

EVENT_PATTERNS = {
    "hack":       ["hack", "exploit", "attack", "breach", "stolen", "vulnerability"],
    "etf":        ["etf", "spot etf", "bitcoin etf", "approval", "approved"],
    "regulation": ["regulation", "banned", "crackdown", "sec", "lawsuit", "compliance"],
    "listing":    ["listing", "listed", "coinbase listing", "binance listing"],
    "macro":      ["fed", "federal reserve", "interest rate", "inflation", "cpi", "gdp"],
}


class NewsEngine:
    """
    Aggregates news from multiple free sources.
    Scores sentiment and detects market-moving events.
    """

    def __init__(self, cryptopanic_token: str = ""):
        self.token  = cryptopanic_token
        self._cache: Optional[NewsReport] = None
        self._cache_ts: Optional[datetime] = None

    # ── PUBLIC ────────────────────────────────────────────────

    def fetch(self, symbols: List[str] = None,
              limit: int = 20, force: bool = False) -> NewsReport:
        """Fetch and analyze news. Uses cache if fresh."""
        if not force and self._cache and self._cache_ts:
            age = (datetime.now() - self._cache_ts).seconds / 60
            if age < CACHE_MINUTES:
                return self._cache

        symbols = symbols or ["BTC", "ETH"]
        items: List[NewsItem] = []

        # Try CryptoPanic
        if self.token:
            cp_items = self._fetch_cryptopanic(symbols, limit)
            items.extend(cp_items)

        # Always try free sources
        alt_items = self._fetch_alternative_me()
        items.extend(alt_items)

        cg_items = self._fetch_coingecko_trending()
        items.extend(cg_items)

        # Deduplicate by title similarity
        items = self._deduplicate(items)

        # Score all items
        for item in items:
            self._score_item(item)
            self._detect_event(item)
            item.ai_analysis = self._generate_ai_analysis(item)

        report = self._build_report(items)
        self._cache    = report
        self._cache_ts = datetime.now()

        log.info(f"[News] Fetched {len(items)} items | Score: {report.overall_score:+.2f} ({report.overall_label})")
        return report

    # ── SOURCES ───────────────────────────────────────────────

    def _fetch_cryptopanic(self, symbols: List[str], limit: int) -> List[NewsItem]:
        """CryptoPanic API — requires free token from cryptopanic.com"""
        items = []
        try:
            currencies = ",".join(symbols[:3])
            url = (f"https://cryptopanic.com/api/v1/posts/"
                   f"?auth_token={self.token}&currencies={currencies}"
                   f"&filter=hot&limit={limit}&public=true")
            r = requests.get(url, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            data = r.json()

            for post in data.get("results", []):
                coins = [c["code"] for c in post.get("currencies", [])]
                published = post.get("published_at", "")
                items.append(NewsItem(
                    title=post.get("title", ""),
                    source=post.get("source", {}).get("title", "CryptoPanic"),
                    url=post.get("url", ""),
                    published=published[:16] if published else "",
                    coins=coins,
                ))
        except Exception as e:
            log.warning(f"[News] CryptoPanic failed: {e}")
        return items

    def _fetch_alternative_me(self) -> List[NewsItem]:
        """Fear & Greed context as a news item."""
        items = []
        try:
            r = requests.get("https://api.alternative.me/fng/?limit=3",
                             timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            data = r.json().get("data", [])
            if data:
                score = int(data[0]["value"])
                label = data[0]["value_classification"]
                prev  = int(data[-1]["value"]) if len(data) > 1 else score
                trend = "improving" if score > prev else "declining" if score < prev else "stable"
                cat   = "bullish" if score < 30 else "bearish" if score > 70 else "neutral"
                title = f"Crypto Fear & Greed Index: {score}/100 — {label} ({trend})"
                items.append(NewsItem(
                    title=title,
                    source="Alternative.me",
                    category=cat,
                    sentiment=0.3 if score < 30 else -0.3 if score > 70 else 0.0,
                    impact="medium",
                    coins=["BTC","ETH"],
                ))
        except Exception as e:
            log.warning(f"[News] Alternative.me failed: {e}")
        return items

    def _fetch_coingecko_trending(self) -> List[NewsItem]:
        """CoinGecko trending coins as market intelligence."""
        items = []
        try:
            r = requests.get(
                "https://api.coingecko.com/api/v3/search/trending",
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": "CryptoBot/6.0"},
            )
            r.raise_for_status()
            data = r.json()
            coins = [c["item"]["symbol"].upper()
                     for c in data.get("coins", [])[:5]]
            if coins:
                items.append(NewsItem(
                    title=f"Trending: {', '.join(coins)} — high retail interest",
                    source="CoinGecko",
                    category="neutral",
                    sentiment=0.1,
                    impact="low",
                    coins=coins,
                ))
        except Exception as e:
            log.warning(f"[News] CoinGecko trending failed: {e}")
        return items

    # ── PROCESSING ────────────────────────────────────────────

    def _score_item(self, item: NewsItem) -> None:
        """Keyword-based sentiment scoring."""
        text = (item.title + " " + item.body).lower()
        pos_count = sum(1 for w in POSITIVE_WORDS if w in text)
        neg_count = sum(1 for w in NEGATIVE_WORDS if w in text)
        total = pos_count + neg_count + 1e-9
        score = (pos_count - neg_count) / total
        item.sentiment = round(max(-1.0, min(1.0, score)), 3)

        # Category
        if item.sentiment > 0.25:   item.category = "bullish"
        elif item.sentiment < -0.25: item.category = "bearish"
        elif any(w in text for w in ["fed","inflation","macro","interest rate"]):
            item.category = "macro"

        # Impact
        high_hits = sum(1 for w in HIGH_IMPACT_WORDS if w in text)
        item.impact = "high" if high_hits >= 2 else "medium" if high_hits == 1 else "low"

    def _detect_event(self, item: NewsItem) -> None:
        """Detect specific market-moving events."""
        text = (item.title + " " + item.body).lower()
        for event_type, keywords in EVENT_PATTERNS.items():
            if any(kw in text for kw in keywords):
                item.event_type = event_type
                if event_type == "hack":
                    item.sentiment = min(-0.7, item.sentiment)
                    item.impact = "high"
                    item.category = "bearish"
                elif event_type == "etf":
                    item.sentiment = max(0.5, item.sentiment)
                    item.impact = "high"
                    item.category = "bullish"
                break

    def _generate_ai_analysis(self, item: NewsItem) -> str:
        """Generate brief AI analysis for each news item."""
        cat = item.category
        sent = item.sentiment
        impact = item.impact

        if cat == "bullish" and impact == "high":
            return f"Strong positive catalyst. Confidence: {abs(sent):.0%}. Monitor for entry."
        elif cat == "bullish":
            return f"Mild bullish signal. Sentiment: {sent:+.2f}. Factor into analysis."
        elif cat == "bearish" and impact == "high":
            return f"High-impact negative event. Risk elevated. Consider reducing exposure."
        elif cat == "bearish":
            return f"Bearish signal. Sentiment: {sent:+.2f}. Tighten stop-losses."
        elif cat == "macro":
            return f"Macro event — monitor dollar strength and risk-off sentiment."
        elif item.event_type == "hack":
            return f"Security incident detected. Short-term selling pressure expected."
        elif item.event_type == "etf":
            return f"ETF news — historically strong bullish catalyst. High confidence BUY signal."
        return f"Neutral news. Sentiment: {sent:+.2f}. No immediate action required."

    def _deduplicate(self, items: List[NewsItem]) -> List[NewsItem]:
        """Remove very similar titles."""
        seen = set()
        result = []
        for item in items:
            key = re.sub(r'[^a-z0-9]', '', item.title.lower())[:50]
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result

    def _build_report(self, items: List[NewsItem]) -> NewsReport:
        if not items:
            return NewsReport(items=[], overall_score=0.0,
                              overall_label="neutral", btc_score=0.0,
                              eth_score=0.0, events=[],
                              last_updated=datetime.now().isoformat())

        overall = sum(i.sentiment for i in items) / len(items)

        # Per-coin scores
        btc_items = [i for i in items if "BTC" in i.coins or "BITCOIN" in i.title.upper()]
        eth_items = [i for i in items if "ETH" in i.coins or "ETHEREUM" in i.title.upper()]
        btc_score = sum(i.sentiment for i in btc_items) / len(btc_items) if btc_items else overall
        eth_score = sum(i.sentiment for i in eth_items) / len(eth_items) if eth_items else overall

        # Label
        if overall > 0.4:   label = "very_positive"
        elif overall > 0.15: label = "positive"
        elif overall < -0.4: label = "very_negative"
        elif overall < -0.15: label = "negative"
        else:                label = "neutral"

        # Events
        events = list(set(i.event_type for i in items if i.event_type))

        return NewsReport(
            items=sorted(items, key=lambda x: abs(x.sentiment), reverse=True),
            overall_score=round(overall, 3),
            overall_label=label,
            btc_score=round(btc_score, 3),
            eth_score=round(eth_score, 3),
            events=events,
            last_updated=datetime.now().isoformat(),
        )

    def format_for_agent(self, report: NewsReport) -> dict:
        """Format for AI agent consumption."""
        return {
            "score":       report.overall_score,
            "label":       report.overall_label,
            "btc_score":   report.btc_score,
            "eth_score":   report.eth_score,
            "events":      report.events,
            "top_title":   report.items[0].title if report.items else "",
            "item_count":  len(report.items),
        }
