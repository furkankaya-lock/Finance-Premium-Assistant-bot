"""
data/collector.py
─────────────────
Binance OHLCV · Order Book · Fear & Greed · CryptoPanic haber skoru
"""

import time
import logging
import requests
import numpy as np
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException

log = logging.getLogger("CryptoBot.Collector")

OHLCV_COLS = [
    "open_time","open","high","low","close","volume",
    "close_time","quote_vol","trades","taker_buy_base",
    "taker_buy_quote","ignore"
]


# ─────────────────────────────────────────────────────────────
# OHLCV
# ─────────────────────────────────────────────────────────────

def ohlcv_al(client: Client,
             symbol: str,
             interval: str = Client.KLINE_INTERVAL_1HOUR,
             limit: int = 500) -> pd.DataFrame:
    """
    Binance'den mum verisi çeker.
    limit=500 → LSTM için yeterli tarihsel veri sağlar.
    """
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=OHLCV_COLS)
        float_cols = ["open", "high", "low", "close", "volume",
                      "quote_vol", "taker_buy_base", "taker_buy_quote"]
        df[float_cols] = df[float_cols].astype(float)
        df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        df = df.set_index("open_time").sort_index()
        log.debug(f"OHLCV alındı: {symbol} | {len(df)} mum")
        return df
    except BinanceAPIException as e:
        log.error(f"OHLCV hatası ({symbol}): {e}")
        raise


def guncel_fiyat(client: Client, symbol: str) -> float:
    """Anlık fiyat"""
    ticker = client.get_symbol_ticker(symbol=symbol)
    return float(ticker["price"])


# ─────────────────────────────────────────────────────────────
# ORDER BOOK ANALİZİ
# ─────────────────────────────────────────────────────────────

def order_book_analiz(client: Client,
                      symbol: str,
                      derinlik: int = 20) -> dict:
    """
    Order book'tan alım/satım baskısını ölçer.
    bid_ask_ratio > 1 → alım baskısı güçlü
    """
    try:
        depth = client.get_order_book(symbol=symbol, limit=derinlik)
        bids  = sum(float(b[1]) for b in depth["bids"])  # alım tarafı hacim
        asks  = sum(float(a[1]) for a in depth["asks"])  # satış tarafı hacim
        total = bids + asks
        ratio = bids / asks if asks > 0 else 1.0
        spread = (float(depth["asks"][0][0]) - float(depth["bids"][0][0]))
        best_bid = float(depth["bids"][0][0])
        spread_pct = (spread / best_bid) * 100 if best_bid > 0 else 0

        return {
            "bid_volume":    round(bids, 4),
            "ask_volume":    round(asks, 4),
            "bid_ask_ratio": round(ratio, 4),
            "spread_pct":    round(spread_pct, 5),
            "buy_pressure":  round(bids / total * 100, 2) if total > 0 else 50.0,
        }
    except Exception as e:
        log.warning(f"Order book hatası ({symbol}): {e}")
        return {"bid_ask_ratio": 1.0, "spread_pct": 0.0, "buy_pressure": 50.0,
                "bid_volume": 0, "ask_volume": 0}


# ─────────────────────────────────────────────────────────────
# FEAR & GREED INDEX
# ─────────────────────────────────────────────────────────────

def fear_greed_al() -> dict:
    """
    Alternative.me Fear & Greed Index (0=Extreme Fear, 100=Extreme Greed)
    Ücretsiz API, kayıt gerektirmez.
    """
    try:
        r = requests.get(
            "https://api.alternative.me/fng/?limit=1",
            timeout=5
        )
        r.raise_for_status()
        data = r.json()["data"][0]
        deger = int(data["value"])
        sinif = data["value_classification"]
        return {
            "score":      deger,
            "label":      sinif,
            "sentiment":  _deger_to_sentiment(deger),
        }
    except Exception as e:
        log.warning(f"Fear & Greed alınamadı: {e}")
        return {"score": 50, "label": "Neutral", "sentiment": "neutral"}


def _deger_to_sentiment(deger: int) -> str:
    if deger <= 20:   return "extreme_fear"
    if deger <= 40:   return "fear"
    if deger <= 60:   return "neutral"
    if deger <= 80:   return "greed"
    return "extreme_greed"


# ─────────────────────────────────────────────────────────────
# HABER SENTİMENTİ (CryptoPanic)
# ─────────────────────────────────────────────────────────────

def haber_skoru_al(symbol: str = "BTC",
                   cryptopanic_token: str = "") -> dict:
    """
    CryptoPanic API'den son haberleri çeker ve pozitif/negatif skor üretir.
    Token yoksa neutral döner.
    """
    if not cryptopanic_token:
        return {"score": 0.0, "positive": 0, "negative": 0,
                "total": 0, "label": "no_api_key"}
    try:
        coin = symbol.replace("USDT", "")
        url  = (
            f"https://cryptopanic.com/api/v1/posts/"
            f"?auth_token={cryptopanic_token}"
            f"&currencies={coin}&filter=hot&kind=news"
        )
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        haberler = r.json().get("results", [])[:20]

        pozitif = sum(1 for h in haberler if h.get("votes", {}).get("positive", 0) >
                      h.get("votes", {}).get("negative", 0))
        negatif = len(haberler) - pozitif
        total   = len(haberler) if haberler else 1
        score   = (pozitif - negatif) / total

        return {
            "score":    round(score, 3),
            "positive": pozitif,
            "negative": negatif,
            "total":    total,
            "label":    "positive" if score > 0.2 else "negative" if score < -0.2 else "neutral",
        }
    except Exception as e:
        log.warning(f"Haber skoru hatası ({symbol}): {e}")
        return {"score": 0.0, "positive": 0, "negative": 0, "total": 0, "label": "error"}


# ─────────────────────────────────────────────────────────────
# TOPLU VERİ PAKETİ
# ─────────────────────────────────────────────────────────────

def tam_veri_paketi(client: Client,
                    symbol: str,
                    interval: str,
                    cryptopanic_token: str = "") -> dict:
    """
    Tüm veriyi bir seferde toplar ve tek sözlük olarak döner.
    """
    df      = ohlcv_al(client, symbol, interval, limit=500)
    fiyat   = float(df["close"].iloc[-1])
    ob      = order_book_analiz(client, symbol)
    fg      = fear_greed_al()
    haber   = haber_skoru_al(symbol, cryptopanic_token)

    return {
        "df":     df,
        "fiyat":  fiyat,
        "ob":     ob,
        "fg":     fg,
        "haber":  haber,
        "symbol": symbol,
        "zaman":  pd.Timestamp.now().isoformat(),
    }
