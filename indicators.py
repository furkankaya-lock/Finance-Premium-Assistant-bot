"""
data/indicators.py
──────────────────
Tüm teknik göstergeler tek yerden hesaplanır.
RSI · EMA · MACD · ATR · Bollinger · Momentum · OBV · VWAP
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# TEMEL İNDİKATÖRLER
# ─────────────────────────────────────────────────────────────

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def ema(close: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average"""
    return close.ewm(span=span, adjust=False).mean()


def sma(close: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average"""
    return close.rolling(window=window).mean()


def macd(close: pd.Series,
         fast: int = 12,
         slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    """MACD · Signal · Histogram"""
    ema_fast   = ema(close, fast)
    ema_slow   = ema(close, slow)
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return pd.DataFrame({
        "macd":      macd_line,
        "signal":    signal_line,
        "histogram": histogram,
    })


def bollinger_bands(close: pd.Series,
                    window: int = 20,
                    num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands — upper / middle / lower"""
    mid   = sma(close, window)
    std   = close.rolling(window).std()
    return pd.DataFrame({
        "upper":  mid + num_std * std,
        "middle": mid,
        "lower":  mid - num_std * std,
        "width":  (2 * num_std * std) / mid,   # bant genişliği
    })


def atr(high: pd.Series,
        low:  pd.Series,
        close: pd.Series,
        period: int = 14) -> pd.Series:
    """Average True Range — dinamik SL için kullanılır"""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def momentum(close: pd.Series, period: int = 10) -> pd.Series:
    """Momentum (Rate of Change)"""
    return close.pct_change(periods=period) * 100


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume"""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def vwap(high: pd.Series,
         low:  pd.Series,
         close: pd.Series,
         volume: pd.Series) -> pd.Series:
    """Volume Weighted Average Price"""
    tp = (high + low + close) / 3
    return (tp * volume).cumsum() / volume.cumsum()


def stochastic(high: pd.Series,
               low:  pd.Series,
               close: pd.Series,
               k: int = 14,
               d: int = 3) -> pd.DataFrame:
    """Stochastic Oscillator %K ve %D"""
    lowest  = low.rolling(k).min()
    highest = high.rolling(k).max()
    k_line  = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    d_line  = k_line.rolling(d).mean()
    return pd.DataFrame({"k": k_line, "d": d_line})


def williams_r(high: pd.Series,
               low:  pd.Series,
               close: pd.Series,
               period: int = 14) -> pd.Series:
    """Williams %R"""
    highest = high.rolling(period).max()
    lowest  = low.rolling(period).min()
    return -100 * (highest - close) / (highest - lowest).replace(0, np.nan)


def cci(high: pd.Series,
        low:  pd.Series,
        close: pd.Series,
        period: int = 20) -> pd.Series:
    """Commodity Channel Index"""
    tp      = (high + low + close) / 3
    tp_sma  = tp.rolling(period).mean()
    mean_dev = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
    return (tp - tp_sma) / (0.015 * mean_dev.replace(0, np.nan))


# ─────────────────────────────────────────────────────────────
# TOPLU HESAPLAMA
# ─────────────────────────────────────────────────────────────

def hesapla_hepsi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ham OHLCV DataFrame'ini alır, tüm göstergeleri ekler ve döner.
    Girdi sütunları: open, high, low, close, volume
    """
    df = df.copy()

    # EMA ailesi
    for span in [9, 12, 20, 21, 26, 50, 200]:
        df[f"ema_{span}"] = ema(df["close"], span)

    # Temel göstergeler
    df["rsi_14"]    = rsi(df["close"], 14)
    df["rsi_21"]    = rsi(df["close"], 21)
    df["momentum"]  = momentum(df["close"], 10)
    df["obv"]       = obv(df["close"], df["volume"])
    df["vwap"]      = vwap(df["high"], df["low"], df["close"], df["volume"])
    df["atr_14"]    = atr(df["high"], df["low"], df["close"], 14)
    df["williams_r"]= williams_r(df["high"], df["low"], df["close"], 14)
    df["cci"]       = cci(df["high"], df["low"], df["close"], 20)

    # MACD
    _macd = macd(df["close"])
    df["macd"]       = _macd["macd"]
    df["macd_signal"]= _macd["signal"]
    df["macd_hist"]  = _macd["histogram"]

    # Bollinger
    _bb = bollinger_bands(df["close"])
    df["bb_upper"]  = _bb["upper"]
    df["bb_middle"] = _bb["middle"]
    df["bb_lower"]  = _bb["lower"]
    df["bb_width"]  = _bb["width"]
    df["bb_pos"]    = (df["close"] - _bb["lower"]) / (_bb["upper"] - _bb["lower"]).replace(0, np.nan)

    # Stochastic
    _stoch = stochastic(df["high"], df["low"], df["close"])
    df["stoch_k"] = _stoch["k"]
    df["stoch_d"] = _stoch["d"]

    # Golden/Death Cross
    df["golden_cross"] = (df["ema_50"] > df["ema_200"]).astype(int)
    df["ema_trend"]    = np.where(df["ema_9"] > df["ema_21"], 1,
                          np.where(df["ema_9"] < df["ema_21"], -1, 0))

    # Hacim göstergeleri
    df["volume_sma_20"] = sma(df["volume"], 20)
    df["volume_ratio"]  = df["volume"] / df["volume_sma_20"].replace(0, np.nan)

    # Fiyat değişimi
    df["price_change_1h"]  = df["close"].pct_change(1)
    df["price_change_4h"]  = df["close"].pct_change(4)
    df["price_change_24h"] = df["close"].pct_change(24)

    # Volatilite
    df["volatility"] = df["close"].pct_change().rolling(20).std() * np.sqrt(24) * 100

    return df.dropna(subset=["rsi_14", "ema_50", "macd"])


def ozellik_vektoru(df: pd.DataFrame) -> dict:
    """
    Son satırın tüm özelliklerini sözlük olarak döner.
    Random Forest ve Claude API'ye gönderilir.
    """
    son = df.iloc[-1]
    return {
        "rsi":          round(float(son["rsi_14"]), 2),
        "rsi_21":       round(float(son["rsi_21"]), 2),
        "macd":         round(float(son["macd"]), 4),
        "macd_signal":  round(float(son["macd_signal"]), 4),
        "macd_hist":    round(float(son["macd_hist"]), 4),
        "ema_9":        round(float(son["ema_9"]), 2),
        "ema_21":       round(float(son["ema_21"]), 2),
        "ema_50":       round(float(son["ema_50"]), 2),
        "ema_200":      round(float(son["ema_200"]), 2),
        "ema_trend":    int(son["ema_trend"]),
        "golden_cross": int(son["golden_cross"]),
        "bb_pos":       round(float(son["bb_pos"]), 3),
        "bb_width":     round(float(son["bb_width"]), 4),
        "stoch_k":      round(float(son["stoch_k"]), 2),
        "stoch_d":      round(float(son["stoch_d"]), 2),
        "williams_r":   round(float(son["williams_r"]), 2),
        "cci":          round(float(son["cci"]), 2),
        "momentum":     round(float(son["momentum"]), 4),
        "volume_ratio": round(float(son["volume_ratio"]), 3),
        "volatility":   round(float(son["volatility"]), 4),
        "atr_14":       round(float(son["atr_14"]), 4),
        "price_chg_1h": round(float(son["price_change_1h"]), 5),
        "price_chg_4h": round(float(son["price_change_4h"]), 5),
        "price_chg_24h":round(float(son["price_change_24h"]), 5),
        "close":        round(float(son["close"]), 4),
        "vwap":         round(float(son["vwap"]), 4),
    }
