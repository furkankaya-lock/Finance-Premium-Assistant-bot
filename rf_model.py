"""
ai_engine/rf_model.py
─────────────────────
Random Forest tabanlı AL / SAT / BEKLE sinyal üretici.
Özellik tabanlı çalışır — fiyat tahmini değil, sinyal sınıflandırması yapar.
"""

import os
import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

log = logging.getLogger("CryptoBot.RF")

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

SINIFLAR  = {0: "SAT", 1: "BEKLE", 2: "AL"}
SINIFLAR_R = {"SAT": 0, "BEKLE": 1, "AL": 2}

RF_OZELLIKLER = [
    "rsi_14", "rsi_21", "macd", "macd_signal", "macd_hist",
    "ema_trend", "golden_cross", "bb_pos", "bb_width",
    "stoch_k", "stoch_d", "williams_r", "cci",
    "momentum", "volume_ratio", "volatility", "atr_14",
    "price_chg_1h", "price_chg_4h", "price_chg_24h",
]


class RandomForestSinyalModeli:
    """
    Teknik göstergelerden AL/SAT/BEKLE sinyali üretir.
    Çıktı: {"sinyal": "AL"|"SAT"|"BEKLE", "olasiliklar": {…}, "guven": float}
    """

    def __init__(self, symbol: str):
        self.symbol     = symbol
        self.model      = None
        self.boost_model= None
        self.scaler     = StandardScaler()
        self.egitildi   = False
        self.model_path = os.path.join(MODEL_DIR, f"rf_{symbol}.pkl")
        self._yukle()

    # ── ETİKET ÜRETİMİ ────────────────────────────────────

    @staticmethod
    def etiket_uret(df: pd.DataFrame,
                    gelecek_mum: int = 3,
                    kar_esik: float = 0.008) -> pd.Series:
        """
        Gelecekteki N mum bazında AL/SAT/BEKLE etiketi oluşturur.
        kar_esik = %0.8 → bu kadar hareket olmadıkça BEKLE
        """
        gelecek_getiri = df["close"].shift(-gelecek_mum) / df["close"] - 1
        labels = np.where(
            gelecek_getiri >  kar_esik, 2,      # AL
            np.where(
            gelecek_getiri < -kar_esik, 0,      # SAT
            1                                    # BEKLE
        ))
        return pd.Series(labels, index=df.index, name="label")

    # ── EĞİTİM ────────────────────────────────────────────

    def egit(self, df: pd.DataFrame) -> dict:
        """
        Geçmiş veri üzerinde modeli eğitir.
        En az 200 satır veri gerekir.
        """
        df = df.copy().dropna(subset=RF_OZELLIKLER)
        if len(df) < 200:
            log.warning(f"[{self.symbol}] RF için yetersiz veri: {len(df)}")
            return {"durum": "yetersiz_veri"}

        etiketler = self.etiket_uret(df)
        df["label"] = etiketler

        # Son 3 mumu çıkar (gelecek bilgisi eksik)
        df = df.dropna(subset=["label"]).iloc[:-3]
        if len(df) < 100:
            return {"durum": "yetersiz_etiket"}

        X = df[RF_OZELLIKLER].values
        y = df["label"].values.astype(int)

        # Sınıf dengesizliği kontrolü
        sinif_dagilim = {SINIFLAR[k]: int((y == k).sum()) for k in [0, 1, 2]}
        log.info(f"[{self.symbol}] Etiket dağılımı: {sinif_dagilim}")

        X_scaled = self.scaler.fit_transform(X)
        split    = int(len(X_scaled) * 0.85)

        # Ana model: Random Forest
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_scaled[:split], y[:split])

        # İkincil model: Gradient Boosting (ensemble için)
        self.boost_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
        )
        self.boost_model.fit(X_scaled[:split], y[:split])

        # Değerlendirme
        X_te = X_scaled[split:]
        y_te = y[split:]
        rf_acc  = self.model.score(X_te, y_te)
        gb_acc  = self.boost_model.score(X_te, y_te)

        rapor = classification_report(
            y_te, self.model.predict(X_te),
            target_names=["SAT", "BEKLE", "AL"],
            output_dict=True, zero_division=0
        )

        self.egitildi = True
        self._kaydet()
        log.info(f"[{self.symbol}] RF: {rf_acc:.2%} | GB: {gb_acc:.2%}")
        return {
            "durum":    "tamam",
            "rf_acc":   round(rf_acc, 4),
            "gb_acc":   round(gb_acc, 4),
            "rapor":    rapor,
            "dagilim":  sinif_dagilim,
        }

    # ── TAHMİN ────────────────────────────────────────────

    def tahmin(self, df: pd.DataFrame) -> dict:
        """
        Son satır için sinyal üretir.
        """
        if not self.egitildi or self.model is None:
            log.warning(f"[{self.symbol}] RF henüz eğitilmedi, eğitiliyor...")
            sonuc = self.egit(df)
            if sonuc.get("durum") != "tamam":
                return {"sinyal": "BEKLE", "guven": 0.0,
                        "olasiliklar": {"AL": 0.33, "SAT": 0.33, "BEKLE": 0.34}}

        df_clean = df.dropna(subset=RF_OZELLIKLER)
        if len(df_clean) == 0:
            return {"sinyal": "BEKLE", "guven": 0.0,
                    "olasiliklar": {"AL": 0.33, "SAT": 0.33, "BEKLE": 0.34}}

        son = df_clean[RF_OZELLIKLER].values[-1].reshape(1, -1)
        son_scaled = self.scaler.transform(son)

        # Her iki modelden olasılık al → ensemble
        rf_proba = self.model.predict_proba(son_scaled)[0]
        gb_proba = self.boost_model.predict_proba(son_scaled)[0]

        # Sınıf sayısı uyumluluğu kontrolü
        n_sinif = len(self.model.classes_)
        if len(rf_proba) != 3 or len(gb_proba) != 3:
            rf_proba = _sinif_doldur(rf_proba, self.model.classes_)
            gb_proba = _sinif_doldur(gb_proba, self.boost_model.classes_)

        ensemble = 0.6 * rf_proba + 0.4 * gb_proba
        ensemble /= ensemble.sum()

        idx     = int(np.argmax(ensemble))
        sinyal  = SINIFLAR[idx]
        guven   = float(ensemble[idx])

        # Güven eşiği — düşük güven = BEKLE
        if guven < 0.45:
            sinyal = "BEKLE"

        return {
            "sinyal": sinyal,
            "guven":  round(guven, 4),
            "olasiliklar": {
                "SAT":   round(float(ensemble[0]), 4),
                "BEKLE": round(float(ensemble[1]), 4),
                "AL":    round(float(ensemble[2]), 4),
            },
            "rf_proba":    [round(float(p), 4) for p in rf_proba],
            "gb_proba":    [round(float(p), 4) for p in gb_proba],
        }

    def ozellik_onemleri(self) -> dict:
        """Feature importance döner"""
        if not self.egitildi:
            return {}
        return dict(sorted(
            zip(RF_OZELLIKLER, self.model.feature_importances_),
            key=lambda x: x[1], reverse=True
        ))

    # ── YARDIMCI ──────────────────────────────────────────

    def _kaydet(self):
        try:
            with open(self.model_path, "wb") as f:
                pickle.dump({
                    "model":       self.model,
                    "boost_model": self.boost_model,
                    "scaler":      self.scaler,
                    "egitildi":    True,
                }, f)
            log.info(f"[{self.symbol}] RF modeli kaydedildi → {self.model_path}")
        except Exception as e:
            log.warning(f"RF kayıt hatası: {e}")

    def _yukle(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    d = pickle.load(f)
                self.model       = d["model"]
                self.boost_model = d["boost_model"]
                self.scaler      = d["scaler"]
                self.egitildi    = d["egitildi"]
                log.info(f"[{self.symbol}] RF modeli yüklendi → {self.model_path}")
            except Exception as e:
                log.warning(f"RF yükleme hatası: {e}")


def _sinif_doldur(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Eksik sınıfları sıfır ile doldurur (3 sınıf garantisi)"""
    full = np.zeros(3)
    for i, c in enumerate(classes):
        if 0 <= c <= 2:
            full[c] = proba[i]
    if full.sum() == 0:
        full[1] = 1.0
    else:
        full /= full.sum()
    return full
