"""
risk/manager.py
───────────────
ATR tabanlı dinamik Stop Loss · Kelly Criterion pozisyon boyutu
Günlük kayıp limiti · Korelasyon filtresi · Drawdown takibi
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import Optional

log = logging.getLogger("CryptoBot.Risk")


class RiskYoneticisi:
    """
    Tüm risk parametrelerini yönetir.
    Bot ana döngüsüne enjekte edilir.
    """

    def __init__(self,
                 baslangic_sermaye:  float = 1000.0,
                 maks_gunluk_kayip:  float = 0.05,   # %5
                 maks_pozisyon_pct:  float = 0.20,   # Sermayenin %20'si
                 atr_sl_carpan:      float = 2.0,    # ATR × 2 = SL mesafesi
                 atr_tp_carpan:      float = 3.0,    # ATR × 3 = TP mesafesi
                 max_acik_pozisyon:  int   = 2,      # Aynı anda max pozisyon
                 min_guven:          float = 0.60,   # Min karar güveni
                 ):
        self.baslangic_sermaye  = baslangic_sermaye
        self.mevcut_sermaye     = baslangic_sermaye
        self.maks_gunluk_kayip  = maks_gunluk_kayip
        self.maks_pozisyon_pct  = maks_pozisyon_pct
        self.atr_sl_carpan      = atr_sl_carpan
        self.atr_tp_carpan      = atr_tp_carpan
        self.max_acik_pozisyon  = max_acik_pozisyon
        self.min_guven          = min_guven

        # Günlük kayıp takibi
        self.gun_baslangic_sermaye = baslangic_sermaye
        self.son_guncelleme_gunu   = date.today()

        # İşlem geçmişi
        self.islem_gecmisi: list = []
        self.maks_sermaye        = baslangic_sermaye
        self.maks_drawdown       = 0.0

    # ── GÜNLÜK SIFIRLAMA ──────────────────────────────────

    def gunluk_guncelle(self, guncel_sermaye: float) -> None:
        bugun = date.today()
        if bugun != self.son_guncelleme_gunu:
            self.gun_baslangic_sermaye = guncel_sermaye
            self.son_guncelleme_gunu   = bugun
            log.info(f"[Risk] Yeni gün → Günlük sermaye sıfırlandı: ${guncel_sermaye:.2f}")
        self.mevcut_sermaye = guncel_sermaye
        # Max drawdown güncelle
        if guncel_sermaye > self.maks_sermaye:
            self.maks_sermaye = guncel_sermaye
        dd = (self.maks_sermaye - guncel_sermaye) / self.maks_sermaye
        if dd > self.maks_drawdown:
            self.maks_drawdown = dd

    # ── GÜNLÜK LİMİT KONTROLÜ ─────────────────────────────

    def gunluk_limit_asimi(self) -> bool:
        """Günlük kayıp limitini aştıysa True döner"""
        if self.gun_baslangic_sermaye <= 0:
            return True
        kayip_pct = (self.gun_baslangic_sermaye - self.mevcut_sermaye) / self.gun_baslangic_sermaye
        if kayip_pct >= self.maks_gunluk_kayip:
            log.warning(f"🛑 Günlük kayıp limiti: %{kayip_pct:.1%} ≥ %{self.maks_gunluk_kayip:.0%}")
            return True
        return False

    # ── POZİSYON BOYUTU (KELLY) ────────────────────────────

    def pozisyon_boyutu(self,
                        guven:     float,
                        risk_skoru: float,
                        kullanilabilir_usdt: float) -> float:
        """
        Kelly Criterion tabanlı pozisyon boyutu hesaplar.
        guven = model güven skoru (0–1)
        risk_skoru = risk (0–1, yüksek = tehlikeli)
        """
        # Kazanma olasılığı tahmini
        win_prob = guven
        loss_prob = 1 - win_prob

        # Kelly: f = (bp - q) / b
        # b = ödül/risk oranı (TP/SL yaklaşımı)
        b = self.atr_tp_carpan / self.atr_sl_carpan
        kelly = (b * win_prob - loss_prob) / b
        kelly = max(0.0, min(kelly, 0.25))  # Max %25 Kelly

        # Risk düzeltmesi
        risk_duzeltme = 1 - risk_skoru * 0.5
        kelly *= risk_duzeltme

        # Güven eşiği — minimum güven altında sıfır
        if guven < self.min_guven:
            return 0.0

        # USDT miktarı hesapla
        maks_miktar = self.mevcut_sermaye * self.maks_pozisyon_pct
        hesaplanan  = self.mevcut_sermaye * kelly
        miktar      = min(hesaplanan, maks_miktar, kullanilabilir_usdt)

        log.info(
            f"[Risk] Kelly: {kelly:.3f} | "
            f"Risk düzelt: {risk_duzeltme:.3f} | "
            f"Pozisyon: ${miktar:.2f}"
        )
        return round(miktar, 2)

    # ── ATR TABANLI SL / TP ────────────────────────────────

    def dinamik_sl_tp(self,
                      giris_fiyati: float,
                      atr:          float,
                      yon:          str = "long") -> dict:
        """
        ATR tabanlı dinamik Stop Loss ve Take Profit hesaplar.
        yon: "long" (AL) veya "short" (SAT)
        """
        sl_mesafe = atr * self.atr_sl_carpan
        tp_mesafe = atr * self.atr_tp_carpan

        if yon == "long":
            sl = giris_fiyati - sl_mesafe
            tp = giris_fiyati + tp_mesafe
        else:
            sl = giris_fiyati + sl_mesafe
            tp = giris_fiyati - tp_mesafe

        sl_pct = abs(sl_mesafe / giris_fiyati) * 100
        tp_pct = abs(tp_mesafe / giris_fiyati) * 100
        rr     = tp_pct / sl_pct if sl_pct > 0 else 0

        log.info(
            f"[Risk] SL: ${sl:.4f} (-%{sl_pct:.2f}) | "
            f"TP: ${tp:.4f} (+%{tp_pct:.2f}) | R/R: {rr:.2f}"
        )
        return {
            "sl":     round(sl, 6),
            "tp":     round(tp, 6),
            "sl_pct": round(sl_pct, 3),
            "tp_pct": round(tp_pct, 3),
            "rr":     round(rr, 3),
        }

    # ── TRAILING STOP ──────────────────────────────────────

    def trailing_stop_guncelle(self,
                                pozisyon: dict,
                                guncel_fiyat: float,
                                atr: float) -> dict:
        """
        Fiyat yükselirse SL'i yukarı çeker (trailing stop).
        """
        eski_sl = pozisyon.get("sl", 0)
        yeni_sl = guncel_fiyat - atr * self.atr_sl_carpan
        if yeni_sl > eski_sl:
            pozisyon["sl"] = round(yeni_sl, 6)
            log.info(f"[Risk] Trailing SL güncellendi: ${eski_sl:.4f} → ${yeni_sl:.4f}")
        return pozisyon

    # ── POZİSYON KONTROLÜ ─────────────────────────────────

    def pozisyon_kontrol(self,
                         pozisyon: dict,
                         guncel_fiyat: float,
                         atr: float) -> str:
        """
        Açık pozisyon için SL/TP kontrolü yapar.
        Döner: "KAPAT_SL" | "KAPAT_TP" | "DEVAM"
        """
        # Trailing stop güncelle
        pozisyon = self.trailing_stop_guncelle(pozisyon, guncel_fiyat, atr)

        sl = pozisyon.get("sl", 0)
        tp = pozisyon.get("tp", float("inf"))

        if guncel_fiyat <= sl:
            pct = (guncel_fiyat - pozisyon["giris_fiyati"]) / pozisyon["giris_fiyati"] * 100
            log.warning(f"🛑 SL tetiklendi: ${guncel_fiyat:.4f} ≤ ${sl:.4f} | %{pct:.2f}")
            return "KAPAT_SL"
        if guncel_fiyat >= tp:
            pct = (guncel_fiyat - pozisyon["giris_fiyati"]) / pozisyon["giris_fiyati"] * 100
            log.info(f"🎯 TP tetiklendi: ${guncel_fiyat:.4f} ≥ ${tp:.4f} | %{pct:.2f}")
            return "KAPAT_TP"
        return "DEVAM"

    # ── KORELASYON FİLTRESİ ───────────────────────────────

    @staticmethod
    def korelasyon_filtre(acik_pozisyonlar: dict,
                          yeni_symbol: str,
                          esik: float = 0.85) -> bool:
        """
        BTC ve ETH yüksek korelasyonlu varlıklardır.
        İkisi de açıksa yeni pozisyon açmayı engeller.
        esik: korelasyon eşiği
        True = pozisyon açılabilir, False = engellensin
        """
        if not acik_pozisyonlar:
            return True

        # BTC/ETH çiftini engelle
        btc_eth_cift = {"BTCUSDT", "ETHUSDT"}
        mevcut_semboller = set(acik_pozisyonlar.keys())

        if (yeni_symbol in btc_eth_cift and
                mevcut_semboller.intersection(btc_eth_cift)):
            log.warning(
                f"[Risk] Korelasyon filtresi: "
                f"{yeni_symbol} açık pozisyonla ({mevcut_semboller}) çakışıyor"
            )
            return False
        return True

    # ── İSTATİSTİKLER ─────────────────────────────────────

    def istatistikler(self) -> dict:
        if not self.islem_gecmisi:
            return {
                "toplam_islem": 0, "kazanma_orani": 0.0,
                "toplam_kar": 0.0, "maks_drawdown": 0.0,
                "ortalama_kar": 0.0, "profit_factor": 0.0,
            }

        kazanlar = [i for i in self.islem_gecmisi if i.get("pnl", 0) > 0]
        kaybedenler = [i for i in self.islem_gecmisi if i.get("pnl", 0) <= 0]
        toplam_kar  = sum(i.get("pnl", 0) for i in kazanlar)
        toplam_kayip = abs(sum(i.get("pnl", 0) for i in kaybedenler))

        return {
            "toplam_islem":   len(self.islem_gecmisi),
            "kazanma_orani":  round(len(kazanlar) / len(self.islem_gecmisi), 4),
            "toplam_kar":     round(toplam_kar - toplam_kayip, 4),
            "maks_drawdown":  round(self.maks_drawdown, 4),
            "ortalama_kar":   round(
                (toplam_kar - toplam_kayip) / len(self.islem_gecmisi), 4
            ),
            "profit_factor":  round(
                toplam_kar / toplam_kayip if toplam_kayip > 0 else 0, 4
            ),
            "gross_profit":   round(toplam_kar, 4),
            "gross_loss":     round(toplam_kayip, 4),
        }

    def islem_kaydet(self, symbol: str, yon: str,
                     giris: float, cikis: float, miktar: float) -> dict:
        """Tamamlanan işlemi geçmişe kaydeder"""
        pnl = (cikis - giris) * miktar if yon == "long" else (giris - cikis) * miktar
        kayit = {
            "zaman":  datetime.now().isoformat(),
            "symbol": symbol,
            "yon":    yon,
            "giris":  giris,
            "cikis":  cikis,
            "miktar": miktar,
            "pnl":    round(pnl, 4),
        }
        self.islem_gecmisi.append(kayit)
        return kayit
