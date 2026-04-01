"""
exchange/bybit_client.py
========================
Bybit V5 API Client — Spot + Derivatives
"""

import logging
import time
import hmac, hashlib, json
import requests
from typing import Dict
from .adapter import OrderResult, BalanceResult, TickerResult

log = logging.getLogger("CryptoBot.Bybit")

BASE_URL      = "https://api.bybit.com"
BASE_TESTNET  = "https://api-testnet.bybit.com"


class BybitClient:

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.base       = BASE_TESTNET if testnet else BASE_URL
        self.session    = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        log.info(f"[Bybit] Client ready ({'testnet' if testnet else 'mainnet'})")

    # ── AUTH ──────────────────────────────────────────────────

    def _sign(self, params: dict) -> dict:
        ts = str(int(time.time() * 1000))
        recv_window = "5000"
        query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        sign_str = ts + self.api_key + recv_window + query
        signature = hmac.new(
            self.api_secret.encode(), sign_str.encode(), hashlib.sha256
        ).hexdigest()
        return {
            "X-BAPI-API-KEY":     self.api_key,
            "X-BAPI-TIMESTAMP":   ts,
            "X-BAPI-SIGN":        signature,
            "X-BAPI-RECV-WINDOW": recv_window,
        }

    def _get(self, path: str, params: dict = None, auth: bool = False) -> dict:
        params = params or {}
        headers = self._sign(params) if auth else {}
        r = self.session.get(self.base + path, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict) -> dict:
        ts = str(int(time.time() * 1000))
        body_str = json.dumps(body)
        sign_str = ts + self.api_key + "5000" + body_str
        sig = hmac.new(self.api_secret.encode(), sign_str.encode(), hashlib.sha256).hexdigest()
        headers = {
            "X-BAPI-API-KEY":     self.api_key,
            "X-BAPI-TIMESTAMP":   ts,
            "X-BAPI-SIGN":        sig,
            "X-BAPI-RECV-WINDOW": "5000",
        }
        r = self.session.post(self.base + path, json=body, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()

    # ── PUBLIC ────────────────────────────────────────────────

    def ping(self) -> bool:
        try:
            self._get("/v5/market/time")
            return True
        except Exception:
            return False

    def get_price(self, symbol: str) -> float:
        d = self._get("/v5/market/tickers", {"category": "spot", "symbol": symbol})
        return float(d["result"]["list"][0]["lastPrice"])

    def get_ticker(self, symbol: str) -> TickerResult:
        d = self._get("/v5/market/tickers", {"category": "spot", "symbol": symbol})
        t = d["result"]["list"][0]
        return TickerResult(
            symbol=symbol,
            price=float(t["lastPrice"]),
            change_pct=float(t.get("price24hPcnt", 0)) * 100,
            volume_24h=float(t.get("volume24h", 0)),
            high_24h=float(t.get("highPrice24h", 0)),
            low_24h=float(t.get("lowPrice24h", 0)),
        )

    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> list:
        interval_map = {"1m":"1","5m":"5","15m":"15","1h":"60","4h":"240","1d":"D"}
        d = self._get("/v5/market/kline", {
            "category": "spot", "symbol": symbol,
            "interval": interval_map.get(interval, "60"), "limit": limit,
        })
        return d["result"].get("list", [])

    def get_order_book(self, symbol: str, depth: int = 20) -> dict:
        d = self._get("/v5/market/orderbook", {"category": "spot", "symbol": symbol, "limit": depth})
        bids = [[float(b[0]), float(b[1])] for b in d["result"]["b"]]
        asks = [[float(a[0]), float(a[1])] for a in d["result"]["a"]]
        bid_vol = sum(b[1] for b in bids)
        ask_vol = sum(a[1] for a in asks)
        total = bid_vol + ask_vol + 1e-9
        spread_pct = (asks[0][0] - bids[0][0]) / bids[0][0] * 100 if bids and asks else 0
        return {
            "bids": bids, "asks": asks,
            "bid_ask_ratio": round(bid_vol / (ask_vol + 1e-9), 3),
            "buy_pressure":  round(bid_vol / total * 100, 2),
            "spread_pct":    round(spread_pct, 4),
        }

    def get_funding_rate(self, symbol: str) -> float:
        try:
            d = self._get("/v5/market/funding/history", {"category": "linear", "symbol": symbol, "limit": 1})
            return float(d["result"]["list"][0]["fundingRate"]) * 100
        except Exception:
            return 0.0

    # ── ACCOUNT ───────────────────────────────────────────────

    def get_balance(self, asset: str = "USDT") -> float:
        d = self._get("/v5/account/wallet-balance", {"accountType": "UNIFIED", "coin": asset}, auth=True)
        coins = d["result"]["list"][0].get("coin", [])
        for c in coins:
            if c["coin"] == asset:
                return float(c.get("availableToWithdraw", 0))
        return 0.0

    def get_all_balances(self) -> Dict[str, BalanceResult]:
        d = self._get("/v5/account/wallet-balance", {"accountType": "UNIFIED"}, auth=True)
        result = {}
        for c in d["result"]["list"][0].get("coin", []):
            free = float(c.get("availableToWithdraw", 0))
            total = float(c.get("walletBalance", 0))
            if total > 0:
                result[c["coin"]] = BalanceResult(
                    asset=c["coin"], free=free,
                    locked=total-free, total=total,
                )
        return result

    # ── ORDERS ────────────────────────────────────────────────

    def market_buy(self, symbol: str, quote_qty: float) -> OrderResult:
        body = {"category": "spot", "symbol": symbol, "side": "Buy",
                "orderType": "Market", "qty": str(round(quote_qty, 2)),
                "marketUnit": "quoteCoin"}
        d = self._post("/v5/order/create", body)
        oid = d["result"]["orderId"]
        price = self.get_price(symbol)
        qty = quote_qty / price
        return OrderResult(success=True, order_id=oid, symbol=symbol,
                           side="BUY", order_type="MARKET", quantity=qty,
                           price=price, filled_qty=qty, avg_price=price, status="FILLED")

    def market_sell(self, symbol: str, quantity: float) -> OrderResult:
        body = {"category": "spot", "symbol": symbol, "side": "Sell",
                "orderType": "Market", "qty": str(round(quantity, 6))}
        d = self._post("/v5/order/create", body)
        price = self.get_price(symbol)
        return OrderResult(success=True, order_id=d["result"]["orderId"],
                           symbol=symbol, side="SELL", order_type="MARKET",
                           quantity=quantity, price=price, filled_qty=quantity,
                           avg_price=price, status="FILLED")

    def futures_open(self, symbol: str, side: str,
                     quantity: float, leverage: int = 10) -> OrderResult:
        self.set_leverage(symbol, leverage)
        bybit_side = "Buy" if side == "LONG" else "Sell"
        body = {"category": "linear", "symbol": symbol, "side": bybit_side,
                "orderType": "Market", "qty": str(round(quantity, 6))}
        d = self._post("/v5/order/create", body)
        price = self.get_price(symbol)
        return OrderResult(success=True, order_id=d["result"]["orderId"],
                           symbol=symbol, side=side, order_type="MARKET",
                           quantity=quantity, price=price, filled_qty=quantity,
                           avg_price=price, status="FILLED")

    def futures_close(self, symbol: str, side: str, quantity: float) -> OrderResult:
        close_side = "Sell" if side == "LONG" else "Buy"
        body = {"category": "linear", "symbol": symbol, "side": close_side,
                "orderType": "Market", "qty": str(round(quantity, 6)),
                "reduceOnly": True}
        d = self._post("/v5/order/create", body)
        price = self.get_price(symbol)
        return OrderResult(success=True, order_id=d["result"]["orderId"],
                           symbol=symbol, side=side, order_type="MARKET",
                           quantity=quantity, price=price, filled_qty=quantity,
                           avg_price=price, status="FILLED")

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        try:
            self._post("/v5/position/set-leverage", {
                "category": "linear", "symbol": symbol,
                "buyLeverage": str(leverage), "sellLeverage": str(leverage),
            })
            return True
        except Exception as e:
            log.warning(f"[Bybit] set_leverage: {e}")
            return False
