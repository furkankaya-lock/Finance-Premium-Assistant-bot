"""
exchange/bitget_client.py
=========================
Bitget API Client — Spot + Mix (Futures)
"""

import logging, time, hmac, hashlib, base64, json, requests
from .adapter import OrderResult, BalanceResult, TickerResult

log = logging.getLogger("CryptoBot.Bitget")

BASE = "https://api.bitget.com"


class BitgetClient:

    def __init__(self, api_key: str, api_secret: str, passphrase: str = "",
                 testnet: bool = False):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase  # Bitget requires passphrase
        self.session    = requests.Session()
        log.info("[Bitget] Client ready")

    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        msg = timestamp + method.upper() + path + body
        mac = hmac.new(self.api_secret.encode(), msg.encode(), hashlib.sha256)
        return base64.b64encode(mac.digest()).decode()

    def _headers(self, method: str, path: str, body: str = "") -> dict:
        ts = str(int(time.time() * 1000))
        return {
            "ACCESS-KEY":        self.api_key,
            "ACCESS-SIGN":       self._sign(ts, method, path, body),
            "ACCESS-TIMESTAMP":  ts,
            "ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type":      "application/json",
        }

    def _get(self, path: str, params: dict = None) -> dict:
        params = params or {}
        query = "?" + "&".join(f"{k}={v}" for k,v in params.items()) if params else ""
        r = self.session.get(BASE + path + query,
                             headers=self._headers("GET", path+query), timeout=10)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict) -> dict:
        body_str = json.dumps(body)
        r = self.session.post(BASE + path, data=body_str,
                              headers=self._headers("POST", path, body_str), timeout=10)
        r.raise_for_status()
        return r.json()

    def ping(self) -> bool:
        try:
            self._get("/api/v2/public/time")
            return True
        except Exception:
            return False

    def get_price(self, symbol: str) -> float:
        d = self._get("/api/v2/spot/market/tickers", {"symbol": symbol})
        return float(d["data"][0]["lastPr"])

    def get_ticker(self, symbol: str) -> TickerResult:
        d = self._get("/api/v2/spot/market/tickers", {"symbol": symbol})
        t = d["data"][0]
        return TickerResult(
            symbol=symbol, price=float(t["lastPr"]),
            change_pct=float(t.get("change24h", 0)) * 100,
            volume_24h=float(t.get("baseVolume", 0)),
            high_24h=float(t.get("high24h", 0)),
            low_24h=float(t.get("low24h", 0)),
        )

    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> list:
        imap = {"1m":"1min","5m":"5min","15m":"15min","1h":"1H","4h":"4H","1d":"1D"}
        d = self._get("/api/v2/spot/market/candles",
                      {"symbol": symbol, "granularity": imap.get(interval,"1H"), "limit": limit})
        return d.get("data", [])

    def get_order_book(self, symbol: str, depth: int = 20) -> dict:
        d = self._get("/api/v2/spot/market/orderbook", {"symbol": symbol, "limit": depth})
        bids = [[float(b[0]), float(b[1])] for b in d["data"]["bids"]]
        asks = [[float(a[0]), float(a[1])] for a in d["data"]["asks"]]
        bv = sum(b[1] for b in bids); av = sum(a[1] for a in asks); tot = bv+av+1e-9
        return {"bids":bids,"asks":asks,"bid_ask_ratio":round(bv/(av+1e-9),3),
                "buy_pressure":round(bv/tot*100,2),"spread_pct":0.0}

    def get_funding_rate(self, symbol: str) -> float:
        try:
            d = self._get("/api/v2/mix/market/current-fund-rate", {"symbol":symbol,"productType":"usdt-futures"})
            return float(d["data"]["fundingRate"]) * 100
        except Exception:
            return 0.0

    def get_balance(self, asset: str = "USDT") -> float:
        d = self._get("/api/v2/spot/account/assets", {"coin": asset})
        return float(d["data"][0].get("available", 0)) if d.get("data") else 0.0

    def get_all_balances(self):
        d = self._get("/api/v2/spot/account/assets")
        result = {}
        for item in d.get("data", []):
            free = float(item.get("available", 0))
            locked = float(item.get("frozen", 0))
            if free + locked > 0:
                result[item["coin"]] = BalanceResult(
                    asset=item["coin"], free=free, locked=locked, total=free+locked)
        return result

    def market_buy(self, symbol: str, quote_qty: float) -> OrderResult:
        d = self._post("/api/v2/spot/trade/place-order",
                       {"symbol":symbol,"side":"buy","orderType":"market",
                        "force":"gtc","size":str(round(quote_qty,2)),"quoteSize":str(round(quote_qty,2))})
        price = self.get_price(symbol)
        qty = quote_qty / price
        return OrderResult(success=True,order_id=d["data"]["orderId"],symbol=symbol,
                           side="BUY",order_type="MARKET",quantity=qty,price=price,
                           filled_qty=qty,avg_price=price,status="FILLED")

    def market_sell(self, symbol: str, quantity: float) -> OrderResult:
        d = self._post("/api/v2/spot/trade/place-order",
                       {"symbol":symbol,"side":"sell","orderType":"market",
                        "force":"gtc","size":str(round(quantity,6))})
        price = self.get_price(symbol)
        return OrderResult(success=True,order_id=d["data"]["orderId"],symbol=symbol,
                           side="SELL",order_type="MARKET",quantity=quantity,price=price,
                           filled_qty=quantity,avg_price=price,status="FILLED")

    def futures_open(self, symbol: str, side: str, quantity: float, leverage: int = 10) -> OrderResult:
        self.set_leverage(symbol, leverage)
        hold_side = "long" if side == "LONG" else "short"
        d = self._post("/api/v2/mix/order/place-order",
                       {"symbol":symbol,"productType":"usdt-futures","marginMode":"crossed",
                        "marginCoin":"USDT","side":"open_long" if side=="LONG" else "open_short",
                        "orderType":"market","size":str(round(quantity,6))})
        price = self.get_price(symbol)
        return OrderResult(success=True,order_id=d["data"]["orderId"],symbol=symbol,
                           side=side,order_type="MARKET",quantity=quantity,price=price,
                           filled_qty=quantity,avg_price=price,status="FILLED")

    def futures_close(self, symbol: str, side: str, quantity: float) -> OrderResult:
        close_side = "close_long" if side == "LONG" else "close_short"
        d = self._post("/api/v2/mix/order/place-order",
                       {"symbol":symbol,"productType":"usdt-futures","marginMode":"crossed",
                        "marginCoin":"USDT","side":close_side,
                        "orderType":"market","size":str(round(quantity,6)),"reduceOnly":"YES"})
        price = self.get_price(symbol)
        return OrderResult(success=True,order_id=d["data"]["orderId"],symbol=symbol,
                           side=side,order_type="MARKET",quantity=quantity,price=price,
                           filled_qty=quantity,avg_price=price,status="FILLED")

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        try:
            self._post("/api/v2/mix/account/set-leverage",
                       {"symbol":symbol,"productType":"usdt-futures",
                        "marginCoin":"USDT","leverage":str(leverage)})
            return True
        except Exception as e:
            log.warning(f"[Bitget] set_leverage: {e}")
            return False
