"""
exchange/okx_client.py
======================
OKX V5 API Client — Spot + Swap (Perpetual Futures)
"""

import logging, time, hmac, hashlib, base64, json, requests
from datetime import datetime, timezone
from .adapter import OrderResult, BalanceResult, TickerResult

log = logging.getLogger("CryptoBot.OKX")

BASE         = "https://www.okx.com"
BASE_TESTNET = "https://www.okx.com"  # OKX uses same URL, header flag for testnet


class OKXClient:

    def __init__(self, api_key: str, api_secret: str, passphrase: str = "",
                 testnet: bool = False):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.testnet    = testnet
        self.session    = requests.Session()
        log.info(f"[OKX] Client ready ({'testnet' if testnet else 'mainnet'})")

    def _iso_timestamp(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        msg = timestamp + method.upper() + path + body
        mac = hmac.new(self.api_secret.encode(), msg.encode(), hashlib.sha256)
        return base64.b64encode(mac.digest()).decode()

    def _headers(self, method: str, path: str, body: str = "") -> dict:
        ts = self._iso_timestamp()
        h = {
            "OK-ACCESS-KEY":        self.api_key,
            "OK-ACCESS-SIGN":       self._sign(ts, method, path, body),
            "OK-ACCESS-TIMESTAMP":  ts,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type":         "application/json",
        }
        if self.testnet:
            h["x-simulated-trading"] = "1"
        return h

    def _get(self, path: str, params: dict = None) -> dict:
        params = params or {}
        query = "?" + "&".join(f"{k}={v}" for k,v in params.items()) if params else ""
        full_path = path + query
        r = self.session.get(BASE + full_path,
                             headers=self._headers("GET", full_path), timeout=10)
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
            self._get("/api/v5/public/time")
            return True
        except Exception:
            return False

    def get_price(self, symbol: str) -> float:
        okx_sym = symbol.replace("USDT", "-USDT")
        d = self._get("/api/v5/market/ticker", {"instId": okx_sym})
        return float(d["data"][0]["last"])

    def get_ticker(self, symbol: str) -> TickerResult:
        okx_sym = symbol.replace("USDT", "-USDT")
        d = self._get("/api/v5/market/ticker", {"instId": okx_sym})
        t = d["data"][0]
        open_24 = float(t.get("open24h", t["last"]))
        chg_pct = (float(t["last"]) - open_24) / open_24 * 100 if open_24 else 0
        return TickerResult(
            symbol=symbol, price=float(t["last"]),
            change_pct=round(chg_pct, 3),
            volume_24h=float(t.get("vol24h", 0)),
            high_24h=float(t.get("high24h", 0)),
            low_24h=float(t.get("low24h", 0)),
        )

    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> list:
        imap = {"1m":"1m","5m":"5m","15m":"15m","1h":"1H","4h":"4H","1d":"1D"}
        okx_sym = symbol.replace("USDT", "-USDT")
        d = self._get("/api/v5/market/candles",
                      {"instId": okx_sym, "bar": imap.get(interval,"1H"), "limit": limit})
        return d.get("data", [])

    def get_order_book(self, symbol: str, depth: int = 20) -> dict:
        okx_sym = symbol.replace("USDT", "-USDT")
        d = self._get("/api/v5/market/books", {"instId": okx_sym, "sz": depth})
        ob = d["data"][0]
        bids = [[float(b[0]), float(b[1])] for b in ob["bids"]]
        asks = [[float(a[0]), float(a[1])] for a in ob["asks"]]
        bv = sum(b[1] for b in bids); av = sum(a[1] for a in asks); tot = bv+av+1e-9
        sp = (asks[0][0]-bids[0][0])/bids[0][0]*100 if bids and asks else 0
        return {"bids":bids,"asks":asks,"bid_ask_ratio":round(bv/(av+1e-9),3),
                "buy_pressure":round(bv/tot*100,2),"spread_pct":round(sp,4)}

    def get_funding_rate(self, symbol: str) -> float:
        try:
            okx_sym = symbol.replace("USDT", "-USDT-SWAP")
            d = self._get("/api/v5/public/funding-rate", {"instId": okx_sym})
            return float(d["data"][0]["fundingRate"]) * 100
        except Exception:
            return 0.0

    def get_balance(self, asset: str = "USDT") -> float:
        d = self._get("/api/v5/account/balance", {"ccy": asset})
        details = d["data"][0].get("details", [])
        for det in details:
            if det["ccy"] == asset:
                return float(det.get("availBal", 0))
        return 0.0

    def get_all_balances(self):
        d = self._get("/api/v5/account/balance")
        result = {}
        for det in d["data"][0].get("details", []):
            free = float(det.get("availBal", 0))
            total = float(det.get("eq", 0))
            if total > 0:
                result[det["ccy"]] = BalanceResult(
                    asset=det["ccy"], free=free,
                    locked=total-free, total=total)
        return result

    def market_buy(self, symbol: str, quote_qty: float) -> OrderResult:
        okx_sym = symbol.replace("USDT", "-USDT")
        price = self.get_price(symbol)
        qty = round(quote_qty / price, 6)
        d = self._post("/api/v5/trade/order",
                       {"instId":okx_sym,"tdMode":"cash","side":"buy",
                        "ordType":"market","sz":str(qty)})
        return OrderResult(success=True,order_id=d["data"][0]["ordId"],symbol=symbol,
                           side="BUY",order_type="MARKET",quantity=qty,price=price,
                           filled_qty=qty,avg_price=price,status="FILLED")

    def market_sell(self, symbol: str, quantity: float) -> OrderResult:
        okx_sym = symbol.replace("USDT", "-USDT")
        price = self.get_price(symbol)
        d = self._post("/api/v5/trade/order",
                       {"instId":okx_sym,"tdMode":"cash","side":"sell",
                        "ordType":"market","sz":str(round(quantity,6))})
        return OrderResult(success=True,order_id=d["data"][0]["ordId"],symbol=symbol,
                           side="SELL",order_type="MARKET",quantity=quantity,price=price,
                           filled_qty=quantity,avg_price=price,status="FILLED")

    def futures_open(self, symbol: str, side: str, quantity: float, leverage: int = 10) -> OrderResult:
        self.set_leverage(symbol, leverage)
        okx_sym = symbol.replace("USDT", "-USDT-SWAP")
        okx_side = "buy" if side == "LONG" else "sell"
        pos_side = "long" if side == "LONG" else "short"
        d = self._post("/api/v5/trade/order",
                       {"instId":okx_sym,"tdMode":"cross","side":okx_side,
                        "posSide":pos_side,"ordType":"market","sz":str(round(quantity,6))})
        price = self.get_price(symbol)
        return OrderResult(success=True,order_id=d["data"][0]["ordId"],symbol=symbol,
                           side=side,order_type="MARKET",quantity=quantity,price=price,
                           filled_qty=quantity,avg_price=price,status="FILLED")

    def futures_close(self, symbol: str, side: str, quantity: float) -> OrderResult:
        okx_sym = symbol.replace("USDT", "-USDT-SWAP")
        close_side = "sell" if side == "LONG" else "buy"
        pos_side = "long" if side == "LONG" else "short"
        d = self._post("/api/v5/trade/order",
                       {"instId":okx_sym,"tdMode":"cross","side":close_side,
                        "posSide":pos_side,"ordType":"market","sz":str(round(quantity,6)),
                        "reduceOnly":True})
        price = self.get_price(symbol)
        return OrderResult(success=True,order_id=d["data"][0]["ordId"],symbol=symbol,
                           side=side,order_type="MARKET",quantity=quantity,price=price,
                           filled_qty=quantity,avg_price=price,status="FILLED")

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        try:
            okx_sym = symbol.replace("USDT", "-USDT-SWAP")
            for mg in ["long","short"]:
                self._post("/api/v5/account/set-leverage",
                           {"instId":okx_sym,"lever":str(leverage),"mgnMode":"cross","posSide":mg})
            return True
        except Exception as e:
            log.warning(f"[OKX] set_leverage: {e}")
            return False
