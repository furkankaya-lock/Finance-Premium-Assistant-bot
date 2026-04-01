"""
exchange/binance_client.py
==========================
Binance Spot + Futures Client
Also compatible with: MEXC, Gate.io (same API structure)
"""

import logging
from typing import Dict
from .adapter import OrderResult, BalanceResult, TickerResult

log = logging.getLogger("CryptoBot.Binance")


class BinanceClient:

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        from binance.client import Client
        from binance.exceptions import BinanceAPIException
        self.BinanceAPIException = BinanceAPIException
        if testnet:
            self.client = Client(api_key, api_secret,
                                  testnet=True)
        else:
            self.client = Client(api_key, api_secret)
        log.info(f"[Binance] Client ready ({'testnet' if testnet else 'mainnet'})")

    def ping(self) -> bool:
        try:
            self.client.ping()
            return True
        except Exception:
            return False

    def get_price(self, symbol: str) -> float:
        t = self.client.get_symbol_ticker(symbol=symbol)
        return float(t["price"])

    def get_ticker(self, symbol: str) -> TickerResult:
        t = self.client.get_ticker(symbol=symbol)
        return TickerResult(
            symbol=symbol,
            price=float(t["lastPrice"]),
            change_pct=float(t["priceChangePercent"]),
            volume_24h=float(t["volume"]),
            high_24h=float(t["highPrice"]),
            low_24h=float(t["lowPrice"]),
        )

    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> list:
        from binance.client import Client
        interval_map = {
            "1m": Client.KLINE_INTERVAL_1MINUTE,
            "5m": Client.KLINE_INTERVAL_5MINUTE,
            "15m": Client.KLINE_INTERVAL_15MINUTE,
            "1h": Client.KLINE_INTERVAL_1HOUR,
            "4h": Client.KLINE_INTERVAL_4HOUR,
            "1d": Client.KLINE_INTERVAL_1DAY,
        }
        return self.client.get_klines(
            symbol=symbol,
            interval=interval_map.get(interval, Client.KLINE_INTERVAL_1HOUR),
            limit=limit,
        )

    def get_order_book(self, symbol: str, depth: int = 20) -> dict:
        ob = self.client.get_order_book(symbol=symbol, limit=depth)
        bids = [[float(b[0]), float(b[1])] for b in ob["bids"]]
        asks = [[float(a[0]), float(a[1])] for a in ob["asks"]]
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
        rates = self.client.futures_funding_rate(symbol=symbol, limit=1)
        return float(rates[0]["fundingRate"]) * 100 if rates else 0.0

    def get_balance(self, asset: str = "USDT") -> float:
        b = self.client.get_asset_balance(asset=asset)
        return float(b["free"]) if b else 0.0

    def get_all_balances(self) -> Dict[str, BalanceResult]:
        info = self.client.get_account()
        result = {}
        for b in info["balances"]:
            free = float(b["free"])
            locked = float(b["locked"])
            if free + locked > 0:
                result[b["asset"]] = BalanceResult(
                    asset=b["asset"], free=free,
                    locked=locked, total=free+locked,
                )
        return result

    def market_buy(self, symbol: str, quote_qty: float) -> OrderResult:
        order = self.client.order_market_buy(
            symbol=symbol, quoteOrderQty=round(quote_qty, 2)
        )
        fills = order.get("fills", [{}])
        avg_p = float(fills[0].get("price", 0)) if fills else 0
        return OrderResult(
            success=True, order_id=str(order["orderId"]),
            symbol=symbol, side="BUY", order_type="MARKET",
            quantity=float(order["origQty"]),
            price=avg_p, filled_qty=float(order["executedQty"]),
            avg_price=avg_p, status=order["status"],
        )

    def market_sell(self, symbol: str, quantity: float) -> OrderResult:
        order = self.client.order_market_sell(
            symbol=symbol, quantity=round(quantity, 6)
        )
        fills = order.get("fills", [{}])
        avg_p = float(fills[0].get("price", 0)) if fills else 0
        return OrderResult(
            success=True, order_id=str(order["orderId"]),
            symbol=symbol, side="SELL", order_type="MARKET",
            quantity=float(order["origQty"]),
            price=avg_p, filled_qty=float(order["executedQty"]),
            avg_price=avg_p, status=order["status"],
        )

    def futures_open(self, symbol: str, side: str,
                     quantity: float, leverage: int = 10) -> OrderResult:
        self.set_leverage(symbol, leverage)
        binance_side = "BUY" if side == "LONG" else "SELL"
        order = self.client.futures_create_order(
            symbol=symbol, side=binance_side,
            type="MARKET", quantity=round(quantity, 6),
        )
        avg_p = float(order.get("avgPrice", 0))
        return OrderResult(
            success=True, order_id=str(order["orderId"]),
            symbol=symbol, side=side, order_type="MARKET",
            quantity=float(order["origQty"]),
            price=avg_p, filled_qty=float(order["executedQty"]),
            avg_price=avg_p, status=order["status"],
        )

    def futures_close(self, symbol: str, side: str,
                      quantity: float) -> OrderResult:
        close_side = "SELL" if side == "LONG" else "BUY"
        order = self.client.futures_create_order(
            symbol=symbol, side=close_side,
            type="MARKET", quantity=round(quantity, 6),
            reduceOnly=True,
        )
        avg_p = float(order.get("avgPrice", 0))
        return OrderResult(
            success=True, order_id=str(order["orderId"]),
            symbol=symbol, side=side, order_type="MARKET",
            quantity=float(order["origQty"]),
            price=avg_p, filled_qty=float(order["executedQty"]),
            avg_price=avg_p, status=order["status"],
        )

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        try:
            self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
            return True
        except Exception as e:
            log.warning(f"[Binance] set_leverage: {e}")
            return False
