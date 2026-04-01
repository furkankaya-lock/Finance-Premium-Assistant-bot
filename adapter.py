"""
exchange/adapter.py
===================
Universal Exchange Adapter
Supports: Binance · Bybit · Bitget · OKX · MEXC · Gate.io
Single interface — swap exchange without changing any other code.
"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

log = logging.getLogger("CryptoBot.Exchange")


@dataclass
class OrderResult:
    success:    bool
    order_id:   str
    symbol:     str
    side:       str         # 'BUY' | 'SELL'
    order_type: str         # 'MARKET' | 'LIMIT'
    quantity:   float
    price:      float
    filled_qty: float
    avg_price:  float
    status:     str
    raw:        dict = None
    error:      str = ""


@dataclass
class BalanceResult:
    asset:     str
    free:      float
    locked:    float
    total:     float


@dataclass
class TickerResult:
    symbol:     str
    price:      float
    change_pct: float
    volume_24h: float
    high_24h:   float
    low_24h:    float


class ExchangeAdapter:
    """
    Unified exchange interface.
    All exchange-specific logic is isolated in client classes.
    The rest of the bot only interacts with this adapter.
    """

    SUPPORTED = {
        "binance": "exchange.binance_client.BinanceClient",
        "bybit":   "exchange.bybit_client.BybitClient",
        "bitget":  "exchange.bitget_client.BitgetClient",
        "okx":     "exchange.okx_client.OKXClient",
        "mexc":    "exchange.binance_client.BinanceClient",   # MEXC uses Binance-compatible API
        "gateio":  "exchange.binance_client.BinanceClient",   # Gate.io fallback
    }

    def __init__(self,
                 exchange:   str = "binance",
                 api_key:    str = "",
                 api_secret: str = "",
                 testnet:    bool = False,
                 demo_mode:  bool = False):
        self.exchange   = exchange.lower()
        self.api_key    = api_key
        self.api_secret = api_secret
        self.testnet    = testnet
        self.demo_mode  = demo_mode
        self._client    = None
        self._init_client()

    # ── INIT ──────────────────────────────────────────────────

    def _init_client(self):
        """Dynamically load the exchange client."""
        if self.demo_mode:
            log.info(f"[Exchange] Demo mode — no real API calls")
            return

        path = self.SUPPORTED.get(self.exchange)
        if not path:
            log.warning(f"[Exchange] {self.exchange} not supported, falling back to Binance")
            path = self.SUPPORTED["binance"]

        try:
            module_path, class_name = path.rsplit(".", 1)
            import importlib
            module = importlib.import_module(module_path)
            cls    = getattr(module, class_name)
            self._client = cls(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet,
            )
            log.info(f"[Exchange] {self.exchange.upper()} client initialized "
                     f"({'testnet' if self.testnet else 'mainnet'})")
        except Exception as e:
            log.error(f"[Exchange] Client init failed: {e}")
            self._client = None

    def switch_exchange(self, exchange: str, api_key: str, api_secret: str,
                        testnet: bool = False) -> bool:
        """Hot-swap exchange without restarting bot."""
        self.exchange   = exchange.lower()
        self.api_key    = api_key
        self.api_secret = api_secret
        self.testnet    = testnet
        self._init_client()
        return self._client is not None

    # ── MARKET DATA ───────────────────────────────────────────

    def get_price(self, symbol: str) -> float:
        """Current price for symbol."""
        if self.demo_mode or not self._client:
            return self._demo_price(symbol)
        try:
            return self._client.get_price(symbol)
        except Exception as e:
            log.warning(f"[Exchange] get_price({symbol}): {e}")
            return 0.0

    def get_ticker(self, symbol: str) -> TickerResult:
        """Full 24h ticker."""
        if self.demo_mode or not self._client:
            p = self._demo_price(symbol)
            return TickerResult(symbol=symbol, price=p, change_pct=0.0,
                                volume_24h=0, high_24h=p*1.02, low_24h=p*0.98)
        try:
            return self._client.get_ticker(symbol)
        except Exception as e:
            log.warning(f"[Exchange] get_ticker({symbol}): {e}")
            p = self._demo_price(symbol)
            return TickerResult(symbol=symbol, price=p, change_pct=0.0,
                                volume_24h=0, high_24h=p, low_24h=p)

    def get_klines(self, symbol: str, interval: str,
                   limit: int = 500) -> list:
        """OHLCV candlestick data."""
        if self.demo_mode or not self._client:
            return []
        try:
            return self._client.get_klines(symbol, interval, limit)
        except Exception as e:
            log.warning(f"[Exchange] get_klines({symbol}): {e}")
            return []

    def get_order_book(self, symbol: str, depth: int = 20) -> dict:
        """Order book bid/ask."""
        if self.demo_mode or not self._client:
            p = self._demo_price(symbol)
            return {"bids": [[p*0.999, 1.0]], "asks": [[p*1.001, 1.0]],
                    "bid_ask_ratio": 1.0, "buy_pressure": 50.0, "spread_pct": 0.1}
        try:
            return self._client.get_order_book(symbol, depth)
        except Exception as e:
            log.warning(f"[Exchange] get_order_book({symbol}): {e}")
            return {}

    def get_funding_rate(self, symbol: str) -> float:
        """Futures funding rate."""
        if self.demo_mode or not self._client:
            return 0.0001
        try:
            return self._client.get_funding_rate(symbol)
        except Exception as e:
            log.warning(f"[Exchange] get_funding_rate({symbol}): {e}")
            return 0.0

    # ── ACCOUNT ───────────────────────────────────────────────

    def get_balance(self, asset: str = "USDT") -> float:
        """Free balance for asset."""
        if self.demo_mode or not self._client:
            return 0.0
        try:
            return self._client.get_balance(asset)
        except Exception as e:
            log.warning(f"[Exchange] get_balance({asset}): {e}")
            return 0.0

    def get_all_balances(self) -> Dict[str, BalanceResult]:
        """All non-zero balances."""
        if self.demo_mode or not self._client:
            return {}
        try:
            return self._client.get_all_balances()
        except Exception as e:
            log.warning(f"[Exchange] get_all_balances: {e}")
            return {}

    # ── SPOT ORDERS ───────────────────────────────────────────

    def market_buy(self, symbol: str, quote_qty: float) -> OrderResult:
        """Spot market buy using quote amount (e.g., 50 USDT)."""
        if self.demo_mode:
            return self._demo_order(symbol, "BUY", quote_qty)
        if not self._client:
            return OrderResult(success=False, order_id="", symbol=symbol,
                               side="BUY", order_type="MARKET", quantity=0,
                               price=0, filled_qty=0, avg_price=0,
                               status="FAILED", error="No client")
        try:
            return self._client.market_buy(symbol, quote_qty)
        except Exception as e:
            log.error(f"[Exchange] market_buy({symbol}): {e}")
            return OrderResult(success=False, order_id="", symbol=symbol,
                               side="BUY", order_type="MARKET", quantity=0,
                               price=0, filled_qty=0, avg_price=0,
                               status="FAILED", error=str(e))

    def market_sell(self, symbol: str, quantity: float) -> OrderResult:
        """Spot market sell by base quantity."""
        if self.demo_mode:
            return self._demo_order(symbol, "SELL", quantity)
        if not self._client:
            return OrderResult(success=False, order_id="", symbol=symbol,
                               side="SELL", order_type="MARKET", quantity=0,
                               price=0, filled_qty=0, avg_price=0,
                               status="FAILED", error="No client")
        try:
            return self._client.market_sell(symbol, quantity)
        except Exception as e:
            log.error(f"[Exchange] market_sell({symbol}): {e}")
            return OrderResult(success=False, order_id="", symbol=symbol,
                               side="SELL", order_type="MARKET", quantity=0,
                               price=0, filled_qty=0, avg_price=0,
                               status="FAILED", error=str(e))

    # ── FUTURES ORDERS ────────────────────────────────────────

    def futures_open(self, symbol: str, side: str,
                     quantity: float, leverage: int = 10) -> OrderResult:
        """Open futures long or short."""
        if self.demo_mode:
            return self._demo_order(symbol, side, quantity)
        if not self._client:
            return OrderResult(success=False, order_id="", symbol=symbol,
                               side=side, order_type="MARKET", quantity=0,
                               price=0, filled_qty=0, avg_price=0,
                               status="FAILED", error="No client")
        try:
            return self._client.futures_open(symbol, side, quantity, leverage)
        except Exception as e:
            log.error(f"[Exchange] futures_open({symbol}, {side}): {e}")
            return OrderResult(success=False, order_id="", symbol=symbol,
                               side=side, order_type="MARKET", quantity=0,
                               price=0, filled_qty=0, avg_price=0,
                               status="FAILED", error=str(e))

    def futures_close(self, symbol: str, side: str,
                      quantity: float) -> OrderResult:
        """Close futures position (reduce only)."""
        if self.demo_mode:
            return self._demo_order(symbol, side, quantity)
        if not self._client:
            return OrderResult(success=False, order_id="", symbol=symbol,
                               side=side, order_type="MARKET", quantity=0,
                               price=0, filled_qty=0, avg_price=0,
                               status="FAILED", error="No client")
        try:
            return self._client.futures_close(symbol, side, quantity)
        except Exception as e:
            log.error(f"[Exchange] futures_close({symbol}): {e}")
            return OrderResult(success=False, order_id="", symbol=symbol,
                               side=side, order_type="MARKET", quantity=0,
                               price=0, filled_qty=0, avg_price=0,
                               status="FAILED", error=str(e))

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set futures leverage."""
        if self.demo_mode or not self._client:
            return True
        try:
            return self._client.set_leverage(symbol, leverage)
        except Exception as e:
            log.warning(f"[Exchange] set_leverage({symbol}, {leverage}): {e}")
            return False

    # ── DEMO HELPERS ──────────────────────────────────────────

    _DEMO_PRICES = {
        "BTCUSDT": 83420.0, "ETHUSDT": 1912.0,
        "SOLUSDT": 142.3,   "BNBUSDT": 605.0,
        "XRPUSDT": 0.51,    "ADAUSDT": 0.44,
    }

    def _demo_price(self, symbol: str) -> float:
        import random
        base = self._DEMO_PRICES.get(symbol, 100.0)
        return base * (1 + random.uniform(-0.001, 0.001))

    def _demo_order(self, symbol: str, side: str,
                    amount: float) -> OrderResult:
        import random, uuid
        price = self._demo_price(symbol)
        qty   = amount / price if side == "BUY" and amount > 1 else amount
        return OrderResult(
            success=True,
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol, side=side,
            order_type="MARKET", quantity=qty,
            price=price, filled_qty=qty, avg_price=price,
            status="FILLED",
        )

    # ── CONNECTIVITY ──────────────────────────────────────────

    def ping(self) -> bool:
        """Test connectivity."""
        if self.demo_mode: return True
        if not self._client: return False
        try:
            return self._client.ping()
        except Exception:
            return False

    @property
    def name(self) -> str:
        return self.exchange.upper()

    @property
    def is_ready(self) -> bool:
        return self.demo_mode or self._client is not None
