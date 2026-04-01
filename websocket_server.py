"""
websocket_server.py
===================
Real-time WebSocket Bridge — Bot <-> Dashboard
GitHub/Server compatible: dynamic hostname resolution
Supports: Local · VPS · Docker · Render · Railway · Heroku

Usage:
  python websocket_server.py
  Or imported: from websocket_server import start_server_thread, push_price ...
"""

import asyncio, json, logging, os, threading, time
from datetime import datetime

log = logging.getLogger("CryptoBot.WS")

try:
    import websockets
    from websockets.server import serve
    WS_OK = True
except ImportError:
    WS_OK = False
    log.warning("[WS] websockets not installed: pip install websockets==12.0")

WS_HOST = os.getenv("WS_HOST", "0.0.0.0")
WS_PORT = int(os.getenv("WS_PORT", "8765"))


class BotState:
    def __init__(self):
        self._lock = threading.Lock()
        self._clients = set()
        self._state = self._default()

    @staticmethod
    def _default():
        return {
            "status":"stopped","mode":"spot","exchange":"binance","leverage":"1x",
            "scan_tick":0,"prices":{},"indicators":{},"signals":{},"positions":[],
            "stats":{"daily":_es(),"weekly":_es(),"monthly":_es(),"yearly":_es(),"all":_es()},
            "agent_thoughts":[],"log_entries":[],"ict":{},"sr":{},"macro":{},
            "memory":{},"news":{},"xai":{},"last_update":None,
        }

    def set(self, k, v):
        with self._lock:
            self._state[k] = v
            self._state["last_update"] = datetime.now().isoformat()

    def set_price(self, sym, price, chg=0.0):
        with self._lock:
            self._state["prices"][sym] = {"price":round(price,6),"change_pct":round(chg,3),"ts":datetime.now().isoformat()}
            self._state["scan_tick"] += 1
            self._state["last_update"] = datetime.now().isoformat()

    def set_indicator(self, sym, data):
        with self._lock: self._state["indicators"][sym] = data
    def set_signal(self, sym, sig, conf=0.0):
        with self._lock: self._state["signals"][sym] = {"signal":sig,"confidence":conf,"ts":datetime.now().isoformat()}
    def add_log(self, t, icon, msg):
        with self._lock:
            self._state["log_entries"].insert(0,{"type":t,"icon":icon,"msg":msg,"time":datetime.now().strftime("%H:%M:%S")})
            self._state["log_entries"] = self._state["log_entries"][:60]
    def add_thought(self, icon, text):
        with self._lock:
            self._state["agent_thoughts"].insert(0,{"icon":icon,"text":text,"time":datetime.now().strftime("%H:%M:%S")})
            self._state["agent_thoughts"] = self._state["agent_thoughts"][:40]
    def set_positions(self, p):
        with self._lock: self._state["positions"] = p
    def set_stats(self, period, s):
        with self._lock: self._state["stats"][period] = s
    def set_ict(self, sym, d):
        with self._lock: self._state["ict"][sym] = d
    def set_macro(self, d):
        with self._lock: self._state["macro"] = d
    def set_xai(self, sym, d):
        with self._lock: self._state["xai"][sym] = d
    def set_status(self, s):
        with self._lock: self._state["status"] = s

    def snapshot(self):
        with self._lock: return json.loads(json.dumps(self._state))
    def add_client(self, ws):
        with self._lock: self._clients.add(ws)
    def remove_client(self, ws):
        with self._lock: self._clients.discard(ws)
    def clients(self):
        with self._lock: return set(self._clients)

def _es():
    return {"total_trades":0,"wins":0,"losses":0,"win_rate":0.0,
            "total_pnl":0.0,"pnl_pct":0.0,"portfolio":0.0,"profit_factor":0.0}

BOT_STATE = BotState()

async def _handle(ws):
    BOT_STATE.add_client(ws)
    try:
        await ws.send(json.dumps({"type":"full_state","data":BOT_STATE.snapshot()}))
        async for raw in ws:
            try:
                msg = json.loads(raw)
                if msg.get("cmd") == "ping":
                    await ws.send(json.dumps({"type":"pong","ts":datetime.now().isoformat()}))
                elif msg.get("cmd") == "get_state":
                    await ws.send(json.dumps({"type":"full_state","data":BOT_STATE.snapshot()}))
            except Exception: pass
    except Exception: pass
    finally: BOT_STATE.remove_client(ws)

async def _broadcast():
    last_tick = -1
    while True:
        await asyncio.sleep(1)
        clients = BOT_STATE.clients()
        if not clients: continue
        tick = BOT_STATE.snapshot().get("scan_tick", 0)
        if tick != last_tick:
            last_tick = tick
            msg = json.dumps({"type":"update","data":BOT_STATE.snapshot()})
        else:
            msg = json.dumps({"type":"heartbeat","ts":datetime.now().isoformat()})
        dead = set()
        for ws in clients:
            try: await ws.send(msg)
            except Exception: dead.add(ws)
        for ws in dead: BOT_STATE.remove_client(ws)

async def _run():
    log.info(f"[WS] Server ws://{WS_HOST}:{WS_PORT}")
    async with serve(_handle, WS_HOST, WS_PORT, ping_interval=20, ping_timeout=10):
        await _broadcast()

def start_server_thread():
    if not WS_OK:
        log.warning("[WS] websockets missing — dashboard demo mode only")
        return None
    def _t():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try: loop.run_until_complete(_run())
        except Exception as e: log.error(f"[WS] Crashed: {e}")
    t = threading.Thread(target=_t, daemon=True, name="WS")
    t.start()
    time.sleep(0.5)
    return t

# Push helpers
def push_price(sym, price, chg=0.0):      BOT_STATE.set_price(sym, price, chg)
def push_indicator(sym, data):             BOT_STATE.set_indicator(sym, data)
def push_signal(sym, sig, conf=0.0):       BOT_STATE.set_signal(sym, sig, conf)
def push_log(t, icon, msg):               BOT_STATE.add_log(t, icon, msg)
def push_agent_thought(icon, text):        BOT_STATE.add_thought(icon, text)
def push_positions(positions):             BOT_STATE.set_positions(positions)
def push_stats(period, stats):             BOT_STATE.set_stats(period, stats)
def push_ict(sym, data):                   BOT_STATE.set_ict(sym, data)
def push_macro(data):                      BOT_STATE.set_macro(data)
def push_xai(sym, data):                   BOT_STATE.set_xai(sym, data)
def push_status(s):                        BOT_STATE.set_status(s)

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
    if not WS_OK: print("pip install websockets==12.0"); sys.exit(1)
    print(f"CryptoBot WS Server → ws://localhost:{WS_PORT}")
    asyncio.run(_run())
