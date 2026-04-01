"""
╔══════════════════════════════════════════════════════════════════╗
║       HYBRID AI CRYPTO TRADING BOT  v5.0                        ║
║       LSTM · RF · Claude Agent · ICT · S/R · Futures            ║
║       Long/Short · Position Reversal · Partial Exit             ║
║       Long-term Memory · Macro Context (NEW)                    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os, sys, time, logging, json, threading
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from binance.client import Client
from binance.exceptions import BinanceAPIException

from data.collector               import tam_veri_paketi as fetch_market_data
from data.indicators              import hesapla_hepsi as compute_indicators, ozellik_vektoru as feature_vector
from ai_engine.lstm_model         import LSTMFiyatModeli as LSTMModel
from ai_engine.rf_model           import RandomForestSinyalModeli as RFModel
from ai_engine.claude_advisor     import (istemci_baslat as init_claude,
                                           karar_al as get_claude_decision,
                                           oylama_sistemi as voting_system,
                                           set_memory as set_claude_memory,
                                           analiz_rejim as analyze_regime)
from ai_engine.agent              import HybridAIAgent, AgentToolExecutor
from ai_engine.ict_engine         import ICTEngine
from ai_engine.support_resistance import SupportResistanceEngine
from ai_engine.futures_engine     import FuturesEngine
from risk.manager                 import RiskYoneticisi as RiskManager
from memory.memory_manager        import MemoryManager
from memory.macro_context         import MacroContextEngine


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

class Config:
    # API Keys
    BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY",    "YOUR_BINANCE_API_KEY")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "YOUR_BINANCE_SECRET")
    CLAUDE_API_KEY     = os.getenv("ANTHROPIC_API_KEY",  "YOUR_ANTHROPIC_API_KEY")
    CRYPTOPANIC_TOKEN  = os.getenv("CRYPTOPANIC_TOKEN",  "")

    # Trading pairs
    SYMBOLS   = ["BTCUSDT", "ETHUSDT"]
    INTERVAL  = Client.KLINE_INTERVAL_1HOUR

    # Trading mode
    TRADING_MODE     = "spot"      # "spot" | "futures" | "demo"
    DEFAULT_LEVERAGE = 10

    # AI
    RETRAIN_EVERY_HOURS = 24
    MIN_SIGNAL_CONFIDENCE = 0.60
    VOTING_THRESHOLD    = 2        # Out of 3 models

    # Agent
    AGENT_ENABLED         = True
    AGENT_MIN_TRIGGER     = 0.60

    # ICT
    ICT_ENABLED           = True
    ICT_MIN_CONFIDENCE    = 0.40   # ICT adds to final vote

    # Risk
    INITIAL_CAPITAL       = 1000.0
    MAX_DAILY_LOSS_PCT    = 0.05
    MAX_POSITION_PCT      = 0.20
    ATR_SL_MULTIPLIER     = 2.0
    ATR_TP_MULTIPLIER     = 3.0
    MAX_OPEN_POSITIONS    = 2

    # Scan interval
    SCAN_INTERVAL_SECONDS = 60

    # Memory & Macro
    MEMORY_ENABLED        = True
    MACRO_UPDATE_MINUTES  = 15   # Fetch macro data every 15 min
    REGIME_UPDATE_HOURS   = 6    # Re-analyze regime every 6h
    DAILY_SUMMARY_HOUR    = 23   # Hour to store daily summary (23:xx)

    # Fallback (classic strategy)
    RSI_OVERSOLD    = 30
    RSI_OVERBOUGHT  = 70


# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────

os.makedirs("logs",   exist_ok=True)
os.makedirs("models", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-24s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("logs/bot.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("CryptoBot.Main")


# ─────────────────────────────────────────────────────────────
# REAL-TIME STATS
# ─────────────────────────────────────────────────────────────

class StatsStore:
    """Thread-safe trade statistics with time-based filtering."""

    def __init__(self):
        self._lock     = threading.Lock()
        self.trades: list = []
        self.start_capital = 0.0
        self.current_portfolio = 0.0
        self.agent_thoughts: list = []

    def add_trade(self, record: dict):
        with self._lock:
            self.trades.append({**record, 'timestamp': datetime.now().isoformat()})

    def add_thought(self, icon: str, text: str):
        with self._lock:
            self.agent_thoughts.append({'icon': icon, 'text': text,
                                        'time': datetime.now().isoformat()})
            if len(self.agent_thoughts) > 100:
                self.agent_thoughts.pop(0)

    def update_portfolio(self, value: float):
        with self._lock:
            self.current_portfolio = value

    def compute(self, period: str = 'all', mode: str = 'all') -> dict:
        with self._lock:
            now = datetime.now()
            def match(t):
                try:    ts = datetime.fromisoformat(t['timestamp'])
                except: return True
                if period == 'daily':   return ts.date() == now.date()
                if period == 'weekly':  return (now - ts).days <= 7
                if period == 'monthly': return ts.month == now.month and ts.year == now.year
                if period == 'yearly':  return ts.year == now.year
                return True
            trades = [t for t in self.trades
                      if match(t) and (mode == 'all' or t.get('mode') == mode)]
            if not trades:
                return {'total': 0, 'win_rate': 0.0, 'total_pnl': 0.0,
                        'pnl_pct': 0.0, 'portfolio': self.current_portfolio,
                        'profit_factor': 0.0, 'wins': 0, 'losses': 0}
            wins   = [t for t in trades if t.get('pnl', 0) > 0]
            losses = [t for t in trades if t.get('pnl', 0) <= 0]
            total_pnl = sum(t.get('pnl', 0) for t in trades)
            gp = sum(t.get('pnl', 0) for t in wins)
            gl = abs(sum(t.get('pnl', 0) for t in losses))
            pnl_pct = total_pnl / self.start_capital * 100 if self.start_capital else 0
            return {
                'total':          len(trades),
                'wins':           len(wins),
                'losses':         len(losses),
                'win_rate':       round(len(wins) / len(trades) * 100, 1),
                'total_pnl':      round(total_pnl, 4),
                'pnl_pct':        round(pnl_pct, 2),
                'portfolio':      round(self.current_portfolio, 2),
                'profit_factor':  round(gp / gl, 3) if gl > 0 else 0,
                'best_trade':     round(max((t.get('pnl', 0) for t in trades), default=0), 4),
                'worst_trade':    round(min((t.get('pnl', 0) for t in trades), default=0), 4),
            }


# ─────────────────────────────────────────────────────────────
# FALLBACK SIGNAL
# ─────────────────────────────────────────────────────────────

def classic_signal(df_indicators) -> str:
    current  = df_indicators.iloc[-1]
    previous = df_indicators.iloc[-2]
    ma_buy  = previous['ema_9'] <= previous['ema_21'] and current['ema_9'] > current['ema_21']
    ma_sell = previous['ema_9'] >= previous['ema_21'] and current['ema_9'] < current['ema_21']
    if ma_buy  and current['rsi_14'] < Config.RSI_OVERSOLD:  return 'AL'
    if ma_sell and current['rsi_14'] > Config.RSI_OVERBOUGHT: return 'SAT'
    return 'BEKLE'


# ─────────────────────────────────────────────────────────────
# MAIN BOT
# ─────────────────────────────────────────────────────────────

class HybridAIBot:

    def __init__(self):
        self.cfg    = Config()
        self.client = None
        self.stats  = StatsStore()
        self.risk   = RiskManager(
            baslangic_sermaye = self.cfg.INITIAL_CAPITAL,
            maks_gunluk_kayip = self.cfg.MAX_DAILY_LOSS_PCT,
            maks_pozisyon_pct = self.cfg.MAX_POSITION_PCT,
            atr_sl_carpan     = self.cfg.ATR_SL_MULTIPLIER,
            atr_tp_carpan     = self.cfg.ATR_TP_MULTIPLIER,
            max_acik_pozisyon = self.cfg.MAX_OPEN_POSITIONS,
            min_guven         = self.cfg.MIN_SIGNAL_CONFIDENCE,
        )
        # ML models
        self.lstm_models: dict = {}
        self.rf_models:   dict = {}

        # Specialized engines
        self.ict_engine  = ICTEngine()
        self.sr_engine   = SupportResistanceEngine()
        self.futures_eng: FuturesEngine = None

        # Memory + Macro (NEW v5.0)
        self.memory  = MemoryManager()
        self.macro   = MacroContextEngine()  # client injected after binance init
        self._last_macro_update  = None
        self._last_regime_update = None
        self._last_daily_summary = None

        # AI agent
        self.agent: HybridAIAgent = None
        self.market_cache: dict   = {}

        # Spot positions (key: symbol)
        self.spot_positions: dict = {}

        # Training tracker
        self.last_trained: dict = {}
        self.scan_count = 0
        self.claude_active = False

    # ── STARTUP ───────────────────────────────────────────

    def start(self) -> None:
        log.info("=" * 70)
        log.info("  HYBRID AI CRYPTO BOT v4.0")
        log.info("  ICT · S/R · Long/Short · Agent · LSTM · RF")
        log.info("=" * 70)

        self.client = Client(self.cfg.BINANCE_API_KEY, self.cfg.BINANCE_API_SECRET)
        try:
            self.client.ping()
            log.info("✅ Binance connection OK")
        except Exception as e:
            log.critical(f"❌ Binance connection failed: {e}")
            raise SystemExit(1)

        # Claude
        if self.cfg.CLAUDE_API_KEY and "YOUR_" not in self.cfg.CLAUDE_API_KEY:
            try:
                init_claude(self.cfg.CLAUDE_API_KEY)
                self.claude_active = True
                log.info("✅ Claude API OK")
            except Exception as e:
                log.warning(f"⚠️  Claude API failed: {e}")

        # Futures engine
        demo = (self.cfg.TRADING_MODE == 'demo')
        self.futures_eng = FuturesEngine(
            client=self.client,
            default_leverage=self.cfg.DEFAULT_LEVERAGE,
            demo_mode=demo,
        )

        # AI Agent
        if self.claude_active and self.cfg.AGENT_ENABLED:
            executor = AgentToolExecutor(
                binance_client = self.client,
                market_cache   = self.market_cache,
                trade_history  = self.risk.islem_gecmisi,
            )
            self.agent = HybridAIAgent(
                claude_api_key = self.cfg.CLAUDE_API_KEY,
                executor       = executor,
                on_thought     = self.stats.add_thought,
            )
            log.info("✅ AI Agent (ReAct) active")

        # ML models
        for sym in self.cfg.SYMBOLS:
            self.lstm_models[sym] = LSTMModel(sym)
            self.rf_models[sym]   = RFModel(sym)
            self.last_trained[sym] = None

        # Stats init
        usdt = self._get_balance("USDT")
        self.stats.start_capital  = usdt
        self.stats.update_portfolio(usdt)
        self.risk.mevcut_sermaye  = usdt

        log.info(f"💰 Starting capital: ${usdt:.2f} USDT")
        log.info(f"📊 Symbols: {', '.join(self.cfg.SYMBOLS)}")

        # Memory + Macro setup
        if self.cfg.MEMORY_ENABLED:
            self.macro = MacroContextEngine(binance_client=self.client)
            set_claude_memory(self.memory, self.macro)
            mem_stats = self.memory.get_stats()
            log.info(f"🧠 Memory loaded | All-time trades: {mem_stats['all_time']['total_trades']} | "                     f"Win rate: {mem_stats['all_time']['win_rate']:.1%}")
            # Initial macro fetch
            try:
                macro_data = self.macro.fetch_all()
                self.memory.update_macro({
                    'btc_dominance': macro_data.get('btc_dominance', 0),
                    'total_mcap':    macro_data.get('total_mcap_usd', 0),
                    'fear_greed':    macro_data.get('fear_greed', 50),
                    'fg_trend':      macro_data.get('fear_greed_trend', 'stable'),
                    'funding_rate':  macro_data.get('btc_funding_rate', 0),
                })
                log.info(f"📡 Macro: BTC dom {macro_data.get('btc_dominance',0):.1f}% | "                         f"F&G {macro_data.get('fear_greed',50)}/100 | "                         f"{macro_data.get('halving_cycle_phase','?')}")
            except Exception as e:
                log.warning(f"Macro initial fetch failed: {e}")
        log.info(f"⚡ Mode: {self.cfg.TRADING_MODE.upper()} | "
                 f"Leverage: ×{self.cfg.DEFAULT_LEVERAGE}")
        log.info(f"🧠 ICT Engine: {'ON' if self.cfg.ICT_ENABLED else 'OFF'} | "
                 f"Agent: {'ON' if self.agent else 'OFF'} | "
                 f"Claude: {'ON' if self.claude_active else 'OFF'}")
        log.info("-" * 70)

    # ── MAIN LOOP ─────────────────────────────────────────

    def run(self) -> None:
        self.start()

        while True:
            try:
                self.scan_count += 1
                log.info(f"\n{'═' * 70}")
                log.info(f"  SCAN #{self.scan_count} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Mode: {self.cfg.TRADING_MODE.upper()}")
                log.info(f"{'═' * 70}")

                usdt = self._get_balance("USDT")
                self.risk.gunluk_guncelle(usdt)
                self.stats.update_portfolio(usdt)

                if self.risk.gunluk_limit_asimi():
                    log.warning("🛑 Daily loss limit reached — skipping scan")
                    time.sleep(self.cfg.SCAN_INTERVAL_SECONDS)
                    continue

                # ── Periodic memory + macro update ──
                self._update_memory_periodic()

                for sym in self.cfg.SYMBOLS:
                    try:
                        self._process_symbol(sym)
                    except Exception as e:
                        log.error(f"❌ [{sym}] Error: {e}", exc_info=True)

                # Summary
                s = self.stats.compute('all', self.cfg.TRADING_MODE)
                log.info(
                    f"\n📊 PORTFOLIO | Trades:{s['total']} | "
                    f"WR:{s['win_rate']:.1f}% | "
                    f"PnL:${s['total_pnl']:.2f} ({s['pnl_pct']:+.2f}%) | "
                    f"Portfolio:${s['portfolio']:.2f}"
                )

            except KeyboardInterrupt:
                log.info("\n👋 Bot stopped by user.")
                break
            except Exception as e:
                log.error(f"❌ Main loop error: {e}", exc_info=True)

            time.sleep(self.cfg.SCAN_INTERVAL_SECONDS)

    # ── SYMBOL PROCESSING ─────────────────────────────────

    def _process_symbol(self, symbol: str) -> None:
        log.info(f"\n📈 [{symbol}] Analyzing...")

        # Fetch data
        packet      = fetch_market_data(self.client, symbol, self.cfg.INTERVAL, self.cfg.CRYPTOPANIC_TOKEN)
        df_raw      = packet['df']
        price       = packet['fiyat']
        ob_data     = packet['ob']
        fg_data     = packet['fg']
        news_data   = packet['haber']

        df_ind      = compute_indicators(df_raw)
        features    = feature_vector(df_ind)
        atr         = features['atr_14']

        self.market_cache[symbol] = {**features, 'fiyat': price}

        log.info(f"  💰 ${price:.4f} | RSI:{features['rsi']:.1f} | "
                 f"ATR:{atr:.4f} | F&G:{fg_data['score']}/100")

        # ── ICT ANALYSIS ──────────────────────────────────
        ict_signal = None
        sr_analysis = None

        if self.cfg.ICT_ENABLED:
            ict_signal  = self.ict_engine.analyze(df_ind, price)
            sr_analysis = self.sr_engine.analyze(df_ind, price)
            log.info(
                f"  🎯 ICT: {ict_signal.action} ({ict_signal.confidence:.0%}) | "
                f"Trend: {ict_signal.reasons[0] if ict_signal.reasons else 'N/A'}"
            )
            log.info(
                f"  📐 S/R: Support:${sr_analysis.nearest_support:.4f} | "
                f"Resistance:${sr_analysis.nearest_resistance:.4f} | "
                f"Near res: {sr_analysis.near_resistance}"
            )
            # Log ICT reasons
            for reason in ict_signal.reasons[:3]:
                log.info(f"      • {reason}")

        # ── FUTURES POSITION MONITORING ────────────────────
        if self.cfg.TRADING_MODE in ('futures', 'demo'):
            active_pos = self.futures_eng.positions.get(symbol)
            if active_pos:
                partial_exits = sr_analysis.partial_exit_levels if sr_analysis else []
                opposite = self._is_opposite_signal(active_pos.side, ict_signal, features)
                action = self.futures_eng.monitor_position(
                    symbol, price, partial_exits, atr, opposite_signal=opposite
                )
                if action:
                    log.info(f"  ⚡ Futures action: {action}")
                    if action.startswith('REVERSE_TO_'):
                        new_side = action.replace('REVERSE_TO_', '')
                        self._open_futures_position(symbol, new_side, price, atr, ict_signal)
                return

        # ── SPOT POSITION S/R MONITORING ──────────────────
        elif symbol in self.spot_positions and sr_analysis:
            self._monitor_spot_position(symbol, price, atr, sr_analysis)
            return

        # ── MAX POSITIONS CHECK ────────────────────────────
        open_futures = len(self.futures_eng.positions) if self.futures_eng else 0
        open_spot    = len(self.spot_positions)
        if (open_futures + open_spot) >= self.cfg.MAX_OPEN_POSITIONS:
            log.info(f"  ℹ️  [{symbol}] Max positions reached")
            return

        if not self.risk.korelasyon_filtre(self.spot_positions, symbol):
            return

        # ── MODEL TRAINING ────────────────────────────────
        self._check_training(symbol, df_ind)

        # ── ML SIGNALS ────────────────────────────────────
        lstm_result = self.lstm_models[symbol].tahmin(df_ind)
        rf_result   = self.rf_models[symbol].tahmin(df_ind)
        log.info(f"  🧠 LSTM:{lstm_result['yon']}({lstm_result['guven']:.2f}) | "
                 f"RF:{rf_result['sinyal']}({rf_result['guven']:.2f})")

        # ── AI AGENT ──────────────────────────────────────
        agent_decision = None
        if self.agent:
            trigger, reason = self.agent.tetikle_mi(features, lstm_result, rf_result)
            if trigger:
                log.info(f"  🤖 Agent triggered: {reason}")
                try:
                    ag = self.agent.analiz_et(
                        symbol, reason,
                        {'lstm': lstm_result['yon'], 'rf': rf_result['sinyal'], 'price': price}
                    )
                    agent_decision = ag
                    log.info(f"  🤖 Agent: {ag['action']} ({ag['confidence']:.0%}) | {ag['reasoning'][:80]}")
                except Exception as e:
                    log.error(f"  ❌ Agent error: {e}")

        # ── CLAUDE / FALLBACK ──────────────────────────────
        if self.claude_active and not agent_decision:
            claude_dec = get_claude_decision(
                symbol, price, features, lstm_result, rf_result,
                ob_data, fg_data, news_data
            )
        elif not agent_decision:
            sig = classic_signal(df_ind)
            claude_dec = {
                'karar': sig, 'guven': 0.65 if sig != 'BEKLE' else 0.3,
                'risk_skoru': 0.3, 'gerekceler': f'Fallback: {sig}'
            }
        else:
            risk_map = {'low': 0.2, 'medium': 0.4, 'high': 0.7, 'extreme': 0.9}
            claude_dec = {
                'karar': agent_decision['action'],
                'guven': agent_decision['confidence'],
                'risk_skoru': risk_map.get(agent_decision.get('risk_level','medium'), 0.4),
                'gerekceler': agent_decision['reasoning'],
            }

        # ── VOTING (LSTM + RF + Claude + ICT) ─────────────
        vote_result = voting_system(lstm_result['yon'], rf_result['sinyal'], claude_dec['karar'])
        final_action = vote_result['final_karar']

        # ICT confirmation layer
        ict_vote = 0
        if ict_signal and ict_signal.confidence >= self.cfg.ICT_MIN_CONFIDENCE:
            if ict_signal.action == 'LONG' and final_action == 'AL':
                ict_vote = 1
                log.info(f"  ✅ ICT confirms LONG")
            elif ict_signal.action == 'SHORT' and final_action == 'SAT':
                ict_vote = 1
                log.info(f"  ✅ ICT confirms SHORT")
            elif ict_signal.action == 'WAIT':
                log.info(f"  ⚠️  ICT says WAIT — reducing confidence")
                claude_dec['guven'] *= 0.7

        log.info(
            f"  🗳️  Votes: {vote_result['bireysel']} | ICT:{'✓' if ict_vote else '✗'} | "
            f"Final:{final_action} ({vote_result['mutabakat_pct']:.0%})"
        )

        # ── EXECUTE TRADE ──────────────────────────────────
        threshold = self.cfg.VOTING_THRESHOLD / 3
        if final_action == 'AL' and vote_result['mutabakat_pct'] >= threshold:
            if self.cfg.TRADING_MODE in ('futures', 'demo'):
                self._open_futures_position(symbol, 'LONG', price, atr, ict_signal)
            else:
                self._open_spot_position(symbol, price, atr, claude_dec, sr_analysis)

        elif final_action == 'SAT':
            if self.cfg.TRADING_MODE in ('futures', 'demo'):
                self._open_futures_position(symbol, 'SHORT', price, atr, ict_signal)
            else:
                log.info(f"  📉 [{symbol}] SELL signal — spot mode, no short")
        else:
            log.info(f"  ⏳ [{symbol}] WAIT — conditions not met")

    # ── FUTURES POSITION ──────────────────────────────────

    def _open_futures_position(self, symbol: str, side: str,
                               price: float, atr: float,
                               ict_signal=None) -> None:
        usdt   = self._get_balance("USDT")
        amount = self.risk.pozisyon_boyutu(
            self.cfg.MIN_SIGNAL_CONFIDENCE, 0.4, usdt
        )
        if amount < 5:
            log.warning(f"  ⚠️  Amount too low: ${amount:.2f}")
            return

        # Use ICT SL/TP if available, else ATR-based
        if ict_signal and ict_signal.sl_price:
            sl  = ict_signal.sl_price
            tp1 = ict_signal.tp1_price
            tp2 = ict_signal.tp2_price
            tp3 = ict_signal.tp3_price or price * (1.06 if side=='LONG' else 0.94)
        else:
            sl_tp = self.risk.dinamik_sl_tp(price, atr, yon='long' if side=='LONG' else 'short')
            sl  = sl_tp['sl']
            tp1 = price + atr * 2   if side == 'LONG' else price - atr * 2
            tp2 = price + atr * 3.5 if side == 'LONG' else price - atr * 3.5
            tp3 = sl_tp['tp']

        if side == 'LONG':
            result = self.futures_eng.open_long(symbol, amount, sl, tp1, tp2, tp3)
        else:
            result = self.futures_eng.open_short(symbol, amount, sl, tp1, tp2, tp3)

        if result.success:
            self.stats.add_trade({
                'symbol': symbol, 'side': side, 'price': price,
                'pnl': 0, 'mode': self.cfg.TRADING_MODE,
            })
            log.info(f"  ✅ FUTURES {side} opened | ${price:.4f} | "
                     f"SL:${sl:.4f} TP1:${tp1:.4f} TP3:${tp3:.4f}")
        else:
            log.error(f"  ❌ Futures open failed: {result.reason}")

    # ── SPOT POSITION ─────────────────────────────────────

    def _open_spot_position(self, symbol: str, price: float,
                            atr: float, claude_dec: dict,
                            sr_analysis=None) -> None:
        usdt   = self._get_balance("USDT")
        amount = self.risk.pozisyon_boyutu(claude_dec['guven'], claude_dec['risk_skoru'], usdt)
        if amount < 5:
            log.warning(f"  ⚠️  Amount too low: ${amount:.2f}")
            return

        sl_tp = self.risk.dinamik_sl_tp(price, atr)
        partial_exits = sr_analysis.partial_exit_levels if sr_analysis else []

        try:
            order = self.client.order_market_buy(
                symbol=symbol, quoteOrderQty=round(amount, 2)
            )
            actual_price = float(order.get('fills', [{}])[0].get('price', price)) if order.get('fills') else price
            actual_qty   = float(order['executedQty'])

            self.spot_positions[symbol] = {
                'entry_price':    actual_price,
                'quantity':       actual_qty,
                'remaining_qty':  actual_qty,
                'sl':             sl_tp['sl'],
                'tp':             sl_tp['tp'],
                'partial_exits':  partial_exits,
                'done_exits':     [],
                'opened_at':      datetime.now().isoformat(),
                'mode':           'spot',
            }
            log.info(f"  ✅ SPOT opened | ${actual_price:.4f} × {actual_qty:.6f} | "
                     f"SL:${sl_tp['sl']:.4f} TP:${sl_tp['tp']:.4f}")
        except BinanceAPIException as e:
            log.error(f"  ❌ Spot open error: {e}")

    def _monitor_spot_position(self, symbol: str, price: float,
                               atr: float, sr_analysis) -> None:
        pos = self.spot_positions.get(symbol)
        if not pos: return

        # SL
        if price <= pos['sl']:
            self._close_spot(symbol, price, 'SL_HIT')
            return

        # Partial exits at resistance
        for level in pos.get('partial_exits', []):
            lp = level['price']
            if lp in pos['done_exits']: continue
            if price >= lp:
                exit_qty = round(pos['remaining_qty'] * level['exit_pct'], 6)
                if exit_qty > 0:
                    try:
                        self.client.order_market_sell(symbol=symbol, quantity=exit_qty)
                        pos['remaining_qty'] -= exit_qty
                        pos['done_exits'].append(lp)
                        # Move SL to breakeven after first partial exit
                        if not pos['done_exits']:
                            pos['sl'] = pos['entry_price']
                        pnl = (price - pos['entry_price']) * exit_qty
                        self.stats.add_trade({'symbol': symbol, 'pnl': round(pnl, 4), 'mode': 'spot'})
                        log.info(f"  📤 Partial exit {symbol} | {level['exit_pct']:.0%} at ${price:.4f} | pnl:${pnl:.4f}")
                    except BinanceAPIException as e:
                        log.error(f"  ❌ Partial exit error: {e}")

        # Full TP
        if price >= pos['tp']:
            self._close_spot(symbol, price, 'TP_HIT')

    def _close_spot(self, symbol: str, price: float, reason: str) -> None:
        pos = self.spot_positions.get(symbol)
        if not pos: return
        try:
            self.client.order_market_sell(symbol=symbol, quantity=round(pos['remaining_qty'], 6))
            pnl = (price - pos['entry_price']) * pos['remaining_qty']
            self.stats.add_trade({'symbol': symbol, 'pnl': round(pnl, 4), 'mode': 'spot'})
            record = self.risk.islem_kaydet(symbol, 'long', pos['entry_price'], price, pos['remaining_qty'])
            log.info(f"  ✅ SPOT closed | {symbol} {reason} | pnl:${record['pnl']:.4f}")
            # Record in long-term memory
            if self.cfg.MEMORY_ENABLED:
                try:
                    self.memory.record_trade(symbol=symbol, pnl=record['pnl'],
                                             strategy='hybrid', side='long')
                except Exception: pass
            del self.spot_positions[symbol]
        except BinanceAPIException as e:
            log.error(f"  ❌ Spot close error: {e}")

    # ── HELPERS ───────────────────────────────────────────

    def _is_opposite_signal(self, current_side: str, ict_signal, features: dict) -> bool:
        rsi = features.get('rsi', 50)
        if current_side == 'LONG':
            ict_short = ict_signal and ict_signal.action == 'SHORT' and ict_signal.confidence > 0.6
            rsi_short = rsi > Config.RSI_OVERBOUGHT + 5
            return ict_short or rsi_short
        elif current_side == 'SHORT':
            ict_long = ict_signal and ict_signal.action == 'LONG' and ict_signal.confidence > 0.6
            rsi_long = rsi < Config.RSI_OVERSOLD - 5
            return ict_long or rsi_long
        return False

    def _check_training(self, symbol: str, df_ind) -> None:
        last  = self.last_trained.get(symbol)
        now   = datetime.now()
        if last is None or (now - last).total_seconds() / 3600 >= self.cfg.RETRAIN_EVERY_HOURS:
            log.info(f"  🎓 [{symbol}] Training models...")
            lr = self.lstm_models[symbol].egit(df_ind)
            rr = self.rf_models[symbol].egit(df_ind)
            log.info(f"  🎓 LSTM:{lr.get('durum')} | RF:{rr.get('durum')} acc:{rr.get('rf_acc','?')}")
            self.last_trained[symbol] = now

    def _update_memory_periodic(self) -> None:
        """Run periodic memory + macro updates."""
        if not self.cfg.MEMORY_ENABLED:
            return

        now = datetime.now()

        # Macro update every 15 minutes
        if (self._last_macro_update is None or
                (now - self._last_macro_update).seconds >= self.cfg.MACRO_UPDATE_MINUTES * 60):
            try:
                macro_data = self.macro.fetch_all()
                self.memory.update_macro({
                    'btc_dominance': macro_data.get('btc_dominance', 0),
                    'total_mcap':    macro_data.get('total_mcap_usd', 0),
                    'fear_greed':    macro_data.get('fear_greed', 50),
                    'fg_trend':      macro_data.get('fear_greed_trend', 'stable'),
                    'funding_rate':  macro_data.get('btc_funding_rate', 0),
                })
                self._last_macro_update = now
                log.info(f"[Memory] Macro updated | F&G:{macro_data.get('fear_greed',50)} | "                         f"BTCdom:{macro_data.get('btc_dominance',0):.1f}%")
            except Exception as e:
                log.warning(f"[Memory] Macro update failed: {e}")

        # Regime analysis every 6 hours
        if (self._last_regime_update is None or
                (now - self._last_regime_update).seconds >= self.cfg.REGIME_UPDATE_HOURS * 3600):
            try:
                cache = self.market_cache
                if cache:
                    piyasa = {sym: {k: v for k, v in data.items()
                                    if k in ['rsi', 'ema_trend', 'volatility', 'macd']}
                              for sym, data in cache.items()}
                    regime = analyze_regime(piyasa)
                    self.memory.update_regime(regime)
                    self._last_regime_update = now
                    log.info(f"[Memory] Regime updated: {regime}")
            except Exception as e:
                log.warning(f"[Memory] Regime update failed: {e}")

        # Daily summary at 23:xx
        if (now.hour == self.cfg.DAILY_SUMMARY_HOUR and
                (self._last_daily_summary is None or
                 self._last_daily_summary.date() < now.date())):
            try:
                macro = self.macro.fetch_all()
                btc_chg = macro.get('btc_change_24h', 0)
                eth_chg = 0  # Would need separate fetch
                regime  = self.memory.memory['market_regime']['current']
                stats   = self.memory.get_stats()
                notes   = (f"Trades today: {stats['all_time']['total_trades']} | "                           f"Regime: {regime}")
                self.memory.add_daily_summary(btc_chg, eth_chg, regime, notes)
                self._last_daily_summary = now
                log.info("[Memory] Daily summary stored")
            except Exception as e:
                log.warning(f"[Memory] Daily summary failed: {e}")

    def _get_balance(self, asset: str) -> float:
        try:
            b = self.client.get_asset_balance(asset=asset)
            return float(b['free']) if b else 0.0
        except Exception as e:
            log.warning(f"Balance error ({asset}): {e}")
            return 0.0


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    HybridAIBot().run()
