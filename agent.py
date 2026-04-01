"""
ai_engine/agent.py  ─ Hybrid AI Agent v2.0
ReAct (Reasoning + Acting) döngüsü
5 Yetenek: Araştırma · Strateji Seçimi · Planlama · Öz-düzeltme · Portföy Dengesi
"""

import json, logging, time, requests
from datetime import datetime
from typing import Callable, Optional
import anthropic

log = logging.getLogger("CryptoBot.Agent")

# ─────────────────────────────────────────────────────────────
# ARAÇ ŞEMASI
# ─────────────────────────────────────────────────────────────

AGENT_TOOLS = [
    {"name":"get_market_data","description":"Anlık fiyat, RSI, EMA, MACD, ATR, volatilite, hacim verisi döner.",
     "input_schema":{"type":"object","properties":{"symbol":{"type":"string"},"fields":{"type":"array","items":{"type":"string"}}},"required":["symbol"]}},
    {"name":"get_news_sentiment","description":"Kripto para için son haberler ve sentiment skoru (-1 ile +1 arası).",
     "input_schema":{"type":"object","properties":{"symbol":{"type":"string"},"limit":{"type":"integer","default":10}},"required":["symbol"]}},
    {"name":"get_fear_greed","description":"Fear & Greed Index (0=Extreme Fear, 100=Extreme Greed) + trend.",
     "input_schema":{"type":"object","properties":{"days":{"type":"integer","default":3}}}},
    {"name":"get_order_book","description":"Order book derinliği: bid/ask oranı, alım baskısı, spread.",
     "input_schema":{"type":"object","properties":{"symbol":{"type":"string"},"depth":{"type":"integer","default":20}},"required":["symbol"]}},
    {"name":"get_macro_data","description":"BTC dominance, toplam piyasa değeri, funding rate, 24h değişim.",
     "input_schema":{"type":"object","properties":{"metrics":{"type":"array","items":{"type":"string"}}}}},
    {"name":"analyze_trade_history","description":"Son N işlemin performansı: kazanma oranı, avg K/Z, profit factor, kötü stratejiler.",
     "input_schema":{"type":"object","properties":{"limit":{"type":"integer","default":20},"symbol":{"type":"string","default":""}}}},
    {"name":"get_portfolio_balance","description":"Portföy dağılımı, toplam USDT değeri, her coin ağırlığı.",
     "input_schema":{"type":"object","properties":{}}},
    {"name":"check_market_regime","description":"Piyasa rejimi: strong_uptrend, downtrend, range, volatile, accumulation + en iyi strateji önerisi.",
     "input_schema":{"type":"object","properties":{"symbol":{"type":"string"},"timeframe":{"type":"string","default":"1h"}},"required":["symbol"]}},
    {"name":"set_decision","description":"Final karar kaydı. Tüm analiz bittikten sonra çağrılır.",
     "input_schema":{"type":"object","properties":{
         "action":{"type":"string","enum":["AL","SAT","BEKLE","AZALT","ARTIR"]},
         "symbol":{"type":"string"},"confidence":{"type":"number"},
         "risk_level":{"type":"string","enum":["low","medium","high","extreme"]},
         "reasoning":{"type":"string"},"position_size_pct":{"type":"number"},
         "suggested_sl_pct":{"type":"number"},"suggested_tp_pct":{"type":"number"},
         "next_check_seconds":{"type":"integer"}},"required":["action","symbol","confidence","reasoning"]}},
]

# ─────────────────────────────────────────────────────────────
# ARAÇ UYGULAYICI
# ─────────────────────────────────────────────────────────────

class AgentToolExecutor:
    def __init__(self, binance_client=None, market_cache: dict=None, trade_history: list=None):
        self.client        = binance_client
        self.market_cache  = market_cache  or {}
        self.trade_history = trade_history or []

    def execute(self, name: str, inp: dict) -> dict:
        try:
            return getattr(self, f"_t_{name}", lambda **k: {"error":"not found"})(**inp)
        except Exception as e:
            log.error(f"[Tool] {name}: {e}")
            return {"error": str(e)}

    def _t_get_market_data(self, symbol: str, fields: list=None) -> dict:
        c = self.market_cache.get(symbol, {})
        r = {"symbol":symbol,"price":c.get("fiyat",0),"volume_ratio":c.get("volume_ratio",1),
             "rsi":c.get("rsi",50),"ema9":c.get("ema_9",0),"ema21":c.get("ema_21",0),
             "ema50":c.get("ema_50",0),"ema200":c.get("ema_200",0),
             "macd":c.get("macd",0),"macd_hist":c.get("macd_hist",0),
             "atr":c.get("atr_14",0),"volatility":c.get("volatility",0),
             "bb_pos":c.get("bb_pos",0.5),"stoch_k":c.get("stoch_k",50),
             "momentum":c.get("momentum",0),"vwap":c.get("vwap",0),
             "timestamp":datetime.now().isoformat()}
        if fields:
            r = {k:v for k,v in r.items() if k in fields+["symbol","timestamp"]}
        return r

    def _t_get_news_sentiment(self, symbol: str, limit: int=10) -> dict:
        try:
            coin = symbol.replace("USDT","")
            r = requests.get(f"https://cryptopanic.com/api/free/v1/posts/?currencies={coin}&kind=news",timeout=6)
            if r.status_code == 200:
                data = r.json().get("results",[])[:limit]
                pos = sum(1 for h in data if h.get("votes",{}).get("positive",0)>h.get("votes",{}).get("negative",0))
                neg = len(data)-pos; total = len(data) or 1
                score = (pos-neg)/total
                return {"coin":coin,"sentiment_score":round(score,3),
                        "positive":pos,"negative":neg,"total":total,
                        "label":"positive" if score>0.2 else "negative" if score<-0.2 else "neutral",
                        "top_headlines":[h.get("title","")[:80] for h in data[:3]]}
        except Exception: pass
        return {"coin":symbol,"sentiment_score":0.0,"label":"neutral","positive":0,"negative":0,"total":0,"top_headlines":[]}

    def _t_get_fear_greed(self, days: int=3) -> dict:
        try:
            r = requests.get(f"https://api.alternative.me/fng/?limit={min(days,30)}",timeout=5)
            r.raise_for_status()
            data = r.json().get("data",[])
            if data:
                latest = data[0]; score = int(latest["value"])
                trend = score - int(data[-1]["value"]) if len(data)>1 else 0
                return {"score":score,"label":latest["value_classification"],"trend":trend,
                        "history":[{"score":int(d["value"]),"label":d["value_classification"]} for d in data]}
        except Exception: pass
        return {"score":50,"label":"Neutral","trend":0,"history":[]}

    def _t_get_order_book(self, symbol: str, depth: int=20) -> dict:
        if self.client:
            try:
                ob = self.client.get_order_book(symbol=symbol,limit=depth)
                bids = sum(float(b[1]) for b in ob["bids"])
                asks = sum(float(a[1]) for a in ob["asks"])
                total = bids+asks; ratio = bids/asks if asks>0 else 1.0
                bb = float(ob["bids"][0][0]); ba = float(ob["asks"][0][0])
                spread = (ba-bb)/bb*100
                return {"bid_volume":round(bids,4),"ask_volume":round(asks,4),
                        "bid_ask_ratio":round(ratio,4),
                        "buy_pressure":round(bids/total*100,2) if total>0 else 50.0,
                        "spread_pct":round(spread,5),"best_bid":bb,"best_ask":ba}
            except Exception as e: log.warning(f"OB hatası: {e}")
        return {"bid_ask_ratio":1.0,"buy_pressure":50.0,"spread_pct":0.0}

    def _t_get_macro_data(self, metrics: list=None) -> dict:
        result = {}
        try:
            r = requests.get("https://api.coingecko.com/api/v3/global",timeout=8,
                             headers={"User-Agent":"CryptoBot/2.0"})
            if r.status_code==200:
                d = r.json().get("data",{})
                result["btc_dominance"]    = round(d.get("market_cap_percentage",{}).get("btc",0),2)
                result["total_mcap_usd"]   = d.get("total_market_cap",{}).get("usd",0)
                result["total_volume_usd"] = d.get("total_volume",{}).get("usd",0)
                result["mcap_change_24h"]  = round(d.get("market_cap_change_percentage_24h_usd",0),2)
        except Exception: result["btc_dominance"]=52.0
        if self.client and (not metrics or "funding_rate" in (metrics or [])):
            try:
                fr = self.client.futures_funding_rate(symbol="BTCUSDT",limit=1)
                result["btc_funding_rate"] = float(fr[0]["fundingRate"])*100 if fr else 0.0
            except Exception: result["btc_funding_rate"]=0.0
        return result

    def _t_analyze_trade_history(self, limit: int=20, symbol: str="") -> dict:
        trades = [t for t in self.trade_history if not symbol or t.get("symbol")==symbol][-limit:]
        if not trades: return {"message":"Henüz işlem yok","total":0}
        wins   = [t for t in trades if t.get("pnl",0)>0]
        losses = [t for t in trades if t.get("pnl",0)<=0]
        gp = sum(t.get("pnl",0) for t in wins)
        gl = abs(sum(t.get("pnl",0) for t in losses))
        return {"total_trades":len(trades),"wins":len(wins),"losses":len(losses),
                "win_rate":round(len(wins)/len(trades),4) if trades else 0,
                "total_pnl":round(gp-gl,4),"avg_win":round(gp/len(wins),4) if wins else 0,
                "avg_loss":round(gl/len(losses),4) if losses else 0,
                "profit_factor":round(gp/gl,3) if gl>0 else 999,
                "best_trade":max((t.get("pnl",0) for t in trades),default=0),
                "worst_trade":min((t.get("pnl",0) for t in trades),default=0),
                "last_5":[{"symbol":t.get("symbol"),"pnl":t.get("pnl")} for t in trades[-5:]]}

    def _t_get_portfolio_balance(self) -> dict:
        if not self.client: return {"error":"Binance bağlı değil"}
        try:
            acc = self.client.get_account()
            bals = [b for b in acc["balances"] if float(b["free"])+float(b["locked"])>0]
            portfolio=[]; total=0.0
            for b in bals:
                asset=b["asset"]; amt=float(b["free"])+float(b["locked"])
                if asset=="USDT": uv=amt
                else:
                    try: uv=amt*float(self.client.get_symbol_ticker(symbol=f"{asset}USDT")["price"])
                    except: uv=0
                total+=uv; portfolio.append({"asset":asset,"amount":round(amt,6),"usdt_value":round(uv,2)})
            for p in portfolio: p["weight_pct"]=round(p["usdt_value"]/total*100,2) if total>0 else 0
            return {"portfolio":portfolio,"total_usdt":round(total,2),"asset_count":len(portfolio)}
        except Exception as e: return {"error":str(e)}

    def _t_check_market_regime(self, symbol: str, timeframe: str="1h") -> dict:
        c = self.market_cache.get(symbol,{})
        e9=c.get("ema_9",0); e21=c.get("ema_21",0); e50=c.get("ema_50",0); e200=c.get("ema_200",0)
        vol=c.get("volatility",0); bb_w=c.get("bb_width",0); price=c.get("fiyat",1)
        atr=c.get("atr_14",0)
        if e9>e21>e50>e200: regime="strong_uptrend"
        elif e9<e21<e50<e200: regime="strong_downtrend"
        elif abs(e9-e21)/(e21 or 1)<0.005 and bb_w<0.04: regime="accumulation"
        elif vol>3.0: regime="volatile"
        elif e9>e21: regime="weak_uptrend"
        elif e9<e21: regime="weak_downtrend"
        else: regime="range"
        sm={"strong_uptrend":"EMA Crossover + Momentum","strong_downtrend":"RSI Divergence + Bollinger",
            "accumulation":"Bollinger Squeeze + OBV","volatile":"ATR Risk Mgmt + SuperTrend",
            "weak_uptrend":"MACD + RSI","weak_downtrend":"Williams %R + Stochastic","range":"Bollinger + CCI"}
        return {"symbol":symbol,"regime":regime,"timeframe":timeframe,
                "best_strategies":sm.get(regime,"AI Hybrid"),
                "ema_alignment":"bullish" if e9>e50 else "bearish",
                "volatility_pct":round(vol,3),"atr_pct":round(atr/price*100 if price else 0,3)}

    def _t_set_decision(self, action:str, symbol:str, confidence:float, reasoning:str,
                        risk_level:str="medium", position_size_pct:float=10.0,
                        suggested_sl_pct:float=None, suggested_tp_pct:float=None,
                        next_check_seconds:int=60) -> dict:
        return {"recorded":True,"action":action,"symbol":symbol,"confidence":confidence,
                "risk_level":risk_level,"reasoning":reasoning,
                "position_size_pct":position_size_pct,
                "suggested_sl_pct":suggested_sl_pct,"suggested_tp_pct":suggested_tp_pct,
                "next_check_seconds":next_check_seconds,"timestamp":datetime.now().isoformat()}


# ─────────────────────────────────────────────────────────────
# AGENT
# ─────────────────────────────────────────────────────────────

class HybridAIAgent:
    SYSTEM_PROMPT = """Sen profesyonel bir Hybrid AI Trading Agent'sın. Kripto piyasalarında otonom çalışıyorsun.

ANALİZ AKIŞI (sırayı takip et):
1. get_market_data → fiyat ve teknik göstergeler
2. check_market_regime → piyasa rejimi ve en iyi strateji
3. get_news_sentiment → haber sentiment
4. get_fear_greed → piyasa psikolojisi
5. get_order_book → alım/satım baskısı
6. get_macro_data → BTC dominance, mcap
7. analyze_trade_history → geçmiş performans (strateji düzelt)
8. get_portfolio_balance → portföy dengesi
9. set_decision → KARAR (tüm analiz bittikten sonra)

KARAR MATRİSİ:
- Tüm göstergeler uyumlu → AL/SAT (yüksek güven)
- Çelişen sinyaller → BEKLE
- Extreme Fear + RSI<30 + OB alım baskısı → güçlü AL
- Extreme Greed + RSI>70 + negatif haber → güçlü SAT
- Yüksek volatilite + belirsiz → BEKLE + düşük pozisyon
- Portföy tek coinde yoğunlaşmış → AZALT veya ARTIR (dengeleme)

Risk seviyesi: low(<0.3), medium(0.3-0.6), high(0.6-0.8), extreme(>0.8)
Güven <0.55 ise mutlaka BEKLE döndür.
Tüm gerekçeleri Türkçe yaz."""

    MAX_ITER = 12

    def __init__(self, claude_api_key: str, executor: AgentToolExecutor,
                 on_thought: Callable=None):
        self.anthropic  = anthropic.Anthropic(api_key=claude_api_key)
        self.executor   = executor
        self.on_thought = on_thought
        self.gecmis: list = []

    def analiz_et(self, symbol: str, tetikleyen: str="tarama",
                  onceki_sinyal: dict=None) -> dict:
        t0 = time.time()
        log.info(f"[Agent] {symbol} | {tetikleyen}")
        self._bildir("🧠", f"{symbol} agent analizi başladı [{tetikleyen}]")

        ilk = (f"Sembol: {symbol}\nTetikleyen: {tetikleyen}\n"
               f"Zaman: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if onceki_sinyal:
            ilk += f"Önceki: {json.dumps(onceki_sinyal,ensure_ascii=False)}\n"
        ilk += "\nKapsamlı analiz yap ve set_decision ile kararını kaydet."

        msgs = [{"role":"user","content":ilk}]
        zincir=[]; arac_n=0; final=None

        for it in range(self.MAX_ITER):
            try:
                resp = self.anthropic.messages.create(
                    model="claude-sonnet-4-6", max_tokens=4096,
                    system=self.SYSTEM_PROMPT, tools=AGENT_TOOLS, messages=msgs)
            except anthropic.RateLimitError:
                time.sleep(10); continue
            except Exception as e:
                log.error(f"[Agent] API hata: {e}"); break

            tool_results=[]
            for blok in resp.content:
                if blok.type=="text" and blok.text.strip():
                    zincir.append({"tip":"dusunce","metin":blok.text.strip(),"zaman":datetime.now().isoformat()})
                    self._bildir("💭", blok.text.strip()[:120])
                elif blok.type=="tool_use":
                    arac_n+=1
                    self._bildir("🔧", f"{blok.name}({json.dumps(blok.input)[:50]})")
                    sonuc = self.executor.execute(blok.name, blok.input)
                    zincir.append({"tip":"arac","arac":blok.name,"girdi":blok.input,
                                   "sonuc":sonuc,"zaman":datetime.now().isoformat()})
                    if blok.name=="set_decision" and sonuc.get("recorded"):
                        final=sonuc
                        self._bildir("✅",f"KARAR:{sonuc['action']} güven:{sonuc['confidence']:.0%} | {sonuc['reasoning'][:70]}")
                    tool_results.append({"type":"tool_result","tool_use_id":blok.id,
                                         "content":json.dumps(sonuc,ensure_ascii=False)})

            msgs.append({"role":"assistant","content":resp.content})
            if tool_results: msgs.append({"role":"user","content":tool_results})
            if resp.stop_reason=="end_turn" or final: break

        if not final:
            final={"action":"BEKLE","symbol":symbol,"confidence":0.3,"risk_level":"high",
                   "reasoning":"Analiz tamamlanamadı — ihtiyatlı bekle",
                   "position_size_pct":0,"suggested_sl_pct":3.0,"suggested_tp_pct":6.0,
                   "next_check_seconds":60}

        result={**final,"dusunce_zinciri":zincir,"arac_cagrilari":arac_n,
                "iterasyon":it+1,"sure_sn":round(time.time()-t0,2),"tetikleyen":tetikleyen}
        self.gecmis.append(result)
        log.info(f"[Agent] ✓ {symbol} | {final['action']} {final['confidence']:.0%} | {arac_n} araç | {result['sure_sn']}s")
        return result

    def tetikle_mi(self, ozellik: dict, lstm: dict, rf: dict) -> tuple:
        rsi=ozellik.get("rsi",50); vol=ozellik.get("volume_ratio",1)
        lg=lstm.get("guven",0); rg=rf.get("guven",0); rs=rf.get("sinyal","BEKLE")
        nedenler=[]
        if rsi<28: nedenler.append(f"RSI aşırı satılmış ({rsi:.1f})")
        if rsi>72: nedenler.append(f"RSI aşırı alınmış ({rsi:.1f})")
        if lg>0.70: nedenler.append(f"LSTM güçlü ({lg:.0%})")
        if rg>0.68 and rs!="BEKLE": nedenler.append(f"RF:{rs} ({rg:.0%})")
        if vol>2.0: nedenler.append(f"Yüksek hacim ({vol:.1f}x)")
        return (True, " | ".join(nedenler)) if nedenler else (False, "")

    def _bildir(self, icon:str, msg:str):
        log.info(f"[Agent] {icon} {msg}")
        if self.on_thought:
            try: self.on_thought(icon, msg)
            except Exception: pass
