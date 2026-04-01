# FinancialPro v6.0
### AI Hybrid Crypto Trading Platform

> Multi-agent • Multi-exchange • ICT • LSTM • Real-time dashboard

---

## Features

| Module | Description |
|---|---|
| **Multi-Exchange** | Binance, Bybit, Bitget, OKX, MEXC, Gate.io |
| **AI/ML Engine** | TensorFlow BiLSTM + Random Forest + XGBoost + Claude |
| **ICT Analysis** | Order Blocks, FVG, Liquidity Levels, Market Structure, OTE |
| **Multi-Agent** | Market · ML · News · Portfolio · Execution agents |
| **XAI** | Explainable AI — "Why did bot BUY?" |
| **Memory** | Long-term learning + macro context |
| **Demo System** | Real prices, simulated funds (up to $100k) |
| **WebSocket** | Live dashboard ↔ bot bridge |
| **Bilingual** | Turkish / English full UI |

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/yourname/cryptobot-pro
cd cryptobot-pro

# 2. Install
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env with your API keys

# 4. Run WebSocket server (terminal 1)
python websocket_server.py

# 5. Run bot (terminal 2)
python crypto_bot.py

# 6. Open dashboard
open dashboard.html  # or drag into browser
```

---

## Configuration (.env)

```env
# Exchange (choose one)
EXCHANGE=binance          # binance | bybit | bitget | okx

# Binance
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret

# Bybit (if using Bybit)
BYBIT_API_KEY=your_key
BYBIT_API_SECRET=your_secret

# AI
ANTHROPIC_API_KEY=sk-ant-...

# News (optional — free token from cryptopanic.com)
CRYPTOPANIC_TOKEN=your_token

# Bot settings
TRADING_MODE=spot         # spot | futures | demo
DEFAULT_LEVERAGE=10
INITIAL_CAPITAL=1000

# WebSocket (for server deployment)
WS_HOST=0.0.0.0
WS_PORT=8765
```

---

## News Sources

| Source | Type | Cost | API Required |
|---|---|---|---|
| **CryptoPanic** | Crypto news aggregator | Free tier | Yes (cryptopanic.com) |
| **Alternative.me** | Fear & Greed Index | Free | No |
| **CoinGecko** | Market data + trending | Free | No |
| **RSS Feeds** | Major crypto media | Free | No |

### CryptoPanic Setup
1. Go to [cryptopanic.com](https://cryptopanic.com)
2. Create free account
3. Get API token from profile
4. Add to `.env`: `CRYPTOPANIC_TOKEN=your_token`

---

## Indicator Data Sources

| Indicator | Source | Notes |
|---|---|---|
| **OHLCV** | Exchange API (Binance/Bybit/etc) | Real-time, no extra API |
| **RSI, MACD, EMA** | Calculated locally from OHLCV | No external API needed |
| **Bollinger Bands** | Calculated locally | No external API needed |
| **ATR, Stochastic** | Calculated locally | No external API needed |
| **ICT (OB/FVG)** | Calculated locally from OHLCV | Proprietary algorithm |
| **Order Book** | Exchange API | Real-time L2 data |
| **Funding Rate** | Exchange API (futures) | Real-time |
| **Fear & Greed** | alternative.me | Free, no API key |
| **BTC Dominance** | CoinGecko | Free, no API key |

> All indicators are calculated locally from exchange OHLCV data.
> No TradingView API or paid indicator services required.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   ORCHESTRATOR                       │
│  Market Agent · ML Agent · News Agent · Portfolio   │
└────────────┬────────────────────────────────────────┘
             │
    ┌────────▼────────┐    ┌────────────────────┐
    │  SIGNAL ENGINE  │    │   MEMORY SYSTEM    │
    │  ICT + S/R      │    │  Long-term Memory  │
    │  LSTM + RF      │    │  Macro Context     │
    │  Claude AI      │    └────────────────────┘
    └────────┬────────┘
             │
    ┌────────▼────────┐    ┌────────────────────┐
    │  RISK MANAGER   │    │  EXCHANGE ADAPTER  │
    │  Kelly + ATR    │    │  Binance/Bybit/    │
    │  Partial Exit   │    │  Bitget/OKX        │
    └────────┬────────┘    └────────────────────┘
             │
    ┌────────▼────────┐    ┌────────────────────┐
    │   WEBSOCKET     │◄───│     DASHBOARD      │
    │   SERVER        │───►│   HTML/CSS/JS      │
    └─────────────────┘    └────────────────────┘
```

---

## Project Structure

```
cryptobot-pro/
├── dashboard.html              # Web UI
├── crypto_bot.py               # Main bot v6.0
├── websocket_server.py         # Live bridge
├── requirements.txt
├── .env.example
├── README.md
│
├── exchange/
│   ├── adapter.py              # Universal interface
│   ├── binance_client.py
│   ├── bybit_client.py
│   ├── bitget_client.py
│   └── okx_client.py
│
├── ai_engine/
│   ├── orchestrator.py         # Multi-agent coordinator
│   ├── agent.py                # ReAct AI Agent
│   ├── claude_advisor.py       # Claude final decision
│   ├── lstm_model.py           # TensorFlow BiLSTM
│   ├── rf_model.py             # Random Forest
│   ├── ict_engine.py           # ICT analysis
│   ├── support_resistance.py   # Pivot, Fibonacci, S/R
│   ├── futures_engine.py       # Long/Short/Reversal
│   └── xai_explainer.py        # Explainable AI
│
├── data/
│   ├── collector.py            # Market data fetcher
│   ├── indicators.py           # 25+ technical indicators
│   └── news_engine.py          # Multi-source news
│
├── memory/
│   ├── memory_manager.py       # Long-term memory
│   └── macro_context.py        # Macro data
│
└── risk/
    └── manager.py              # Risk management
```

---

## Deployment (Server/Cloud)

### VPS (Ubuntu)
```bash
# Set WS_HOST=0.0.0.0 in .env
# Open port 8765 in firewall
ufw allow 8765

# Run with pm2
pm2 start "python websocket_server.py" --name ws-server
pm2 start "python crypto_bot.py" --name bot

# Dashboard: change WS URL in dashboard uses window.location.hostname automatically
```

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8765
CMD ["python", "crypto_bot.py"]
```

---

## Disclaimer

This software is for educational purposes only. Cryptocurrency trading involves
significant financial risk. Past performance does not guarantee future results.
Always test with demo mode before using real funds. Never invest more than you
can afford to lose.

---

## License

MIT License — Free to use, modify, distribute.
