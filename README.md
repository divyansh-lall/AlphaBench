# AlphaBench: Proof of Concept

A minimal, **event-driven** backtest that demonstrates the toolchain planned for
AlphaBench (an event-driven backtesting framework for systematic trading
strategies) works together end to end: **Python + yfinance + pandas + NumPy +
Matplotlib**.

The demo runs a simple moving-average (SMA) crossover on daily data. Bars are
streamed **one at a time**, and the strategy only sees information up to the
current bar, so there is no look-ahead bias. The strategy is deliberately
trivial; it exists only to prove the pieces interoperate and to establish the
skeleton the full framework will build on.

## Requirements

- Python **3.11+** (developed and tested on Ubuntu 22.04 LTS; also runs on
  macOS 14 and Windows 11)
- No API keys required

## Setup & run

```bash
# 1. Clone
git clone https://github.com/your-username/alphabench-poc.git
cd alphabench-poc

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python poc.py
```

The script prints performance metrics (total/annualised return, volatility,
Sharpe ratio, max drawdown) to the console and writes `equity_curve.png` to the
project directory.

## What it does

| Component      | Role in the demo                                              |
|----------------|--------------------------------------------------------------|
| Data handler   | Downloads daily OHLCV bars via `yfinance`                     |
| Event loop     | Streams bars one at a time; exposes only past data            |
| Strategy       | SMA crossover → long / flat signal                            |
| Portfolio      | Marks to market, applies commission on turnover              |
| Analytics      | Computes metrics and plots the equity curve vs. buy-and-hold |

## License

MIT
