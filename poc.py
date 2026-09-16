"""
AlphaBench -- Proof of Concept
==============================

A minimal, *event-driven* backtest that proves the intended toolchain
(Python + yfinance + pandas + NumPy + Matplotlib) works together end to end.

The strategy is a trivial simple-moving-average (SMA) crossover: go fully long
when the short SMA is above the long SMA, and go flat otherwise. What matters
here is not the strategy but the architecture -- bars are streamed one at a
time and the strategy only ever sees information up to the current bar, so
there is no look-ahead bias. This is the skeleton the full framework builds on.

Tested on Ubuntu 22.04 LTS / macOS 14 / Windows 11 with CPython 3.11.
No API keys required.
"""

from collections import deque

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import yfinance as yf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TICKER = "SPY"
START = "2015-01-01"
END = "2024-12-31"
SHORT_WINDOW = 50
LONG_WINDOW = 200
COMMISSION = 0.0005          # 5 bps per side, charged on turnover
TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Data handler
# ---------------------------------------------------------------------------
def load_bars(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV bars and return a clean, single-index frame."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}. Check the ticker/date range.")
    # Newer yfinance returns MultiIndex columns even for a single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


# ---------------------------------------------------------------------------
# Event-driven loop
# ---------------------------------------------------------------------------
def run_backtest(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Stream bars one at a time. Each iteration the strategy sees only closes up
    to and including the current bar, decides a target position for the NEXT
    bar, and the portfolio is marked to market. Returns a results frame.
    """
    short_hist = deque(maxlen=SHORT_WINDOW)
    long_hist = deque(maxlen=LONG_WINDOW)

    dates, equity, positions, prices = [], [], [], []
    cash_equity = 1.0        # start with 1 unit of capital (normalised)
    target_position = 0      # 0 = flat, 1 = long; decided on the PREVIOUS bar
    prev_close = None

    for date, row in bars.iterrows():
        close = float(row["Close"])

        # --- mark-to-market: apply the return earned since the last bar ---
        if prev_close is not None:
            bar_return = (close / prev_close) - 1.0
            cash_equity *= (1.0 + target_position * bar_return)

        # --- generate the signal using ONLY information up to this bar ---
        short_hist.append(close)
        long_hist.append(close)
        new_target = target_position
        if len(long_hist) == LONG_WINDOW:
            short_sma = np.mean(short_hist)
            long_sma = np.mean(long_hist)
            new_target = 1 if short_sma > long_sma else 0

        # --- charge commission on any change in exposure (turnover) ---
        if new_target != target_position:
            cash_equity *= (1.0 - COMMISSION * abs(new_target - target_position))

        target_position = new_target
        prev_close = close

        dates.append(date)
        equity.append(cash_equity)
        positions.append(target_position)
        prices.append(close)

    results = pd.DataFrame(
        {"price": prices, "position": positions, "equity": equity}, index=pd.Index(dates, name="date")
    )
    results["buy_and_hold"] = results["price"] / results["price"].iloc[0]
    return results


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------
def performance(results: pd.DataFrame) -> dict:
    rets = results["equity"].pct_change().dropna()
    total_return = results["equity"].iloc[-1] - 1.0
    ann_return = (1.0 + total_return) ** (TRADING_DAYS / len(rets)) - 1.0
    ann_vol = rets.std() * np.sqrt(TRADING_DAYS)
    sharpe = (rets.mean() * TRADING_DAYS) / ann_vol if ann_vol > 0 else float("nan")
    running_max = results["equity"].cummax()
    max_drawdown = (results["equity"] / running_max - 1.0).min()
    return {
        "Total return": f"{total_return:6.1%}",
        "Annualised return": f"{ann_return:6.1%}",
        "Annualised volatility": f"{ann_vol:6.1%}",
        "Sharpe ratio": f"{sharpe:6.2f}",
        "Max drawdown": f"{max_drawdown:6.1%}",
    }


def plot(results: pd.DataFrame, path: str = "equity_curve.png") -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results.index, results["equity"], label="SMA crossover strategy", linewidth=1.6)
    ax.plot(results.index, results["buy_and_hold"], label="Buy & hold", linewidth=1.2, alpha=0.7)
    ax.set_title(f"AlphaBench PoC -- {TICKER} {SHORT_WINDOW}/{LONG_WINDOW} SMA crossover")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of 1 unit of capital")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nSaved equity curve to {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Downloading {TICKER} bars ({START} to {END}) ...")
    bars = load_bars(TICKER, START, END)
    print(f"Loaded {len(bars)} daily bars.")

    results = run_backtest(bars)

    print("\nPerformance (strategy):")
    for k, v in performance(results).items():
        print(f"  {k:<24} {v}")

    plot(results)
    print("\nProof of concept ran successfully.")


if __name__ == "__main__":
    main()
