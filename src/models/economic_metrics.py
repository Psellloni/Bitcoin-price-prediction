from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def sharpe_ratio(
    returns: np.ndarray,
    bars_per_year: int = 365 * 96,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Annualised Sharpe ratio.

    Parameters
    ----------
    returns       : per-bar (e.g. 15-min) strategy return series
    bars_per_year : number of bars in one calendar year
                    (15-min crypto: 365 * 96 = 35 040)
    risk_free_rate: annualised risk-free rate (default 0)

    Returns
    -------
    Annualised Sharpe ratio (float).
    """
    returns = np.asarray(returns, dtype=float)
    excess = returns - risk_free_rate / bars_per_year
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(bars_per_year))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of an equity curve."""
    equity_curve = np.asarray(equity_curve, dtype=float)
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    return float(dd.min())


def total_return(equity_curve: np.ndarray) -> float:
    """Total return from start to end of an equity curve."""
    equity_curve = np.asarray(equity_curve, dtype=float)
    return float((equity_curve[-1] - equity_curve[0]) / equity_curve[0])


# ---------------------------------------------------------------------------
# Trading strategy
# ---------------------------------------------------------------------------

def simple_momentum_strategy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    initial_capital: float = 10_000.0,
    transaction_cost: float = 0.001,
) -> dict:
    """
    Simple long-only momentum strategy driven by model predictions.

    Signal rule:
        If y_pred[t] > y_true[t]  (model predicts price will rise) → go long
        Otherwise                                                    → stay flat

    Position sizing: invest 100 % of capital when long, 0 % when flat.

    Parameters
    ----------
    y_true           : realised price series (length n)
    y_pred           : predicted price series (length n, aligned with y_true)
    initial_capital  : starting capital in USD
    transaction_cost : one-way cost as a fraction of trade value

    Returns
    -------
    dict with keys:
        equity_curve  : np.ndarray — capital through time
        bar_returns   : np.ndarray — per-bar strategy return series
        PnL           : float      — final profit/loss in USD
        total_return  : float      — total return as fraction
        sharpe        : float      — annualised Sharpe ratio
        max_drawdown  : float      — maximum drawdown as fraction
        n_trades      : int        — number of position changes
        turnover      : float      — total absolute position turnover
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]

    # Signal: 1 = long, 0 = flat
    signal = (y_pred > y_true).astype(float)

    # Actual bar return: (close[t+1] - close[t]) / close[t]
    price_returns = np.diff(y_true) / y_true[:-1]

    # Strategy return: signal[t] * price_return[t+1]
    strategy_returns = signal[:-1] * price_returns

    # Apply transaction costs on signal changes
    signal_changes = np.diff(signal, prepend=signal[0])
    strategy_returns -= np.abs(signal_changes[:-1]) * transaction_cost

    # Equity curve
    equity = initial_capital * np.cumprod(1.0 + strategy_returns)
    equity = np.insert(equity, 0, initial_capital)

    pnl = equity[-1] - initial_capital
    ret = total_return(equity)
    sharpe = sharpe_ratio(strategy_returns)
    mdd = max_drawdown(equity)
    n_trades = int(np.sum(np.abs(signal_changes)))
    turnover = float(np.sum(np.abs(signal_changes)))

    return {
        "equity_curve": equity,
        "bar_returns": strategy_returns,
        "PnL": pnl,
        "total_return": ret,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "n_trades": n_trades,
        "turnover": turnover,
    }


def buy_and_hold(
    y_true: np.ndarray,
    initial_capital: float = 10_000.0,
) -> dict:
    """
    Buy-and-hold benchmark: invest everything at time 0 and hold.
    """
    y_true = np.asarray(y_true, dtype=float)
    price_returns = np.diff(y_true) / y_true[:-1]
    equity = initial_capital * np.cumprod(1.0 + price_returns)
    equity = np.insert(equity, 0, initial_capital)

    return {
        "equity_curve": equity,
        "bar_returns": price_returns,
        "PnL": equity[-1] - initial_capital,
        "total_return": total_return(equity),
        "sharpe": sharpe_ratio(price_returns),
        "max_drawdown": max_drawdown(equity),
        "n_trades": 1,
        "turnover": 1.0,
    }


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------

def compare_strategies(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    initial_capital: float = 10_000.0,
) -> pd.DataFrame:
    """
    Compare multiple model strategies vs. buy-and-hold.

    Parameters
    ----------
    y_true      : realised price series
    predictions : dict mapping model name → predicted price series
    initial_capital : starting capital

    Returns
    -------
    pd.DataFrame with one row per strategy.
    """
    rows = []

    bh = buy_and_hold(y_true, initial_capital)
    rows.append(
        {
            "Strategy": "Buy & Hold",
            "Total Return (%)": round(bh["total_return"] * 100, 2),
            "Sharpe": round(bh["sharpe"], 3),
            "Max Drawdown (%)": round(bh["max_drawdown"] * 100, 2),
            "Turnover": round(bh["turnover"], 3),
            "Number of Trades": int(bh["n_trades"]),
            "PnL (USD)": round(bh["PnL"], 2),
        }
    )

    for name, y_pred in predictions.items():
        res = simple_momentum_strategy(y_true, y_pred, initial_capital)
        rows.append(
            {
                "Strategy": name,
                "Total Return (%)": round(res["total_return"] * 100, 2),
                "Sharpe": round(res["sharpe"], 3),
                "Max Drawdown (%)": round(res["max_drawdown"] * 100, 2),
                "Turnover": round(res["turnover"], 3),
                "Number of Trades": int(res["n_trades"]),
                "PnL (USD)": round(res["PnL"], 2),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_equity_curves(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    initial_capital: float = 10_000.0,
    title: str = "Strategy Equity Curves",
):
    """Plot equity curves for all strategies and buy-and-hold."""
    bh = buy_and_hold(y_true, initial_capital)

    plt.figure(figsize=(12, 5))
    plt.plot(bh["equity_curve"], label="Buy & Hold", linewidth=2, color="black")

    for name, y_pred in predictions.items():
        res = simple_momentum_strategy(y_true, y_pred, initial_capital)
        plt.plot(res["equity_curve"], label=name, linewidth=1.5)

    plt.title(title)
    plt.xlabel("Bar")
    plt.ylabel("Portfolio Value (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
