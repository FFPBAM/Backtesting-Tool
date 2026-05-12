"""Portfolio simulation engine.

Given a monthly DataFrame with columns ['Equity', 'Bonds'] plus a synthetic
cash leg constructed from a user-supplied Euribor yield, this module
produces:

* a portfolio total-return index (rebalanced at a chosen frequency)
* annual performance per calendar year + total / annualised
* annualised volatility (per calendar year + total, monthly returns × √12)
* maximum drawdown (per calendar year OR rolling 1-year)
* Sharpe ratio, Calmar ratio
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


RebalanceFreq = Literal["none", "M", "Q", "Y"]
DrawdownMode = Literal["calendar_year", "rolling_1y"]


# --------------------------------------------------------------------------
# 1. Build asset return panel (incl. synthetic cash)
# --------------------------------------------------------------------------
def build_returns(
    prices: pd.DataFrame,
    cash_yield_pa: float,
) -> pd.DataFrame:
    """Return monthly simple returns for Equity, Bonds, Cash.

    `prices` is the monthly DataFrame from data_loader.build_monthly_series.
    `cash_yield_pa` is in decimal (e.g. 0.02 for 2 % p.a.). Cash return per
    month is (1 + r_pa) ** (1/12) - 1.
    """
    rets = prices[["Equity", "Bonds"]].pct_change()
    monthly_cash = (1.0 + cash_yield_pa) ** (1.0 / 12.0) - 1.0
    rets["Cash"] = monthly_cash
    rets = rets.dropna(how="any")
    return rets


# --------------------------------------------------------------------------
# 2. Simulate portfolio
# --------------------------------------------------------------------------
def _rebalance_dates(
    index: pd.DatetimeIndex,
    freq: RebalanceFreq,
) -> set[pd.Timestamp]:
    """Return the set of dates on which rebalancing happens.

    'none'  -> never rebalance (buy & hold)
    'M'     -> every month (data frequency = default)
    'Q'     -> quarter-end
    'Y'     -> year-end
    """
    if freq == "none":
        return set()
    if freq == "M":
        return set(index)
    if freq == "Q":
        return set(d for d in index if d.month in (3, 6, 9, 12))
    if freq == "Y":
        return set(d for d in index if d.month == 12)
    raise ValueError(f"Unknown rebalance frequency: {freq}")


@dataclass
class PortfolioResult:
    name: str
    weights: dict[str, float]
    rebalance: RebalanceFreq
    index: pd.Series          # portfolio value (start = 100)
    monthly_returns: pd.Series
    annual_returns: pd.Series  # per calendar year (decimal)


def simulate(
    returns: pd.DataFrame,
    weights: dict[str, float],
    rebalance: RebalanceFreq = "M",
    name: str = "Portfolio",
    start_value: float = 100.0,
) -> PortfolioResult:
    """Run the historical simulation.

    Parameters
    ----------
    returns : DataFrame with monthly simple returns for Equity / Bonds / Cash
    weights : dict with the same keys, must sum to ~1
    rebalance : 'none' | 'M' | 'Q' | 'Y'
    """
    assets = ["Equity", "Bonds", "Cash"]
    w0 = np.array([weights.get(a, 0.0) for a in assets], dtype=float)
    if not np.isclose(w0.sum(), 1.0, atol=1e-6):
        raise ValueError(f"Weights must sum to 1, got {w0.sum():.4f}")

    rets = returns[assets].copy()
    reb_dates = _rebalance_dates(rets.index, rebalance)

    # Track value of each sleeve; rebalance by resetting sleeve values to
    # target weights * total at the chosen dates.
    sleeve_vals = w0 * start_value
    pv = []  # portfolio total values
    for date, row in rets.iterrows():
        sleeve_vals = sleeve_vals * (1.0 + row[assets].values)
        total = sleeve_vals.sum()
        if date in reb_dates:
            sleeve_vals = w0 * total
        pv.append(total)

    pv = pd.Series(pv, index=rets.index, name=name)
    monthly_ret = pv.pct_change().fillna((pv.iloc[0] / start_value) - 1.0)
    annual_ret = (1.0 + monthly_ret).groupby(monthly_ret.index.year).prod() - 1.0
    annual_ret.index.name = "Year"

    return PortfolioResult(
        name=name,
        weights=dict(zip(assets, w0)),
        rebalance=rebalance,
        index=pv,
        monthly_returns=monthly_ret,
        annual_returns=annual_ret,
    )


# --------------------------------------------------------------------------
# 3. Risk & return metrics
# --------------------------------------------------------------------------
def annualised_return(monthly_returns: pd.Series) -> float:
    """CAGR derived from monthly compounded returns."""
    if len(monthly_returns) == 0:
        return float("nan")
    total = (1.0 + monthly_returns).prod()
    years = len(monthly_returns) / 12.0
    if years <= 0 or total <= 0:
        return float("nan")
    return total ** (1.0 / years) - 1.0


def annualised_volatility(monthly_returns: pd.Series) -> float:
    """Std of monthly returns × √12."""
    if len(monthly_returns) < 2:
        return float("nan")
    return monthly_returns.std(ddof=1) * np.sqrt(12.0)


def max_drawdown(value_series: pd.Series) -> float:
    """Max peak-to-trough drawdown of the value series (negative decimal)."""
    if len(value_series) == 0:
        return float("nan")
    peak = value_series.cummax()
    dd = value_series / peak - 1.0
    return dd.min()


def drawdown_series(value_series: pd.Series) -> pd.Series:
    """Full drawdown path (≤ 0)."""
    peak = value_series.cummax()
    return value_series / peak - 1.0


def per_year_metrics(
    result: PortfolioResult,
    dd_mode: DrawdownMode = "calendar_year",
) -> pd.DataFrame:
    """One row per calendar year: return, vola p.a., max drawdown.

    Drawdown anchors at the previous December value so that a fall during
    January is correctly captured. For the very first observable year (no
    prior December) the anchor is the year's first month-end value.
    """
    pv = result.index
    rows = []
    for year, m_ret in result.monthly_returns.groupby(result.monthly_returns.index.year):
        vals_year = pv[pv.index.year == year]
        if len(vals_year) == 0:
            continue
        # Anchor: last value of the previous year if available
        prev = pv[pv.index.year == year - 1]
        if len(prev) > 0:
            anchor = pd.Series([prev.iloc[-1]], index=[prev.index[-1]])
            vals_anchored = pd.concat([anchor, vals_year])
        else:
            vals_anchored = vals_year
        dd = max_drawdown(vals_anchored)
        rows.append({
            "Year": int(year),
            "Return": (1.0 + m_ret).prod() - 1.0,
            "Volatility p.a.": annualised_volatility(m_ret),
            "Max Drawdown": dd,
        })
    return pd.DataFrame(rows).set_index("Year")


def total_metrics(
    result: PortfolioResult,
    risk_free_pa: float = 0.0,
    dd_mode: DrawdownMode = "calendar_year",
) -> dict[str, float]:
    """Summary metrics across the whole period."""
    ann_ret = annualised_return(result.monthly_returns)
    ann_vol = annualised_volatility(result.monthly_returns)

    if dd_mode == "rolling_1y":
        # Worst 12-month return
        rolling_12m = (1.0 + result.monthly_returns).rolling(12).apply(np.prod, raw=True) - 1.0
        mdd = rolling_12m.min()
    else:
        mdd = max_drawdown(result.index)

    sharpe = (ann_ret - risk_free_pa) / ann_vol if ann_vol and ann_vol > 0 else float("nan")
    calmar = ann_ret / abs(mdd) if mdd and mdd < 0 else float("nan")
    total_ret = result.index.iloc[-1] / 100.0 - 1.0
    years = len(result.monthly_returns) / 12.0

    return {
        "Gesamtrendite": total_ret,
        "Rendite p.a. (CAGR)": ann_ret,
        "Volatilität p.a.": ann_vol,
        "Max Drawdown": mdd,
        "Sharpe Ratio": sharpe,
        "Calmar Ratio": calmar,
        "Beobachtungsjahre": years,
    }
