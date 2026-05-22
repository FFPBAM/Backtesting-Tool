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
ASSETS = ["Equity", "Bonds", "Gold", "Cash"]


def build_returns(
    prices: pd.DataFrame,
    cash_yield_pa: float,
) -> pd.DataFrame:
    """Return monthly simple returns for Equity, Bonds, Gold, Cash.

    `prices` is the monthly DataFrame from data_loader.build_monthly_series.
    `cash_yield_pa` is in decimal (e.g. 0.02 for 2 % p.a.). Cash return per
    month is (1 + r_pa) ** (1/12) - 1.
    """
    rets = prices[["Equity", "Bonds", "Gold"]].pct_change()
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


def _portfolio_value_path(
    rets: pd.DataFrame,
    w0: np.ndarray,
    reb_dates: set[pd.Timestamp],
    start_value: float = 100.0,
) -> np.ndarray:
    """Return the portfolio value path as a numpy array.

    Fast path: if rebalancing happens every month, the portfolio return is
    simply the weighted average of asset returns each month — fully
    vectorised, no Python loop. Otherwise fall back to the sleeve loop.
    """
    r = rets.values  # (T, n_assets)
    n_months = r.shape[0]

    # Fast path — monthly rebalancing (reb_dates covers every month)
    if len(reb_dates) >= n_months:
        port_ret = r @ w0                       # weighted average per month
        path = start_value * np.cumprod(1.0 + port_ret)
        return path

    # General path — sleeve tracking with periodic rebalancing
    sleeve = w0 * start_value
    out = np.empty(n_months)
    dates = rets.index
    for t in range(n_months):
        sleeve = sleeve * (1.0 + r[t])
        total = sleeve.sum()
        if dates[t] in reb_dates:
            sleeve = w0 * total
        out[t] = total
    return out


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
    assets = ASSETS
    w0 = np.array([weights.get(a, 0.0) for a in assets], dtype=float)
    if not np.isclose(w0.sum(), 1.0, atol=1e-6):
        raise ValueError(f"Weights must sum to 1, got {w0.sum():.4f}")

    rets = returns[assets]
    reb_dates = _rebalance_dates(rets.index, rebalance)
    pv_arr = _portfolio_value_path(rets, w0, reb_dates, start_value)

    pv = pd.Series(pv_arr, index=rets.index, name=name)
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


# --------------------------------------------------------------------------
# 4. Rolling-window backtest
# --------------------------------------------------------------------------
StepFreq = Literal["M", "Y"]


@dataclass
class RollingResult:
    """Results of a rolling-window backtest.

    Attributes
    ----------
    window_years : length of each window in years
    metrics : DataFrame, one row per window, indexed by window start date.
              Columns: 'Start', 'End', 'CAGR', 'Volatility', 'Max Drawdown',
              'Sharpe', 'Calmar', 'Total Return'.
    paths : DataFrame of normalised equity curves. Each column is one window
            (named by its start date), rows are months-since-start (0..N).
            All paths start at 100.
    """
    name: str
    window_years: int
    metrics: pd.DataFrame
    paths: pd.DataFrame


def rolling_backtest(
    returns: pd.DataFrame,
    weights: dict[str, float],
    window_years: int,
    rebalance: RebalanceFreq = "M",
    step: StepFreq = "M",
    risk_free_pa: float = 0.0,
    name: str = "Portfolio",
) -> RollingResult:
    """Run a rolling-window historical backtest.

    For every start month (or year), simulate the portfolio over the next
    `window_years` years and collect performance / risk metrics. Produces a
    distribution of outcomes rather than a single path.

    Parameters
    ----------
    returns : monthly simple returns (Equity / Bonds / Gold / Cash)
    weights : target allocation, must sum to 1
    window_years : window length in years (e.g. 10)
    rebalance : rebalancing frequency inside each window
    step : 'M' (shift window by 1 month) or 'Y' (shift by 12 months)
    risk_free_pa : risk-free rate for Sharpe
    """
    assets = ASSETS
    w0 = np.array([weights.get(a, 0.0) for a in assets], dtype=float)
    if not np.isclose(w0.sum(), 1.0, atol=1e-6):
        raise ValueError(f"Weights must sum to 1, got {w0.sum():.4f}")

    rets = returns[assets]
    window_months = window_years * 12
    n = len(rets)
    if n < window_months:
        raise ValueError(
            f"Datenhistorie ({n} Monate) ist kürzer als das Fenster "
            f"({window_months} Monate). Fensterlänge reduzieren oder "
            f"früheres Startjahr wählen."
        )

    step_months = 1 if step == "M" else 12
    dates = rets.index
    reb_all = _rebalance_dates(dates, rebalance)
    monthly_reb = len(reb_all) >= n  # fast path applicable

    metric_rows = []
    path_cols = {}
    for start_idx in range(0, n - window_months + 1, step_months):
        sl = slice(start_idx, start_idx + window_months)
        window = rets.iloc[sl]
        if monthly_reb:
            reb_w = set(window.index)  # every month
        else:
            reb_w = {d for d in window.index if d in reb_all}
        pv = _portfolio_value_path(window, w0, reb_w, start_value=100.0)

        # --- metrics straight from the numpy path -----------------------
        # monthly returns within the window
        m_ret = pv / np.concatenate([[100.0], pv[:-1]]) - 1.0
        total = pv[-1] / 100.0
        years = window_months / 12.0
        cagr = total ** (1.0 / years) - 1.0 if total > 0 else float("nan")
        vol = m_ret.std(ddof=1) * np.sqrt(12.0)
        # max drawdown anchored at 100
        path_full = np.concatenate([[100.0], pv])
        peak = np.maximum.accumulate(path_full)
        mdd = (path_full / peak - 1.0).min()
        sharpe = (cagr - risk_free_pa) / vol if vol > 0 else float("nan")
        calmar = cagr / abs(mdd) if mdd < 0 else float("nan")

        start_date = window.index[0]
        metric_rows.append({
            "Start": start_date,
            "End": window.index[-1],
            "CAGR": cagr,
            "Volatility": vol,
            "Max Drawdown": mdd,
            "Sharpe": sharpe,
            "Calmar": calmar,
            "Total Return": total - 1.0,
        })
        # normalised equity path, prepended with 100 at month 0
        path_cols[start_date.strftime("%Y-%m")] = np.concatenate([[100.0], pv])

    metrics = pd.DataFrame(metric_rows).set_index("Start")
    paths = pd.DataFrame(path_cols)
    paths.index.name = "Month"

    return RollingResult(
        name=name,
        window_years=window_years,
        metrics=metrics,
        paths=paths,
    )


def rolling_summary(roll: RollingResult) -> pd.DataFrame:
    """Distribution summary of rolling metrics: min / percentiles / max.

    Returns a DataFrame indexed by statistic, columns are the metrics.
    """
    cols = ["CAGR", "Volatility", "Max Drawdown", "Sharpe", "Calmar"]
    df = roll.metrics[cols]
    stats = {
        "Minimum": df.min(),
        "5 %-Perzentil": df.quantile(0.05),
        "25 %-Perzentil": df.quantile(0.25),
        "Median": df.quantile(0.50),
        "Mittelwert": df.mean(),
        "75 %-Perzentil": df.quantile(0.75),
        "95 %-Perzentil": df.quantile(0.95),
        "Maximum": df.max(),
    }
    out = pd.DataFrame(stats).T
    out["Anzahl Fenster"] = len(df)
    return out


def fan_chart_bands(
    roll: RollingResult,
    percentiles: tuple[float, ...] = (5, 25, 50, 75, 95),
) -> pd.DataFrame:
    """Compute percentile bands across all rolling equity paths.

    Returns a DataFrame indexed by months-since-start, with one column per
    requested percentile (named 'p5', 'p25', ...). All paths normalised to
    100 at month 0.
    """
    paths = roll.paths
    out = {}
    for p in percentiles:
        out[f"p{int(p)}"] = paths.quantile(p / 100.0, axis=1)
    band = pd.DataFrame(out)
    band.index.name = "Month"
    return band
