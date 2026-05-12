"""Data loading and preparation for the portfolio simulator."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st


DATA_PATH = Path(__file__).parent / "data" / "Daten_Verrentung_EUR.xlsx"


@st.cache_data(show_spinner=False)
def load_raw_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the Bloomberg data export.

    Columns of interest:
        - NDDUWI Index : MSCI World Net Total Return (in EUR, per user)
        - REXP Index   : Deutscher Rentenindex (Performance, EUR)
    """
    df = pd.read_excel(path, sheet_name="Import_Daten")
    df["Dates"] = pd.to_datetime(df["Dates"])
    df = df.sort_values("Dates").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def build_monthly_series(start_year: int) -> pd.DataFrame:
    """Return a clean monthly DataFrame indexed by month-end date with
    columns: Equity (MSCI World EUR), Bonds (REXP), Cash (Euribor index).

    The Cash series is synthesized from a constant 2 % p.a. fallback if no
    Euribor column exists in the workbook; otherwise the Euribor column is
    compounded into a total-return index. We approximate Cash with a simple
    rolling-yield index from the 1M Euribor when available (see README).
    """
    df = load_raw_data()
    df = df[df["Dates"].dt.year >= start_year - 1].copy()  # keep prev. Dec
    df = df[["Dates", "NDDUWI Index", "REXP Index"]].dropna()
    df = df.rename(columns={"NDDUWI Index": "Equity", "REXP Index": "Bonds"})

    # ---- Cash leg: synthetic Euribor 1M total-return index --------------
    # We don't have Euribor in the workbook, so the Streamlit UI lets the
    # user pick a constant annualised cash yield (default 2 %). The series
    # here is rebuilt on the fly in `simulate()` to keep this function pure.
    df = df.set_index("Dates")
    return df


def assets_available_from(df: pd.DataFrame) -> pd.Timestamp:
    """Earliest date where all required assets have data."""
    return df.dropna().index.min()
