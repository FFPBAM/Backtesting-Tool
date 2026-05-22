"""Data loading and preparation for the portfolio simulator."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st


DATA_PATH = Path(__file__).parent / "data" / "Daten_Verrentung_EUR.xlsx"

# Bump this string whenever the output schema of build_monthly_series
# changes. It is part of the cache key, so a new value invalidates any
# stale cached result on Streamlit Cloud automatically.
SCHEMA_VERSION = "2024-06-gold"

# Columns the rest of the app relies on (asset price columns).
REQUIRED_PRICE_COLS = ["Equity", "Bonds", "Gold"]


@st.cache_data(show_spinner=False)
def load_raw_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the Bloomberg data export.

    Columns of interest:
        - NDDUWI Index : MSCI World Net Total Return (in EUR, per user)
        - REXP Index   : Deutscher Rentenindex (Performance, EUR)
        - Gold         : Gold spot price
    """
    df = pd.read_excel(path, sheet_name="Import_Daten")
    df["Dates"] = pd.to_datetime(df["Dates"])
    df = df.sort_values("Dates").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def build_monthly_series(start_year: int, _schema: str = SCHEMA_VERSION) -> pd.DataFrame:
    """Return a clean monthly DataFrame indexed by month-end date.

    Columns: Equity (MSCI World EUR), Bonds (REXP), Gold (spot price).
    The Cash leg is synthesised in `simulation.build_returns` from a
    user-supplied constant yield, so it is not part of this frame.

    The `_schema` parameter is part of the cache key only; passing the
    current SCHEMA_VERSION ensures a stale cache from an older app build
    is automatically invalidated.
    """
    df = load_raw_data()
    df = df[df["Dates"].dt.year >= start_year - 1].copy()  # keep prev. Dec

    src_cols = {
        "NDDUWI Index": "Equity",
        "REXP Index": "Bonds",
        "Gold": "Gold",
    }
    missing_src = [c for c in src_cols if c not in df.columns]
    if missing_src:
        raise KeyError(
            f"Erwartete Spalten fehlen in der Excel-Datei: {missing_src}. "
            f"Vorhandene Spalten: {list(df.columns)}"
        )

    df = df[["Dates", *src_cols.keys()]].dropna()
    df = df.rename(columns=src_cols)
    df = df.set_index("Dates")
    return df


def assets_available_from(df: pd.DataFrame) -> pd.Timestamp:
    """Earliest date where all required assets have data."""
    return df.dropna().index.min()
