"""Historische Portfolio-Simulation – Streamlit-App.

Run locally:
    streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import build_monthly_series, load_raw_data
from simulation import (
    build_returns,
    drawdown_series,
    per_year_metrics,
    simulate,
    total_metrics,
)

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Portfolio-Simulation",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Historische Portfolio-Simulation")
st.caption(
    "MSCI World (EUR) · REXP · Liquidität (Euribor-Approximation) — "
    "Datenbasis: Bloomberg-Monatsultimo"
)

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
raw = load_raw_data()
first_full_year = 2000  # MSCI World EUR + REXP both complete from 1999-12-31
last_year = int(raw["Dates"].max().year)

# --------------------------------------------------------------------------
# Sidebar — global settings
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Globale Einstellungen")

    start_year = st.slider(
        "Startjahr",
        min_value=first_full_year,
        max_value=last_year - 1,
        value=first_full_year,
        step=1,
        help="Simulation beginnt am 31.12. des Vorjahres.",
    )

    end_year = st.slider(
        "Endjahr",
        min_value=start_year + 1,
        max_value=last_year,
        value=last_year,
        step=1,
    )

    st.divider()
    st.subheader("Liquidität (Cash-Verzinsung)")
    cash_yield_pct = st.number_input(
        "Cash-Rendite p.a. (%)",
        min_value=-2.0,
        max_value=15.0,
        value=2.0,
        step=0.25,
        help=(
            "Da keine Euribor-Zeitreihe in den Daten enthalten ist, wird "
            "Cash als konstante Verzinsung modelliert. Häufige Werte: "
            "0 % (vereinfachend), ~2 % (langfr. Durchschnitt), aktuell ~3 %."
        ),
    )
    cash_yield = cash_yield_pct / 100.0

    st.divider()
    st.subheader("Risiko & Kennzahlen")
    dd_mode = st.radio(
        "Max-Drawdown-Definition (jährlich)",
        options=["calendar_year", "rolling_1y"],
        format_func=lambda x: {
            "calendar_year": "Innerhalb des Kalenderjahres (Default)",
            "rolling_1y": "Rollierend 1 Jahr (für Gesamtmetriken)",
        }[x],
        index=0,
    )
    rf_pct = st.number_input(
        "Risk-free für Sharpe Ratio (% p.a.)",
        min_value=-2.0, max_value=10.0, value=2.0, step=0.25,
    )
    risk_free = rf_pct / 100.0

# --------------------------------------------------------------------------
# Build returns universe once
# --------------------------------------------------------------------------
prices = build_monthly_series(start_year)
prices = prices[(prices.index.year >= start_year - 1) & (prices.index.year <= end_year)]
returns = build_returns(prices, cash_yield_pa=cash_yield)
# Keep only the configured window for the simulation
returns = returns[(returns.index.year >= start_year) & (returns.index.year <= end_year)]

if returns.empty:
    st.error("Keine Daten im gewählten Zeitraum verfügbar.")
    st.stop()

# --------------------------------------------------------------------------
# Portfolio definitions — up to 3 side-by-side
# --------------------------------------------------------------------------
st.subheader("🎯 Portfolio-Konfiguration")

n_portfolios = st.radio(
    "Anzahl der zu vergleichenden Portfolios",
    options=[1, 2, 3],
    index=1,
    horizontal=True,
)

DEFAULT_PRESETS = [
    {"name": "Defensiv",    "equity": 30, "bonds": 60, "cash": 10},
    {"name": "Ausgewogen",  "equity": 50, "bonds": 40, "cash": 10},
    {"name": "Wachstum",    "equity": 70, "bonds": 25, "cash":  5},
]

configs = []
cols = st.columns(n_portfolios)
for i in range(n_portfolios):
    with cols[i]:
        preset = DEFAULT_PRESETS[i]
        st.markdown(f"**Portfolio {i + 1}**")
        name = st.text_input(
            "Name", value=preset["name"], key=f"name_{i}",
            label_visibility="collapsed",
        )
        eq = st.slider("Aktien (MSCI World) %", 0, 100, preset["equity"], key=f"eq_{i}")
        bd = st.slider("Renten (REXP) %", 0, 100 - eq, preset["bonds"], key=f"bd_{i}")
        cash = 100 - eq - bd
        st.metric("Liquidität %", cash)

        reb = st.selectbox(
            "Rebalancing",
            options=["M", "Q", "Y", "none"],
            format_func=lambda x: {
                "M": "Monatlich (Default)",
                "Q": "Quartalsweise",
                "Y": "Jährlich",
                "none": "Nie (Buy & Hold)",
            }[x],
            key=f"reb_{i}",
        )
        configs.append({
            "name": name,
            "weights": {"Equity": eq / 100, "Bonds": bd / 100, "Cash": cash / 100},
            "rebalance": reb,
        })

# --------------------------------------------------------------------------
# Run simulations
# --------------------------------------------------------------------------
results = []
for cfg in configs:
    res = simulate(
        returns=returns,
        weights=cfg["weights"],
        rebalance=cfg["rebalance"],
        name=cfg["name"],
    )
    results.append(res)

# --------------------------------------------------------------------------
# Summary cards
# --------------------------------------------------------------------------
st.subheader("📊 Gesamtkennzahlen")
summary_rows = []
for res in results:
    m = total_metrics(res, risk_free_pa=risk_free, dd_mode=dd_mode)
    summary_rows.append({"Portfolio": res.name, **m})

summary_df = pd.DataFrame(summary_rows).set_index("Portfolio")
fmt = {
    "Gesamtrendite": "{:.2%}",
    "Rendite p.a. (CAGR)": "{:.2%}",
    "Volatilität p.a.": "{:.2%}",
    "Max Drawdown": "{:.2%}",
    "Sharpe Ratio": "{:.2f}",
    "Calmar Ratio": "{:.2f}",
    "Beobachtungsjahre": "{:.1f}",
}
st.dataframe(
    summary_df.style.format(fmt),
    use_container_width=True,
)

# --------------------------------------------------------------------------
# Chart 1 — Wertentwicklung (indexiert auf 100)
# --------------------------------------------------------------------------
st.subheader("📈 Wertentwicklung (Index = 100 zu Beginn)")
fig_perf = go.Figure()
for res in results:
    fig_perf.add_trace(
        go.Scatter(
            x=res.index.index, y=res.index.values,
            mode="lines", name=res.name,
            hovertemplate="%{x|%b %Y}<br>%{y:.1f}<extra>" + res.name + "</extra>",
        )
    )
fig_perf.update_layout(
    height=450, hovermode="x unified",
    yaxis_title="Portfoliowert (Index)",
    xaxis_title=None,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_perf, use_container_width=True)

# --------------------------------------------------------------------------
# Chart 2 — Drawdowns
# --------------------------------------------------------------------------
st.subheader("📉 Drawdown-Verlauf")
fig_dd = go.Figure()
for res in results:
    dd = drawdown_series(res.index)
    fig_dd.add_trace(
        go.Scatter(
            x=dd.index, y=dd.values * 100,
            mode="lines", name=res.name, fill="tozeroy",
            hovertemplate="%{x|%b %Y}<br>%{y:.2f}%<extra>" + res.name + "</extra>",
        )
    )
fig_dd.update_layout(
    height=350, hovermode="x unified",
    yaxis_title="Drawdown (%)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_dd, use_container_width=True)

# --------------------------------------------------------------------------
# Chart 3 — Jahresrenditen als Balken
# --------------------------------------------------------------------------
st.subheader("📅 Jahresrenditen")
annual_frames = []
for res in results:
    f = res.annual_returns.to_frame(name=res.name)
    annual_frames.append(f)
annual_df = pd.concat(annual_frames, axis=1)

fig_yrs = go.Figure()
for col in annual_df.columns:
    fig_yrs.add_trace(
        go.Bar(
            x=annual_df.index, y=annual_df[col] * 100,
            name=col,
            hovertemplate="%{x}<br>%{y:.2f}%<extra>" + col + "</extra>",
        )
    )
fig_yrs.update_layout(
    height=400, barmode="group",
    yaxis_title="Rendite (%)",
    xaxis_title="Jahr",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_yrs, use_container_width=True)

# --------------------------------------------------------------------------
# Per-year metrics tables (one expander per portfolio)
# --------------------------------------------------------------------------
st.subheader("🧮 Jahreskennzahlen je Portfolio")
yr_fmt = {
    "Return": "{:.2%}",
    "Volatility p.a.": "{:.2%}",
    "Max Drawdown": "{:.2%}",
}
for res in results:
    with st.expander(f"📂 {res.name}  ·  "
                     f"Aktien {res.weights['Equity']:.0%} / "
                     f"Renten {res.weights['Bonds']:.0%} / "
                     f"Cash {res.weights['Cash']:.0%}  ·  "
                     f"Rebal: {res.rebalance}"):
        yr = per_year_metrics(res, dd_mode=dd_mode)
        yr = yr.rename(columns={
            "Return": "Rendite",
            "Volatility p.a.": "Volatilität p.a.",
            "Max Drawdown": "Max Drawdown",
        })
        st.dataframe(
            yr.style.format({
                "Rendite": "{:.2%}",
                "Volatilität p.a.": "{:.2%}",
                "Max Drawdown": "{:.2%}",
            }),
            use_container_width=True,
        )

# --------------------------------------------------------------------------
# Download buttons
# --------------------------------------------------------------------------
st.subheader("💾 Export")
exp_cols = st.columns(len(results) + 1)
with exp_cols[0]:
    st.download_button(
        "📥 Zusammenfassung (CSV)",
        data=summary_df.to_csv().encode("utf-8"),
        file_name="portfolio_summary.csv",
        mime="text/csv",
    )
for i, res in enumerate(results, start=1):
    yr = per_year_metrics(res, dd_mode=dd_mode)
    with exp_cols[i]:
        st.download_button(
            f"📥 {res.name} – Jahre (CSV)",
            data=yr.to_csv().encode("utf-8"),
            file_name=f"{res.name.replace(' ', '_')}_yearly.csv",
            mime="text/csv",
        )

with st.expander("ℹ️ Methodische Hinweise"):
    st.markdown(
        """
**Datenquellen**
- *Aktien*: MSCI World Net Total Return (Bloomberg-Ticker NDDUWI) – in EUR.
- *Renten*: REXP – Deutscher Rentenindex (Performance).
- *Liquidität*: Konstante Annahme über `Cash-Rendite p.a.` in der Sidebar.
  Eine echte Euribor-Zeitreihe lässt sich später analog der Bond-Spalte
  ergänzen (siehe README).

**Rechenkonventionen**
- Monatsultimo-Daten, Renditen als einfache prozentuale Veränderung.
- Jahresrenditen = Produkt der Monatsrenditen − 1.
- Annualisierte Volatilität = Stdabw. der Monatsrenditen × √12.
- Max Drawdown (Default) = größter Peak-to-Trough-Verlust innerhalb des
  jeweiligen Kalenderjahres bzw. über den Gesamtzeitraum.
- Sharpe Ratio = (CAGR − Risk-free) / Vola p.a.
- Calmar Ratio = CAGR / |Max Drawdown|.
- Rebalancing erfolgt zum Monatsultimo des gewählten Intervalls (Q = März/
  Juni/Sept/Dez, Y = Dezember). „Nie“ entspricht Buy & Hold.

**Steuern & Kosten** werden in dieser Simulation nicht abgebildet.
        """
    )
