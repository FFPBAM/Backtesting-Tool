"""Historische Portfolio-Simulation – Streamlit-App.

Run locally:
    streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_loader import build_monthly_series, load_raw_data
from simulation import (
    build_returns,
    drawdown_series,
    fan_chart_bands,
    per_year_metrics,
    rolling_backtest,
    rolling_summary,
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
    "MSCI World (EUR) · REXP · Gold · Liquidität (Euribor-Approximation) — "
    "Datenbasis: Bloomberg-Monatsultimo"
)

# --------------------------------------------------------------------------
# Risk disclaimer — must be visible above the fold
# --------------------------------------------------------------------------
st.warning(
    "⚠️ **Wichtiger Hinweis:** Die hier dargestellten Berechnungen beruhen "
    "ausschließlich auf historischen Daten und stellen **keine Anlage- oder "
    "Steuerberatung** dar. **Vergangene Wertentwicklungen sind kein "
    "verlässlicher Indikator für zukünftige Ergebnisse.** Wertpapieranlagen "
    "unterliegen Marktschwankungen und können zu Kapitalverlusten bis hin "
    "zum Totalverlust des eingesetzten Kapitals führen. Steuern, "
    "Transaktionskosten und Produktgebühren (z. B. TER) sind in dieser "
    "Simulation nicht berücksichtigt und können die tatsächliche Rendite "
    "erheblich mindern.",
    icon="⚠️",
)

# --------------------------------------------------------------------------
# Data — earliest common history of MSCI World (EUR) + REXP is 1970
# --------------------------------------------------------------------------
raw = load_raw_data()
EARLIEST_YEAR = 1970
last_year = int(raw["Dates"].max().year)

# --------------------------------------------------------------------------
# Sidebar — global settings
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Globale Einstellungen")

    start_year = st.slider(
        "Startjahr",
        min_value=EARLIEST_YEAR,
        max_value=last_year - 1,
        value=2000,
        step=1,
        help=(
            "Frühester gemeinsamer Datenpunkt von MSCI World (EUR) und REXP "
            "ist 1970. Die Simulation beginnt am Jahresanfang."
        ),
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
# Build returns universe
# --------------------------------------------------------------------------
prices = build_monthly_series(start_year)
prices = prices[(prices.index.year >= start_year - 1) & (prices.index.year <= end_year)]
returns_full = build_returns(prices, cash_yield_pa=cash_yield)
returns = returns_full[
    (returns_full.index.year >= start_year) & (returns_full.index.year <= end_year)
]

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
    {"name": "Defensiv",    "equity": 25, "bonds": 60, "gold":  5, "cash": 10},
    {"name": "Ausgewogen",  "equity": 50, "bonds": 45, "gold":  0, "cash":  5},
    {"name": "Wachstum",    "equity": 65, "bonds": 20, "gold": 10, "cash":  5},
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
        eq = st.slider(
            "Aktien (MSCI World) %", 0, 100, preset["equity"], key=f"eq_{i}",
        )
        bd_max = max(0, 100 - eq)
        bd = st.slider(
            "Renten (REXP) %", 0, bd_max,
            min(preset["bonds"], bd_max), key=f"bd_{i}",
        )
        gd_max = max(0, 100 - eq - bd)
        gd = st.slider(
            "Gold %", 0, gd_max,
            min(preset["gold"], gd_max), key=f"gd_{i}",
        )
        cash = 100 - eq - bd - gd
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
            "weights": {
                "Equity": eq / 100,
                "Bonds": bd / 100,
                "Gold": gd / 100,
                "Cash": cash / 100,
            },
            "rebalance": reb,
        })

# --------------------------------------------------------------------------
# Run point-in-time simulations
# --------------------------------------------------------------------------
results = []
for cfg in configs:
    results.append(simulate(
        returns=returns,
        weights=cfg["weights"],
        rebalance=cfg["rebalance"],
        name=cfg["name"],
    ))

# Colour palette shared across charts
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c"]

# ==========================================================================
# TABS
# ==========================================================================
tab_sim, tab_roll = st.tabs([
    "📊 Klassische Simulation (fester Zeitraum)",
    "🔁 Rollierender Backtest (Fächerchart)",
])

# --------------------------------------------------------------------------
# TAB 1 — point-in-time simulation
# --------------------------------------------------------------------------
with tab_sim:
    st.caption(
        f"Ein durchgehender Backtest über den gewählten Zeitraum "
        f"{start_year}–{end_year}."
    )

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
    st.dataframe(summary_df.style.format(fmt), use_container_width=True)

    # Chart 1 — Wertentwicklung
    st.subheader("📈 Wertentwicklung (Index = 100 zu Beginn)")
    fig_perf = go.Figure()
    for j, res in enumerate(results):
        fig_perf.add_trace(go.Scatter(
            x=res.index.index, y=res.index.values,
            mode="lines", name=res.name,
            line=dict(color=PALETTE[j % 3]),
            hovertemplate="%{x|%b %Y}<br>%{y:.1f}<extra>" + res.name + "</extra>",
        ))
    fig_perf.update_layout(
        height=450, hovermode="x unified",
        yaxis_title="Portfoliowert (Index)", xaxis_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_perf, use_container_width=True)

    # Chart 2 — Drawdowns
    st.subheader("📉 Drawdown-Verlauf")
    fig_dd = go.Figure()
    for j, res in enumerate(results):
        dd = drawdown_series(res.index)
        fig_dd.add_trace(go.Scatter(
            x=dd.index, y=dd.values * 100,
            mode="lines", name=res.name, fill="tozeroy",
            line=dict(color=PALETTE[j % 3]),
            hovertemplate="%{x|%b %Y}<br>%{y:.2f}%<extra>" + res.name + "</extra>",
        ))
    fig_dd.update_layout(
        height=350, hovermode="x unified", yaxis_title="Drawdown (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    # Chart 3 — Jahresrenditen
    st.subheader("📅 Jahresrenditen")
    annual_df = pd.concat(
        [res.annual_returns.to_frame(name=res.name) for res in results], axis=1
    )
    fig_yrs = go.Figure()
    for j, col in enumerate(annual_df.columns):
        fig_yrs.add_trace(go.Bar(
            x=annual_df.index, y=annual_df[col] * 100, name=col,
            marker_color=PALETTE[j % 3],
            hovertemplate="%{x}<br>%{y:.2f}%<extra>" + col + "</extra>",
        ))
    fig_yrs.update_layout(
        height=400, barmode="group", yaxis_title="Rendite (%)", xaxis_title="Jahr",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_yrs, use_container_width=True)

    # Per-year tables
    st.subheader("🧮 Jahreskennzahlen je Portfolio")
    for res in results:
        with st.expander(f"📂 {res.name}  ·  "
                         f"Aktien {res.weights['Equity']:.0%} / "
                         f"Renten {res.weights['Bonds']:.0%} / "
                         f"Gold {res.weights['Gold']:.0%} / "
                         f"Cash {res.weights['Cash']:.0%}  ·  "
                         f"Rebal: {res.rebalance}"):
            yr = per_year_metrics(res, dd_mode=dd_mode).rename(columns={
                "Return": "Rendite", "Volatility p.a.": "Volatilität p.a.",
            })
            st.dataframe(
                yr.style.format({
                    "Rendite": "{:.2%}", "Volatilität p.a.": "{:.2%}",
                    "Max Drawdown": "{:.2%}",
                }),
                use_container_width=True,
            )

    # Export
    st.subheader("💾 Export")
    exp_cols = st.columns(len(results) + 1)
    with exp_cols[0]:
        st.download_button(
            "📥 Zusammenfassung (CSV)",
            data=summary_df.to_csv().encode("utf-8"),
            file_name="portfolio_summary.csv", mime="text/csv",
        )
    for i, res in enumerate(results, start=1):
        with exp_cols[i]:
            st.download_button(
                f"📥 {res.name} – Jahre (CSV)",
                data=per_year_metrics(res, dd_mode=dd_mode).to_csv().encode("utf-8"),
                file_name=f"{res.name.replace(' ', '_')}_yearly.csv",
                mime="text/csv",
            )

# --------------------------------------------------------------------------
# TAB 2 — rolling-window backtest with fan chart
# --------------------------------------------------------------------------
with tab_roll:
    st.caption(
        "Statt eines einzigen Zeitraums werden viele überlappende Fenster "
        "fester Länge ausgewertet. Das Ergebnis ist eine **Verteilung** von "
        "Rendite- und Risikokennzahlen — unabhängig vom zufälligen Startpunkt."
    )

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        max_window = max(1, (end_year - start_year))
        window_years = st.slider(
            "Fensterlänge (Jahre)",
            min_value=1, max_value=min(30, max_window),
            value=min(10, max_window), step=1,
            help="Länge jedes rollierenden Backtests.",
        )
    with rc2:
        step_label = st.selectbox(
            "Schrittweite der Fenster",
            options=["M", "Y"],
            format_func=lambda x: {
                "M": "Monatlich (max. Überlappung)",
                "Y": "Jährlich (schneller)",
            }[x],
        )
    with rc3:
        roll_portfolio = st.selectbox(
            "Portfolio für den Fächerchart",
            options=list(range(len(results))),
            format_func=lambda i: results[i].name,
        )

    # Run rolling backtest for every portfolio (metrics) + fan chart for one
    roll_results = []
    error = None
    for cfg in configs:
        try:
            roll_results.append(rolling_backtest(
                returns=returns,
                weights=cfg["weights"],
                window_years=window_years,
                rebalance=cfg["rebalance"],
                step=step_label,
                risk_free_pa=risk_free,
                name=cfg["name"],
            ))
        except ValueError as e:
            error = str(e)
            break

    if error:
        st.error(f"⚠️ {error}")
    else:
        n_windows = len(roll_results[0].metrics)
        st.info(
            f"**{n_windows} überlappende {window_years}-Jahres-Fenster** "
            f"ausgewertet (Zeitraum {start_year}–{end_year}, Schrittweite "
            f"{'monatlich' if step_label == 'M' else 'jährlich'})."
        )

        # ---- Distribution summary table -------------------------------
        st.subheader("📊 Verteilung der Kennzahlen über alle Fenster")
        for roll in roll_results:
            with st.expander(f"📂 {roll.name}", expanded=(len(roll_results) == 1)):
                s = rolling_summary(roll)
                s_fmt = s.copy()
                st.dataframe(
                    s_fmt.style.format({
                        "CAGR": "{:.2%}", "Volatility": "{:.2%}",
                        "Max Drawdown": "{:.2%}", "Sharpe": "{:.2f}",
                        "Calmar": "{:.2f}", "Anzahl Fenster": "{:.0f}",
                    }),
                    use_container_width=True,
                )

        # ---- Fan chart ------------------------------------------------
        roll = roll_results[roll_portfolio]
        st.subheader(f"🪭 Fächerchart der Equity-Kurven – {roll.name}")
        st.caption(
            f"Alle {len(roll.metrics)} normierten {window_years}-Jahres-Pfade "
            f"(Start = 100). Die Bänder zeigen die Perzentile der Pfadverteilung."
        )

        fan_tab1, fan_tab2 = st.tabs([
            "Ab gemeinsamem Start (t = 0)", "Nach Kalenderdatum",
        ])

        # --- Fan chart variant 1: aligned at t=0 ----------------------
        with fan_tab1:
            bands = fan_chart_bands(roll, percentiles=(5, 25, 50, 75, 95))
            months = bands.index / 12.0  # x-axis in years
            fig_fan = go.Figure()
            # 5–95 band
            fig_fan.add_trace(go.Scatter(
                x=months, y=bands["p95"], mode="lines",
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))
            fig_fan.add_trace(go.Scatter(
                x=months, y=bands["p5"], mode="lines", fill="tonexty",
                fillcolor="rgba(31,119,180,0.15)", line=dict(width=0),
                name="5 – 95 % Perzentil",
                hovertemplate="Jahr %{x:.1f}<br>%{y:.1f}<extra>P5</extra>",
            ))
            # 25–75 band
            fig_fan.add_trace(go.Scatter(
                x=months, y=bands["p75"], mode="lines",
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))
            fig_fan.add_trace(go.Scatter(
                x=months, y=bands["p25"], mode="lines", fill="tonexty",
                fillcolor="rgba(31,119,180,0.35)", line=dict(width=0),
                name="25 – 75 % Perzentil",
                hovertemplate="Jahr %{x:.1f}<br>%{y:.1f}<extra>P25</extra>",
            ))
            # median
            fig_fan.add_trace(go.Scatter(
                x=months, y=bands["p50"], mode="lines",
                line=dict(color="#1f77b4", width=2.5), name="Median",
                hovertemplate="Jahr %{x:.1f}<br>%{y:.1f}<extra>Median</extra>",
            ))
            fig_fan.update_layout(
                height=480, hovermode="x unified",
                xaxis_title=f"Jahre seit Anlagebeginn (Fenster = {window_years} J.)",
                yaxis_title="Portfoliowert (Index, Start = 100)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
            )
            st.plotly_chart(fig_fan, use_container_width=True)

            final = roll.paths.iloc[-1]
            fc1, fc2, fc3 = st.columns(3)
            fc1.metric(
                "Endwert – schlechtestes Fenster",
                f"{final.min():.1f}",
                f"{(final.min()/100 - 1):.1%}",
            )
            fc2.metric(
                "Endwert – Median",
                f"{final.median():.1f}",
                f"{(final.median()/100 - 1):.1%}",
            )
            fc3.metric(
                "Endwert – bestes Fenster",
                f"{final.max():.1f}",
                f"{(final.max()/100 - 1):.1%}",
            )

        # --- Fan chart variant 2: by calendar date --------------------
        with fan_tab2:
            fig_cal = go.Figure()
            # Thin out to keep the browser responsive: at most ~80 paths.
            all_cols = list(roll.paths.columns)
            max_lines = 80
            stride = max(1, len(all_cols) // max_lines)
            shown_cols = all_cols[::stride]
            for col in shown_cols:
                start = pd.Period(col, freq="M").to_timestamp()
                dates = pd.date_range(start, periods=len(roll.paths), freq="ME")
                fig_cal.add_trace(go.Scatter(
                    x=dates, y=roll.paths[col].values, mode="lines",
                    line=dict(color="rgba(31,119,180,0.25)", width=1),
                    showlegend=False,
                    hovertemplate="%{x|%b %Y}<br>%{y:.1f}<extra>Start " + col + "</extra>",
                ))
            fig_cal.update_layout(
                height=480,
                xaxis_title="Kalenderdatum",
                yaxis_title="Portfoliowert (Index, Start = 100)",
                title=dict(
                    text=f"Jeder Pfad = ein {window_years}-Jahres-Fenster, "
                         f"verankert an seinem realen Startdatum"
                         + (f"  ·  Anzeige: {len(shown_cols)} von "
                            f"{len(all_cols)} Pfaden" if stride > 1 else ""),
                    font=dict(size=12),
                ),
            )
            st.plotly_chart(fig_cal, use_container_width=True)
            st.caption(
                "Diese Ansicht zeigt, in welchen Marktphasen die Pfade "
                "starteten — gut sichtbar werden günstige vs. ungünstige "
                "Einstiegszeitpunkte."
            )

        # ---- Distribution of CAGR (histogram) -------------------------
        st.subheader(f"📊 Verteilung der Jahresrendite (CAGR) – {roll.name}")
        cagr = roll.metrics["CAGR"] * 100
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=cagr, nbinsx=40, marker_color="#1f77b4",
            hovertemplate="%{x:.1f} %<br>%{y} Fenster<extra></extra>",
        ))
        for q, lbl, col in [
            (0.05, "5 %-Perzentil", "#d62728"),
            (0.50, "Median", "#2ca02c"),
            (0.95, "95 %-Perzentil", "#d62728"),
        ]:
            v = cagr.quantile(q)
            fig_hist.add_vline(
                x=v, line=dict(color=col, dash="dash"),
                annotation_text=f"{lbl}: {v:.1f}%",
                annotation_position="top",
            )
        fig_hist.update_layout(
            height=360, xaxis_title="Annualisierte Rendite (CAGR, %)",
            yaxis_title="Anzahl Fenster", bargap=0.05,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # ---- Export ---------------------------------------------------
        st.subheader("💾 Export")
        rexp_cols = st.columns(len(roll_results))
        for i, roll in enumerate(roll_results):
            with rexp_cols[i]:
                st.download_button(
                    f"📥 {roll.name} – Fenster-Kennzahlen (CSV)",
                    data=roll.metrics.to_csv().encode("utf-8"),
                    file_name=f"{roll.name.replace(' ', '_')}_rolling_{window_years}y.csv",
                    mime="text/csv",
                )

# --------------------------------------------------------------------------
# Methodology + disclaimer (shared, below tabs)
# --------------------------------------------------------------------------
with st.expander("ℹ️ Methodische Hinweise"):
    st.markdown(
        """
**Datenquellen**
- *Aktien*: MSCI World Net Total Return (Bloomberg-Ticker NDDUWI) – in EUR,
  verfügbar ab Dezember 1969.
- *Renten*: REXP – Deutscher Rentenindex (Performance), verfügbar ab 1970.
- *Gold*: Spotpreis (i. d. R. in USD denominiert – siehe Hinweis unten).
- *Liquidität*: Konstante Annahme über `Cash-Rendite p.a.` in der Sidebar.

**Klassische Simulation** (Tab 1) berechnet einen einzigen durchgehenden
Backtest über den in der Sidebar gewählten Zeitraum.

**Rollierender Backtest** (Tab 2) wertet alle überlappenden Fenster fester
Länge aus (z. B. jedes mögliche 10-Jahres-Fenster seit 1970). Daraus ergibt
sich eine **Verteilung** der Kennzahlen, die unabhängig vom zufällig
gewählten Startzeitpunkt ist. Der **Fächerchart** legt alle normierten
Equity-Kurven übereinander; die farbigen Bänder zeigen die Perzentile
(5/25/50/75/95 %) der Pfadverteilung.

**Hinweis zu Gold**
Die hinterlegte Gold-Zeitreihe ist üblicherweise in **USD** denominiert.
Für eine vollständig konsistente EUR-Sicht müsste sie mit dem EURUSD-
Wechselkurs umgerechnet werden (EURUSD-Spalte in "Tabelle2" der Excel).

**Rechenkonventionen**
- Monatsultimo-Daten, Renditen als einfache prozentuale Veränderung.
- Annualisierte Volatilität = Stdabw. der Monatsrenditen × √12.
- Max Drawdown = größter Peak-to-Trough-Verlust.
- Sharpe Ratio = (CAGR − Risk-free) / Vola p.a.
- Calmar Ratio = CAGR / |Max Drawdown|.
- Rebalancing zum Monatsultimo des gewählten Intervalls.

**Steuern & Kosten** werden in dieser Simulation nicht abgebildet.
        """
    )

st.divider()
st.caption(
    "**Rechtliche Hinweise / Disclaimer**  \n"
    "Die in dieser Anwendung dargestellten Informationen, Simulationen und "
    "Auswertungen dienen ausschließlich der allgemeinen Information sowie "
    "Veranschaulichung historischer Kapitalmarktentwicklungen. Sie stellen "
    "**weder eine Anlageberatung noch eine Anlageempfehlung** im Sinne des "
    "Wertpapierhandelsgesetzes (WpHG) oder eine Aufforderung zum Kauf oder "
    "Verkauf von Finanzinstrumenten dar und ersetzen keine individuelle "
    "Beratung durch einen qualifizierten Anlage- oder Steuerberater.  \n\n"
    "**Die frühere Wertentwicklung ist kein verlässlicher Indikator für die "
    "künftige Wertentwicklung.** Berechnungen basieren auf historischen "
    "Indexdaten ohne Berücksichtigung von Steuern, Transaktionskosten, "
    "Produktgebühren (z. B. TER von ETFs/Fonds), Ausgabeaufschlägen oder "
    "Spreads. Eine direkte Investition in einen Index ist nicht möglich; "
    "real verfügbare Anlageprodukte können in ihrer Wertentwicklung von der "
    "Indexentwicklung abweichen (Tracking Error). Wertpapiere und insbesondere "
    "Aktien- und Rohstoffanlagen unterliegen Marktschwankungen, die zu "
    "Kursverlusten bis hin zum **Totalverlust** des eingesetzten Kapitals "
    "führen können. Währungsschwankungen können zusätzliche Risiken bergen.  \n\n"
    "Für die Richtigkeit, Vollständigkeit und Aktualität der zugrunde "
    "liegenden Daten wird keine Gewähr übernommen."
)
