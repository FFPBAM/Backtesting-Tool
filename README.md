# 📈 Historische Portfolio-Simulation

Streamlit-Tool zur Simulation eines Mischportfolios aus **MSCI World (EUR)**,
**REXP** und **Liquidität** ab dem Jahr 2000.
Die Quoten, der Zeitraum, das Rebalancing-Intervall und die Cash-Verzinsung
sind interaktiv einstellbar. Bis zu drei Portfolios lassen sich
nebeneinander vergleichen.

## Features

- ✅ Konfigurierbare Asset-Allocation (Aktien / Renten / Liquidität)
- ✅ Variabler Zeitraum (Start- und Endjahr)
- ✅ Wählbares Rebalancing-Intervall (M / Q / Y / nie)
- ✅ Performance p.a. je Kalenderjahr + Gesamtperiode (CAGR)
- ✅ Volatilität p.a., Max Drawdown (innerhalb Jahres oder rollierend 1Y)
- ✅ Sharpe Ratio und Calmar Ratio
- ✅ Vergleich von bis zu **3 Portfolios** nebeneinander
- ✅ Interaktive Plotly-Charts: Wertentwicklung, Drawdown-Verlauf, Jahresrenditen
- ✅ CSV-Export aller Kennzahlen

## Lokale Ausführung

```bash
git clone <repo-url>
cd portfolio_simulator
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Die App ist anschließend unter http://localhost:8501 erreichbar.

## Deployment auf Streamlit Community Cloud

1. Repository auf GitHub pushen (inkl. `data/Daten_Verrentung_EUR.xlsx`).
2. Auf https://share.streamlit.io → "New app" → Repo, Branch und
   `app.py` auswählen.
3. Deploy. Fertig.

> ⚠️ Hinweis: Die Excel-Datei mit den Bloomberg-Daten wird mit dem Repo
> deployed. Falls die Daten urheberrechtlich geschützt sind oder du sie
> aktuell halten möchtest, kannst du die App stattdessen mit einem
> privaten Repo deployen oder die Datei dynamisch von einem geschützten
> Speicher (z. B. S3 mit Streamlit-Secrets) laden.

## Projektstruktur

```
portfolio_simulator/
├── app.py                # Streamlit-UI
├── data_loader.py        # Excel-Einlesen & Aufbereitung
├── simulation.py         # Portfolio-Engine + Risikokennzahlen
├── requirements.txt
├── README.md
└── data/
    └── Daten_Verrentung_EUR.xlsx
```

## Methodische Hinweise

### Datenquellen

| Anlageklasse | Ticker | Bezeichnung | Währung |
|-------------|--------|-------------|---------|
| Aktien      | NDDUWI | MSCI World Net Total Return | EUR |
| Renten      | REXP   | Deutscher Rentenindex Performance | EUR |
| Liquidität  | –      | Konstante Cash-Verzinsung (Sidebar) | EUR |

### Cash-Approximation

Da der Datensatz keine Euribor-Zeitreihe enthält, wird Liquidität durch
eine **konstante annualisierte Verzinsung** dargestellt, die in der
Sidebar gesetzt wird. Default: 2 % p.a. (langfristiger Mittelwert).

#### Echte Euribor-Reihe ergänzen

Möchtest du die historische Euribor-Verzinsung exakt abbilden, ergänze in
der Excel-Datei eine Spalte `EURIBOR 1M` (Monats-Ultimo-Yield, % p.a.)
und passe `data_loader.py` und `simulation.build_returns` an, sodass der
Monatsertrag aus dem gleitenden Yield abgeleitet wird:

```python
# in build_returns:
rets["Cash"] = (1 + prices["EURIBOR 1M"] / 100) ** (1/12) - 1
```

### Rechenkonventionen

| Größe | Formel |
|-------|--------|
| Monatsrendite | `P_t / P_{t-1} − 1` |
| Jahresrendite | `Π(1 + r_m) − 1` über die 12 Monate |
| CAGR (p.a.) | `Π(1 + r_m)^(1/Jahre) − 1` |
| Volatilität p.a. | `σ(r_m) · √12` |
| Max Drawdown | `min(P_t / cummax(P_t) − 1)` |
| Sharpe Ratio | `(CAGR − r_f) / σ_pa` |
| Calmar Ratio | `CAGR / |MaxDD|` |

### Rebalancing

- **Monatlich (M):** Default — Allokation wird jeden Monatsultimo wieder
  auf Zielgewichte gesetzt.
- **Quartalsweise (Q):** März, Juni, September, Dezember.
- **Jährlich (Y):** Dezember.
- **Nie:** Buy & Hold; Gewichte driften mit der Marktentwicklung.

### Was die Simulation **nicht** abbildet

- Steuern (Abgeltungsteuer, Soli, Vorabpauschale)
- Transaktions- und Verwaltungskosten / TER
- Bid/Ask-Spreads beim Rebalancing
- Tracking Error eines konkreten ETFs vs. dem Index
- Währungseffekte über die in NDDUWI bereits eingerechnete EUR-Konvertierung hinaus

## Lizenz

Privat / intern. Datenrechte liegen bei Bloomberg bzw. den jeweiligen
Indexanbietern (MSCI, Deutsche Börse).
