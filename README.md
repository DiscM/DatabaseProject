# Semantic News and Oil Price Database Project

This project compiles workspace datasets into a MySQL database and trains a
scikit-learn predictive model for Brent crude oil prices, including a
Monte Carlo stochastic 10-day forward forecast with momentum blending.

For the shortest setup path, start with [QUICKSTART.md](QUICKSTART.md).

The data combines:

- Geopolitical risk and semantic news signals (`ops_gpr_daily`, `ops_gpr_monthly`, `ops_events`)
- Oil and market prices (`ops_market_daily`, `ops_crude_oil_daily`)
- Country-level petrol exposure and impacts (`ops_countries`, `ops_country_impact`, `ops_petrol_price_snapshots`)
- Warehouse-ready dimensions and facts (`dim_*`, `fact_*`)

## Database Schema

The project keeps two complementary database layers: operational source-style tables for modeling and exploration, plus dimensional tables for BI-style reporting. Full column-level ERDs are available in [docs/database_schema.md](docs/database_schema.md).

### Operational ERD

The operational layer preserves the source-style CSV tables used for exploration, joins, and model feature engineering.

```mermaid
erDiagram
    ops_countries ||--o{ ops_country_impact : country_id
    ops_countries ||--o{ ops_petrol_price_snapshots : country_id
    ops_countries ||--o{ ops_gpr_country_monthly : iso3
    ops_market_daily ||--o| ops_gpr_daily : "market_date = gpr_date"
    ops_market_daily ||--o| ops_events : "market_date = event_date"
    ops_market_daily ||--o| ops_crude_oil_daily : "market_date = trade_date"
```

### Dimensional Reporting ERD

The dimensional layer organizes dates, countries, events, market facts, GPR facts, and petrol impact facts for BI-style reporting.

```mermaid
erDiagram
    dim_date ||--o{ dim_event : date_key
    dim_date ||--o{ fact_market_daily : date_key
    dim_date ||--o{ fact_gpr_daily : date_key
    dim_date ||--o{ fact_gpr_monthly : date_key
    dim_date ||--o{ fact_petrol_prices : date_key
    dim_country ||--o{ fact_country_impact : country_key
    dim_country ||--o{ fact_petrol_prices : country_key
```

### Analysis Views

After loading the CSV tables, `src/load_mysql.py` applies `sql/analytics_views.sql` to create analysis-ready joins:

- `vw_daily_oil_news_features`: daily Brent/WTI, GPR, event, and volatility features for modeling.
- `vw_event_price_reaction`: event dates paired with same-day oil price and risk indicators.
- `vw_country_petrol_impact`: country exposure, petrol price snapshots, and vulnerability indicators.

## Project Layout

```text
main.py                    One-command pipeline runner (no MySQL required)
main_with_db.py            One-command pipeline runner WITH MySQL integration
src/db_config.py           MySQL connection config (reads .env)
src/load_mysql.py          CSV-to-MySQL loader with inferred table schemas
src/train_oil_model.py     Ridge regression training (10 horizons, h=1..10)
src/predict_oil_price.py   Monte Carlo stochastic 10-day forecast
Visualization/
  visualize_predictions.py Interactive Plotly charts (scatter + forecast fan)
compile_notebook.py        Regenerates oil_news_project_demo.ipynb from sources
datasets/                  Source CSV files
docs/database_schema.md    Full column-level operational and dimensional ERDs
sql/analytics_views.sql    MySQL views for analysis-ready joins
model_artifacts/           Trained models, test predictions, forecast CSV
reports/                   Markdown project report
output/pdf/                Exported PDF report
docker-compose.yml         Local MySQL 8.4 service
```

## Quick Run (No MySQL)

Run the full train → predict → visualize pipeline from a single command:

```powershell
python main.py
```

This trains the model, generates a Monte Carlo forecast, and opens two
interactive Plotly charts in the browser. MySQL is **not** required.

## Quick Run (With MySQL)

Start the Docker container, then run the full pipeline including CSV import:

```powershell
docker compose up -d
python main_with_db.py
```

`main_with_db.py` automatically waits for MySQL to finish initialising before
proceeding — no manual sleep or retry is needed.

## Manual Step-by-Step

### 1. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

### 2. Start MySQL (optional — skip if not using the database)

```powershell
docker compose up -d
```

### 3. Load Datasets into MySQL

```powershell
python src\load_mysql.py --replace
```

The loader creates the database, one table per CSV, infers column types from
`datasets/data_dictionary.csv`, and applies `sql/analytics_views.sql`.

Example query after loading:

```sql
SELECT market_date, brent_price_usd, gpr_index, event_type, event_severity
FROM vw_daily_oil_news_features
WHERE event_flag = 1
ORDER BY market_date DESC
LIMIT 20;
```

### 4. Train the Model

```powershell
python src\train_oil_model.py
```

Trains one `StandardScaler → Ridge` model per horizon (h=1 through h=10).
Each model predicts directly N days ahead from today's real data (direct
multi-step strategy — no iterative error accumulation).

Artifacts saved to `model_artifacts/`:

- `oil_price_model.joblib` (h=1, primary)
- `oil_price_model_h{N}.joblib` for N = 2..10
- `oil_price_model.json` (metrics + per-horizon RMSE)
- `test_predictions.csv`

Current benchmark (h=1):

| Metric | Model | Baseline (naive persistence) |
|--------|-------|------------------------------|
| RMSE   | 1.52 USD | 1.54 USD |
| MAE    | 1.12 USD | 1.11 USD |
| R²     | 0.967 | — |

### 5. Generate the Forecast

```powershell
python src\predict_oil_price.py
```

Runs 500 Monte Carlo simulation paths with Gaussian noise (sigma = h=1 RMSE)
injected at each step. A linear momentum blend (40% weight over the last 10
trading days) corrects for Ridge's mean-reversion bias.

Output: `model_artifacts/forward_forecast.csv` — median + P10/P25/P75/P90
for each of the next 10 trading days.

CLI options:

```powershell
python src\predict_oil_price.py --n-sims 1000 --momentum-blend 0.5 --momentum-window 5
```

### 6. Visualize

```powershell
python Visualization\visualize_predictions.py
```

Opens two interactive Plotly charts in the browser:

1. **Prediction accuracy scatter** — actual vs predicted on the test set
2. **Model vs Baseline vs Actual** — time-series with the 10-day Monte Carlo
   probability fan (P10–P90 bands) bridged from the last real data point

### 7. Regenerate the Demo Notebook

```powershell
python compile_notebook.py
```

Rebuilds `oil_news_project_demo.ipynb` by reading the current source files
verbatim. Open and run it top-to-bottom in Jupyter for an end-to-end demo.
