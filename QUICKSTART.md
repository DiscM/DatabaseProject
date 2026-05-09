# Quickstart Guide

Get from local files to a trained model and interactive forecast charts in
under a minute — no MySQL required for the core pipeline.

## Prerequisites

- Python 3.10 or newer
- Docker Desktop (optional — only needed for MySQL integration)
- PowerShell on Windows

---

## Path A — No MySQL (recommended for demos)

The fastest path. Trains the model, generates a Monte Carlo forecast, and
opens interactive charts without touching a database.

### 1. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

### 2. Run the Pipeline

```powershell
python main.py
```

That's it. The script runs three steps automatically:

| Step | What happens |
|------|-------------|
| **1 — Train** | Fits 10 Ridge models (one per horizon, h=1..10) on `ops_market_daily.csv` |
| **2 — Predict** | Runs 500 Monte Carlo paths → 10-day forecast with P10/P25/P75/P90 bands |
| **3 — Visualize** | Opens two interactive Plotly charts in the browser |

Total runtime: ~2–3 seconds.

---

## Path B — Full Pipeline with MySQL

Includes importing all CSV datasets into MySQL before training.

### 1. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

### 2. Configure MySQL Connection (required)

Copy the example env file — this is **required** because the Docker container
uses a non-empty root password that must be provided to the Python scripts:

```powershell
Copy-Item .env.example .env
```

```text
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=change-me
MYSQL_DATABASE=oil_news_project
```

The defaults above match `docker-compose.yml` exactly — no edits needed if
you are using the included Docker container.

### 3. Start the Docker MySQL Container

```powershell
docker compose up -d
```

### 4. Run the Full Pipeline

```powershell
python main_with_db.py
```

`main_with_db.py` automatically polls MySQL until the container has finished
initialising — you do not need to wait or sleep manually.

| Step | What happens |
|------|-------------|
| **0 — Wait** | Polls TCP port then login until MySQL is ready (up to 90 s) |
| **1 — Load** | Upserts all CSVs in `datasets/` into MySQL + applies analytics views |
| **2 — Train** | Same 10-horizon Ridge training as Path A |
| **3 — Predict** | Same Monte Carlo forecast as Path A |
| **4 — Visualize** | Same interactive charts as Path A |

---

## Running Individual Steps

If you prefer to run steps manually:

```powershell
# Load CSVs into MySQL (requires MySQL to be running)
python src\load_mysql.py --replace

# Load only specific tables
python src\load_mysql.py --replace --only ops_market_daily ops_gpr_daily ops_events

# Train the model
python src\train_oil_model.py

# Generate the Monte Carlo forecast
python src\predict_oil_price.py

# Customise the forecast (more paths, stronger momentum)
python src\predict_oil_price.py --n-sims 1000 --momentum-blend 0.6 --momentum-window 5

# Open the visualizations
python Visualization\visualize_predictions.py

# Regenerate the demo notebook from source files
python compile_notebook.py
```

---

## Verify the Database Import (Path B only)

After running `main_with_db.py` (or `load_mysql.py`), confirm the data loaded:

```sql
USE oil_news_project;

SHOW TABLES;

SELECT COUNT(*) AS market_rows FROM ops_market_daily;
-- Expected: 4,047 rows

SELECT market_date, brent_price_usd, gpr_index, event_type
FROM vw_daily_oil_news_features
ORDER BY market_date DESC
LIMIT 10;
```

Useful views created automatically:

- `vw_daily_oil_news_features` — daily oil, market, event, and GPR features
- `vw_event_price_reaction` — event dates with same-day market indicators
- `vw_country_petrol_impact` — country-level exposure and petrol price data

---

## Demo Notebook

Open `oil_news_project_demo.ipynb` in Jupyter for a cell-by-cell walkthrough
of the entire pipeline. Regenerate it any time from the current source files:

```powershell
python compile_notebook.py
```

---

## Troubleshooting

**MySQL connection fails**
- Confirm the container is running: `docker compose ps`
- Confirm `.env` credentials match `docker-compose.yml`
- Check container logs: `docker compose logs mysql`

**Tables already contain data**
```powershell
python src\load_mysql.py --replace
```

**Missing Python packages**
```powershell
python -m pip install -r requirements.txt
```

**Charts show static images instead of interactive Plotly**
- Run the Setup cell in the notebook first (sets `pio.renderers.default = "notebook"`)
- Or run the visualization standalone: `python Visualization\visualize_predictions.py`
