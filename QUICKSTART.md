# Quickstart Guide

This guide gets the project from local files to a loaded MySQL database, trained scikit-learn model, and exported report.

## Prerequisites

- Python 3.10 or newer
- MySQL 8.x, or Docker Desktop if you want to use the included MySQL container
- PowerShell on Windows

## 1. Install Python Dependencies

From the project root:

```powershell
python -m pip install -r requirements.txt
```

The dependencies include:

- `mysql-connector-python` for importing CSV files into MySQL
- `scikit-learn` and `joblib` for model training and prediction

## 2. Start or Choose a MySQL Server

### Option A: Use Docker

Start the included MySQL service:

```powershell
docker compose up -d
```

This starts MySQL on `127.0.0.1:3306` with:

```text
MYSQL_USER=root
MYSQL_PASSWORD=change-me
MYSQL_DATABASE=oil_news_project
```

### Option B: Use an Existing MySQL Server

Use your own MySQL host, port, username, password, and target database name.

## 3. Configure MySQL Connection

Create a local `.env` file from the example:

```powershell
Copy-Item .env.example .env
```

Edit `.env`:

```text
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=change-me
MYSQL_DATABASE=oil_news_project
```

Use your actual password if you are connecting to an existing MySQL server.

## 4. Import the CSV Datasets into MySQL

Run:

```powershell
python src\load_mysql.py --replace
```

What this does:

- Creates the database if it does not exist
- Creates one MySQL table per CSV file in `datasets/`
- Infers column types from `datasets/data_dictionary.csv`
- Loads all CSV rows in batches
- Applies the analysis views from `sql/analytics_views.sql`

Use `--replace` when you want to truncate and reload existing tables.

To load only specific tables:

```powershell
python src\load_mysql.py --replace --only ops_market_daily ops_gpr_daily ops_events
```

## 5. Verify the Database Import

Open MySQL and run:

```sql
USE oil_news_project;

SHOW TABLES;

SELECT COUNT(*) AS market_rows
FROM ops_market_daily;

SELECT market_date, brent_price_usd, gpr_index, event_type
FROM vw_daily_oil_news_features
ORDER BY market_date DESC
LIMIT 10;
```

Expected result: `ops_market_daily` should contain 4,047 rows.

Useful views:

- `vw_daily_oil_news_features`
- `vw_event_price_reaction`
- `vw_country_petrol_impact`

## 6. Train the Predictive Oil Model

Run:

```powershell
python src\train_oil_model.py
```

The training script fits:

```text
StandardScaler -> Ridge Regression
```

Generated files:

- `model_artifacts/oil_price_model.joblib`
- `model_artifacts/oil_price_model.json`
- `model_artifacts/test_predictions.csv`

Current model benchmark:

- Test RMSE: 1.516 USD
- Test MAE: 1.118 USD
- Previous-price baseline RMSE: 1.536 USD

## 7. Run a Prediction

Run:

```powershell
python src\predict_oil_price.py
```

Current bundled latest-row example:

```text
Input market date: 2026-03-12
Current Brent: $95.76
Predicted next trading-day Brent: $95.90
```

## 8. Open the Report and Schema

Project report:

- Markdown: `reports/oil_news_database_model_report.md`
- PDF: `output/pdf/oil_news_database_model_report.pdf`

Database schema:

- `docs/database_schema.md`

Regenerate the PDF report:

```powershell
python scripts\build_report_pdf.py
```

## Troubleshooting

If MySQL connection fails:

- Confirm the MySQL server is running.
- Confirm `.env` has the correct host, port, user, password, and database.
- If using Docker, run `docker compose ps`.

If imports fail because tables already contain data:

```powershell
python src\load_mysql.py --replace
```

If Python cannot import packages:

```powershell
python -m pip install -r requirements.txt
```

If Docker is not available, install MySQL locally and use Option B.

