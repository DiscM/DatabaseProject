# Semantic News and Oil Price Database Project Report

**Prepared:** May 6, 2026  
**Workspace:** `DatabaseProject`  
**Database target:** MySQL  
**Modeling framework:** scikit-learn

## Executive Summary

This project combines semantic/geopolitical news data with oil market pricing data to support both relational analysis and predictive oil price modeling. The workspace datasets were organized into a MySQL-ready schema with operational source tables, dimensional reporting tables, and analysis views. A scikit-learn model was trained to predict the next trading-day Brent crude oil price using market, lag, volatility, event, and geopolitical risk features.

The final model uses a `StandardScaler -> Ridge Regression` pipeline. On a chronological test split covering December 22, 2022 through March 12, 2026, the model achieved a test RMSE of **1.516 USD** and MAE of **1.118 USD**. The previous-price baseline RMSE was **1.536 USD**, so the fitted model performs slightly better than a strong short-horizon benchmark.

## Project Objectives

- Compile all supplied CSV datasets into a MySQL database.
- Preserve both operational data and dimensional/fact reporting structures.
- Create analysis-ready SQL views for daily oil/news features, event reaction analysis, and country petrol impact analysis.
- Train and export a predictive model for next trading-day Brent oil pricing.
- Produce reusable artifacts for loading, modeling, prediction, and reporting.

## Data Sources and Coverage

The workspace contains 20 CSV files under `datasets/`. The most important modeling and database tables are:

| Dataset | Rows | Purpose |
|---|---:|---|
| `ops_market_daily.csv` | 4,047 | Daily Brent/WTI prices, macro-market indicators, lag features, volatility, and event flags |
| `ops_gpr_daily.csv` | 15,078 | Daily geopolitical risk/news indices and article counts |
| `ops_gpr_monthly.csv` | 1,515 | Monthly geopolitical risk indicators |
| `ops_gpr_country_monthly.csv` | 66,660 | Country-month geopolitical risk measures |
| `ops_events.csv` | 55 | Named geopolitical, war, sanctions, disaster, and market events |
| `ops_countries.csv` | 18 | Country dimension for impact and petrol price analysis |
| `ops_petrol_price_snapshots.csv` | 28 | Country petrol price snapshots and price change indicators |
| `dim_*` / `fact_*` files | mixed | Warehouse-ready star schema tables |

The main modeling table, `ops_market_daily`, spans **February 17, 2010 through March 12, 2026**. Daily GPR data spans **January 1, 1985 through April 13, 2026**.

## Database Design

The database design uses two complementary layers.

### Operational Layer

The operational layer directly represents the supplied source-style datasets. It is useful for model training, exploration, and tracing records back to the original CSVs.

Key tables:

- `ops_market_daily`: primary daily modeling table with oil prices, lagged oil prices, market indicators, volatility metrics, and event fields.
- `ops_gpr_daily`: daily semantic/geopolitical news risk features.
- `ops_events`: named events with event type, category, description, date, and severity.
- `ops_countries`, `ops_country_impact`, and `ops_petrol_price_snapshots`: country exposure and petrol price impact analysis.
- `ops_gpr_country_monthly`: country-month geopolitical risk history.

Important joins:

- `ops_market_daily.market_date = ops_gpr_daily.gpr_date`
- `ops_market_daily.market_date = ops_events.event_date`
- `ops_market_daily.market_date = ops_crude_oil_daily.trade_date`
- `ops_countries.country_id = ops_country_impact.country_id`
- `ops_countries.country_id = ops_petrol_price_snapshots.country_id`
- `ops_countries.iso3 = ops_gpr_country_monthly.iso3`

### Dimensional Layer

The dimensional layer supports BI-style reporting with reusable dimensions and facts.

Key dimensions:

- `dim_date`
- `dim_country`
- `dim_event`

Key facts:

- `fact_market_daily`
- `fact_gpr_daily`
- `fact_gpr_monthly`
- `fact_country_impact`
- `fact_petrol_prices`

This layout enables date-based time-series reporting, country impact comparison, and event-linked price analysis.

## SQL Views

The project creates three analysis views in `sql/analytics_views.sql`.

| View | Description |
|---|---|
| `vw_daily_oil_news_features` | Daily joined Brent/WTI, GPR, event, and volatility features for analysis and modeling |
| `vw_event_price_reaction` | Event-date oil price and risk indicators for event study analysis |
| `vw_country_petrol_impact` | Country exposure, petrol price snapshots, and vulnerability indicators |

Example analysis query:

```sql
SELECT market_date, brent_price_usd, gpr_index, event_type, event_severity
FROM vw_daily_oil_news_features
WHERE event_flag = 1
ORDER BY market_date DESC
LIMIT 20;
```

## MySQL Loading Workflow

The MySQL loader is implemented in `src/load_mysql.py`. It:

- Reads all CSV files from `datasets/`.
- Uses `datasets/data_dictionary.csv` to infer MySQL column types.
- Creates one table per CSV file.
- Batches inserts for efficient loading.
- Applies the SQL analysis views after loading.

Recommended load command:

```powershell
python src\load_mysql.py --replace
```

The optional `docker-compose.yml` starts a local MySQL 8.4 container for the project.

## Predictive Modeling Approach

The predictive target is:

**Next trading-day Brent crude oil price in USD**

The training script is `src/train_oil_model.py`. It uses a chronological train/test split, which is appropriate for time-series style financial data because future rows are not allowed to influence past training rows.

Model pipeline:

```text
StandardScaler -> Ridge Regression
```

Ridge regression was selected because it is transparent, fast, reproducible, and suitable for a database project showcase. Standardization helps keep the regularization behavior consistent across features with different units.

## Model Features

The model uses 20 features from `ops_market_daily.csv`:

| Feature Group | Columns |
|---|---|
| Current oil prices | `brent_price_usd`, `wti_price_usd` |
| Market indicators | `dxy_index`, `vix_index` |
| News/risk indicator | `gpr_index` |
| Returns | `brent_return`, `wti_return` |
| Lagged prices | `brent_lag_1`, `brent_lag_3`, `brent_lag_7`, `wti_lag_1`, `wti_lag_3`, `wti_lag_7` |
| Volatility | `brent_volatility_7d`, `brent_volatility_30d`, `wti_volatility_7d`, `wti_volatility_30d` |
| Spread | `brent_wti_spread` |
| Event features | `event_severity`, `event_flag` |

## Model Results

Training metadata:

| Item | Value |
|---|---:|
| Training rows | 3,236 |
| Test rows | 810 |
| Training date range | 2010-02-18 to 2022-12-21 |
| Test date range | 2022-12-22 to 2026-03-12 |
| Ridge alpha | 0.1 |

Performance:

| Metric | Training | Test | Previous-Price Baseline |
|---|---:|---:|---:|
| MAE | 1.080 | 1.118 | 1.113 |
| RMSE | 1.567 | 1.516 | 1.536 |
| MAPE | 1.515% | 1.463% | 1.457% |
| R-squared | 0.996 | 0.967 | 0.966 |

The model narrowly improves RMSE versus the previous-price baseline. This is a realistic outcome for next-day oil price prediction, where yesterday's price is usually a very strong benchmark. The model's value is that it gives a reproducible framework for incorporating market, event, and semantic news features while remaining explainable.

Latest example prediction:

| Input Date | Current Brent | Predicted Next Trading-Day Brent |
|---|---:|---:|
| 2026-03-12 | 95.76 USD | 95.90 USD |

## Conclusion

The completed project demonstrates a full data workflow: source CSVs are transformed into a MySQL-ready analytical database, semantic news and geopolitical risk signals are joined with oil market features, and a scikit-learn predictive model is trained and exported. The result is suitable for a database project presentation because it includes schema design, SQL loading, analysis views, reproducible modeling, and report-ready artifacts.
