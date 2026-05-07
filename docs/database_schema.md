# Database Schema

The project keeps both the operational source-style tables and the dimensional/fact tables supplied in the workspace. The operational layer is best for modeling and exploratory analysis; the dimensional layer is useful for BI-style reporting.

## Core Operational ERD

```mermaid
erDiagram
    ops_countries {
        int country_id PK
        varchar country_name
        varchar iso3
        varchar region_name
        varchar currency_code
        varchar oil_import_dependency
    }

    ops_country_impact {
        int country_impact_id PK
        int country_id FK
        int oil_import_pct
        double gdp_impact_pct
        varchar inflation_risk
        double stock_market_change_pct
        varchar currency_pressure
        varchar policy_response
        varchar vulnerability
        int population_millions
    }

    ops_petrol_price_snapshots {
        int price_snapshot_id PK
        int country_id FK
        varchar snapshot_stage
        date snapshot_date
        double price_local_per_liter
        double price_usd_per_liter
        varchar currency_code
        varchar unit
        double amount_change_local
        double pct_increase
        varchar trend
    }

    ops_market_daily {
        int market_day_id PK
        date market_date
        double brent_price_usd
        double wti_price_usd
        double dxy_index
        double vix_index
        double gpr_index
        double brent_return
        double wti_return
        double brent_lag_1
        double brent_lag_3
        double brent_lag_7
        double wti_lag_1
        double wti_lag_3
        double wti_lag_7
        double brent_volatility_7d
        double brent_volatility_30d
        double wti_volatility_7d
        double wti_volatility_30d
        double brent_wti_spread
        varchar event_type
        text event_description
        double event_severity
        int event_flag
    }

    ops_gpr_daily {
        int gpr_daily_id PK
        date gpr_date
        int article_count_n10d
        double gpr_daily_index
        double gpr_daily_acts_index
        double gpr_daily_threat_index
        double gpr_ma30
        double gpr_ma7
        text annotated_event
    }

    ops_gpr_monthly {
        int gpr_monthly_id PK
        date month_start
        double gpr_recent_index
        double gpr_recent_threat_index
        double gpr_recent_act_index
        double gpr_historical_index
        double gpr_historical_threat_index
        double gpr_historical_act_index
        double share_gpr
        int article_count_n10
        double share_gprh
        int article_count_n3h
    }

    ops_gpr_country_monthly {
        int gpr_country_monthly_id PK
        date month_start
        varchar iso3
        double gpr_country_recent
        double gpr_country_historical
    }

    ops_events {
        int event_id PK
        date event_date
        varchar event_name
        varchar event_type
        varchar event_category
        text event_description
        varchar event_location
        double event_severity
        varchar source_dataset
    }

    ops_crude_oil_daily {
        int crude_day_id PK
        date trade_date
        double brent_usd_per_barrel
        double wti_usd_per_barrel
        double brent_change_pct
        double wti_change_pct
        varchar conflict_phase
        varchar strait_hormuz_status
    }

    ops_countries ||--o{ ops_country_impact : "country_id"
    ops_countries ||--o{ ops_petrol_price_snapshots : "country_id"
    ops_countries ||--o{ ops_gpr_country_monthly : "iso3"
    ops_market_daily ||--o| ops_gpr_daily : "market_date = gpr_date"
    ops_market_daily ||--o| ops_events : "market_date = event_date"
    ops_market_daily ||--o| ops_crude_oil_daily : "market_date = trade_date"
```

## Dimensional Reporting ERD

```mermaid
erDiagram
    dim_date {
        int date_key PK
        date full_date
        int year
        int quarter
        int month
        varchar month_name
        int day
        varchar day_of_week
        int is_month_start
    }

    dim_country {
        int country_key PK
        varchar country_name
        varchar iso3
        varchar region_name
        varchar currency_code
        varchar oil_import_dependency
    }

    dim_event {
        int event_id PK
        int date_key FK
        date event_date
        varchar event_name
        varchar event_type
        varchar event_category
        text event_description
        varchar event_location
        double event_severity
        varchar source_dataset
    }

    fact_market_daily {
        int market_day_id PK
        int date_key FK
        double brent_price_usd
        double wti_price_usd
        double dxy_index
        double vix_index
        double gpr_index
        double brent_return
        double wti_return
        double brent_volatility_7d
        double brent_volatility_30d
        double wti_volatility_7d
        double wti_volatility_30d
        double brent_wti_spread
        int event_flag
        double event_severity
        varchar event_type
        text event_description
    }

    fact_gpr_daily {
        int gpr_daily_id PK
        int date_key FK
        int article_count_n10d
        double gpr_daily_index
        double gpr_daily_acts_index
        double gpr_daily_threat_index
        double gpr_ma30
        double gpr_ma7
        text annotated_event
    }

    fact_gpr_monthly {
        int gpr_monthly_id PK
        int date_key FK
        double gpr_recent_index
        double gpr_recent_threat_index
        double gpr_recent_act_index
        double gpr_historical_index
        double gpr_historical_threat_index
        double gpr_historical_act_index
        double share_gpr
        int article_count_n10
        double share_gprh
        int article_count_n3h
    }

    fact_country_impact {
        int country_impact_id PK
        int country_key FK
        int oil_import_pct
        double gdp_impact_pct
        varchar inflation_risk
        double stock_market_change_pct
        varchar currency_pressure
        varchar policy_response
        varchar vulnerability
        int population_millions
    }

    fact_petrol_prices {
        int price_snapshot_id PK
        int country_key FK
        int date_key FK
        varchar snapshot_stage
        double price_local_per_liter
        double price_usd_per_liter
        double amount_change_local
        double pct_increase
        varchar trend
    }

    dim_date ||--o{ dim_event : "date_key"
    dim_date ||--o{ fact_market_daily : "date_key"
    dim_date ||--o{ fact_gpr_daily : "date_key"
    dim_date ||--o{ fact_gpr_monthly : "date_key"
    dim_date ||--o{ fact_petrol_prices : "date_key"
    dim_country ||--o{ fact_country_impact : "country_key"
    dim_country ||--o{ fact_petrol_prices : "country_key"
```

## Analysis Views

The loader also applies these MySQL views from `sql/analytics_views.sql`:

- `vw_daily_oil_news_features`: modeling-ready daily Brent/WTI, GPR, event, and volatility features.
- `vw_event_price_reaction`: event-date oil price and risk indicators.
- `vw_country_petrol_impact`: country exposure, petrol price snapshots, and vulnerability indicators.

