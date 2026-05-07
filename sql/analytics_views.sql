USE oil_news_project;

CREATE OR REPLACE VIEW vw_daily_oil_news_features AS
SELECT
    m.market_day_id,
    m.market_date,
    m.brent_price_usd,
    m.wti_price_usd,
    m.dxy_index,
    m.vix_index,
    m.gpr_index,
    g.gpr_daily_acts_index,
    g.gpr_daily_threat_index,
    g.article_count_n10d,
    m.brent_return,
    m.wti_return,
    m.brent_volatility_7d,
    m.brent_volatility_30d,
    m.wti_volatility_7d,
    m.wti_volatility_30d,
    m.brent_wti_spread,
    m.event_flag,
    m.event_type,
    m.event_severity,
    m.event_description
FROM ops_market_daily AS m
LEFT JOIN ops_gpr_daily AS g
    ON g.gpr_date = m.market_date;

CREATE OR REPLACE VIEW vw_event_price_reaction AS
SELECT
    e.event_id,
    e.event_date,
    e.event_name,
    e.event_type,
    e.event_category,
    e.event_severity,
    m.brent_price_usd,
    m.wti_price_usd,
    m.gpr_index,
    m.brent_return,
    m.wti_return,
    m.brent_volatility_7d,
    m.brent_volatility_30d
FROM ops_events AS e
LEFT JOIN ops_market_daily AS m
    ON m.market_date = e.event_date;

CREATE OR REPLACE VIEW vw_country_petrol_impact AS
SELECT
    c.country_name,
    c.iso3,
    c.region_name,
    c.oil_import_dependency,
    i.oil_import_pct,
    i.gdp_impact_pct,
    i.inflation_risk,
    i.stock_market_change_pct,
    i.currency_pressure,
    i.policy_response,
    i.vulnerability,
    p.snapshot_stage,
    p.snapshot_date,
    p.price_local_per_liter,
    p.price_usd_per_liter,
    p.pct_increase,
    p.trend
FROM ops_countries AS c
LEFT JOIN ops_country_impact AS i
    ON i.country_id = c.country_id
LEFT JOIN ops_petrol_price_snapshots AS p
    ON p.country_id = c.country_id;
