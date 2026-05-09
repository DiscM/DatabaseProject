# Oil News Database and Predictive Pricing Deck Source

Editable source outline for the project presentation.

## Slide 1 - Cover

**Title:** Semantic News and Oil Price Database Project  
**Message:** This project turns oil-market, geopolitical risk, semantic news, and country exposure CSV files into a reproducible analytics workflow with MySQL tables, SQL views, and a transparent pricing model.  
**Proof points:** 4,047 daily market rows; 66,660 country-month GPR rows; MySQL, SQL views, scikit-learn.

## Slide 2 - What the Project Builds

**Title:** One workflow, three outputs  
**Message:** Source CSVs become a queryable MySQL database, model-ready analytical views, and a trained predictive oil pricing model.  
**Flow:** Datasets -> MySQL operational tables -> SQL views -> scikit-learn model -> reports/deck/artifacts.

## Slide 3 - Data Foundation

**Title:** The data mixes market prices with semantic risk  
**Message:** The project combines daily oil market data, geopolitical risk/news indices, event annotations, and country-level petrol exposure.  
**Key rows:** `ops_market_daily` 4,047; `ops_gpr_daily` 15,078; `ops_gpr_country_monthly` 66,660; `ops_events` 55; `ops_countries` 18.

## Slide 4 - MySQL Schema

**Title:** Two database layers keep analysis flexible  
**Message:** Operational tables preserve source shape while dimensions and facts support BI-style reporting.  
**Key joins:** `market_date = gpr_date`, `market_date = event_date`, `country_id`, and `iso3`.

## Slide 5 - Analysis Views

**Title:** Views turn raw tables into analysis surfaces  
**Message:** Three SQL views package the useful joins for daily model features, event reaction analysis, and country petrol impact.

## Slide 6 - Predictive Model

**Title:** A transparent scikit-learn model predicts next-day Brent  
**Message:** The model uses a chronological split and a `StandardScaler -> Ridge` pipeline to predict the next trading-day Brent crude price.

## Slide 7 - Model Features

**Title:** Features combine prices, risk, volatility, and events  
**Message:** The 20 core inputs include current Brent/WTI, macro indicators, GPR index, returns, lagged prices, volatility, spread, and event flags. The training scripts also derive Brent momentum, short-term acceleration, and a volatility regime ratio.

## Slide 8 - Model Results

**Title:** The model narrowly beats a strong baseline  
**Message:** Test RMSE is 1.516 USD versus 1.536 USD for the previous-price baseline. The margin is small but credible for next-day oil pricing.

## Slide 9 - How to Run It

**Title:** The handoff is reproducible  
**Message:** Install dependencies, configure `.env`, load MySQL, train the model, and use the Markdown, PPTX, PDF, and HTML presentation artifacts.
