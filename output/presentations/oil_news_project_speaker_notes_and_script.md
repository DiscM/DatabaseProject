# Speaker Notes and General Script

Deck: `oil_news_project_presentation.pptx`

Suggested pacing: about 6 to 8 minutes total, roughly 35 to 55 seconds per slide depending on how much detail you want to add during the database/modeling sections.

## Opening

Use the first slide to set the promise: this is an end-to-end database and modeling project that turns semantic news and geopolitical risk data into oil price signals.

## Slide-by-slide notes

### Slide 1: Semantic News to Oil Price Signals

Speaker notes:
- Open by framing the project as an end-to-end analytics workflow, not just a model. The point is to connect semantic news risk, oil price history, geopolitical events, and country exposure in a way that can be queried and reproduced.
- Call out the three headline numbers: roughly 4,000 daily market rows, more than 66,000 country-month GPR rows, and a test RMSE of 1.516 dollars. These establish scale, breadth, and model performance.
- Transition: I will walk through how the data becomes a database, how the database becomes analysis views, and how those views support a transparent Brent price model.

Script:

Today I am presenting an end-to-end project that turns semantic news and geopolitical risk data into oil price signals. The project is built around MySQL and scikit-learn, so the work is reproducible from raw CSV inputs through database tables, analytics views, model artifacts, and presentation outputs. The quick headline is that we combine about 4,047 daily market rows with much broader geopolitical risk coverage, including 66,660 country-month observations, and the final next-day Brent model reaches a test RMSE of 1.516 dollars.

### Slide 2: One workflow, three outputs

Speaker notes:
- Explain the pipeline in four steps: source datasets, MySQL storage, analytics views, and a scikit-learn model.
- Emphasize that the outputs serve different audiences: database tables for traceability, views for analysis, and model artifacts for prediction.
- Transition: Before getting into schema and modeling, show what the data actually contains.

Script:

The workflow has four main stages. First, the source datasets bring together semantic news, oil prices, geopolitical events, and country impact data. Second, MySQL stores those datasets in operational tables and dimensional reporting tables. Third, SQL views package the joins into analysis-ready surfaces. Finally, the modeling layer uses a StandardScaler into Ridge Regression to predict next-day Brent prices. The value here is that the project does not stop at a notebook; it produces a queryable database, reusable analysis views, and saved model artifacts.

### Slide 3: The data mixes market prices with semantic risk

Speaker notes:
- Use this slide to establish coverage and credibility. The market data runs from 2010 to 2026, while daily GPR coverage reaches back to 1985.
- Briefly describe each table: market prices and volatility, GPR/news signals, country-month risk, events, and countries.
- Transition: With those ingredients, the database uses two layers so raw data and reporting data each have a clear home.

Script:

The dataset is deliberately mixed: it includes daily oil market data, geopolitical risk and news indices, named events, and country-level exposure. The operational market table has 4,047 rows covering Brent, WTI, macro indicators, volatility, lags, and event fields. The GPR daily table adds more than 15,000 rows of risk and news index data, while the country-month table adds 66,660 rows for country-level geopolitical risk measures. That gives the project both a daily market lens and a broader geopolitical context.

### Slide 4: Two database layers keep analysis flexible

Speaker notes:
- Contrast the operational layer and dimensional layer. The operational layer preserves the source shape; the dimensional layer makes reporting easier.
- Mention the main joins: dates connect market, GPR, and events; country identifiers connect country and impact tables.
- Transition: Once the schema is in place, the views turn those joins into reusable surfaces.

Script:

The schema is organized into two layers. The operational layer keeps the source-oriented tables close to their original structure, which makes loading and auditing easier. The dimensional layer introduces standard reporting tables such as date, country, event, and fact tables. This separation gives us flexibility: we can preserve the raw shape of the inputs, but still support cleaner BI-style joins and analysis. The most important joins are date-based joins across market, GPR, and event data, plus country identifiers for country-level impact analysis.

### Slide 5: Views turn raw tables into analysis surfaces

Speaker notes:
- Frame the three views as reusable products of the database layer.
- Explain that vw_daily_oil_news_features supports modeling, vw_event_price_reaction supports event study, and vw_country_petrol_impact supports exposure reporting.
- Transition: The first view feeds the transparent machine learning workflow.

Script:

Rather than asking every analysis to repeat the same joins, the project defines three MySQL views. The daily oil-news feature view brings together Brent, WTI, geopolitical risk, events, and volatility features for analysis and model inputs. The event reaction view is meant for inspecting price and risk indicators around named geopolitical events. The country petrol impact view combines country exposure, petrol price snapshots, and vulnerability indicators. These views make the database more usable because they package the repeated logic once.

### Slide 6: A transparent scikit-learn model predicts next-day Brent

Speaker notes:
- Stress that the split is chronological, which is important for time-series credibility.
- Point out the model is intentionally simple and explainable: scaled features plus Ridge Regression.
- Mention the train/test ranges and exported artifacts.
- Transition: The next slide explains what signals the model actually sees.

Script:

For modeling, the project uses a transparent scikit-learn pipeline: standardize the features, then fit Ridge Regression. The split is chronological, with training through December 21, 2022 and testing through March 12, 2026. That matters because the test set represents future holdout rows rather than a shuffled sample. The workflow saves the trained joblib pipeline, JSON metadata, and CSV test predictions, so the result can be reviewed and reused outside the training script.

### Slide 7: Features combine prices, risk, volatility, and events

Speaker notes:
- Describe the feature groups at a high level rather than reading every metric.
- Emphasize interpretability: current prices, macro fields, GPR, returns, lags, volatility, spread, and event features.
- Transition: Now that the inputs are clear, move to performance and baseline comparison.

Script:

The model uses 20 fields from the market daily data. They are intentionally transparent: current Brent and WTI prices, macro indicators, the GPR risk index, daily returns, lagged prices, volatility fields, the Brent-WTI spread, and event features. This is not a black-box feature set. The goal is to combine market momentum, risk context, volatility, and event information in a way that is easy to explain during review.

### Slide 8: The model narrowly beats a strong baseline

Speaker notes:
- Set expectations: next-day price prediction is difficult because yesterday?s price is already a very strong baseline.
- Highlight the most important comparison: RMSE improves from 1.536 baseline to 1.516 model.
- Mention R-squared and MAPE as supporting metrics, but avoid overstating the gain.
- Transition: Close by showing how to reproduce and extend the handoff.

Script:

The performance story is modest but useful. Next-day oil price prediction is hard because the previous trading-day Brent price is already an excellent predictor. Against that strong baseline, the model narrowly improves RMSE from 1.536 dollars to 1.516 dollars. MAE is essentially tied, MAPE is 1.463 percent, and R-squared is 0.967 on the chronological holdout period. The right takeaway is not that this model perfectly predicts oil prices; it is that the database and feature workflow produce a credible, testable signal that slightly improves on a strong naive baseline.

### Slide 9: The handoff is reproducible

Speaker notes:
- Close with reproducibility: install requirements, load MySQL, train the model, and run prediction.
- Summarize deliverables: loader, views, schema guide, model, report, and presentation exports.
- End with next steps: keys/indexes, dashboarding, rolling-window validation, and prediction tables.

Script:

The final handoff is designed to be reproducible. A reviewer can install the requirements, load the MySQL database, train the model, and run the prediction script using the commands on the slide. The deliverables include the MySQL loader and analytics views, a schema guide, the trained scikit-learn model, the report, and the editable presentation outputs. The natural next steps would be adding stronger foreign keys and indexes, building a dashboard, moving to rolling-window validation, and storing predictions back into database tables.

## Closing

Overall, this project shows how semantic news data can be operationalized: stored in a relational database, joined into reusable analytical views, and carried into a transparent forecasting workflow. The model result is intentionally framed carefully: it is a narrow improvement over a strong baseline, but the larger contribution is the reproducible data pipeline that makes future analysis and model iteration straightforward.
