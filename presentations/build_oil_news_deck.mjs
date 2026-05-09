import {
  Presentation,
  PresentationFile,
  column,
  row,
  grid,
  panel,
  text,
  rule,
  fill,
  hug,
  fixed,
  wrap,
  grow,
  fr,
  auto,
} from "@oai/artifact-tool";

const OUT = "output/presentations/oil_news_project_presentation.pptx";

const presentation = Presentation.create({
  slideSize: { width: 1920, height: 1080 },
});

const colors = {
  ink: "#10202E",
  muted: "#5C6C7A",
  faint: "#E9EEF3",
  line: "#B8C5D0",
  blue: "#1E5C8A",
  teal: "#1F8A7B",
  gold: "#C58A1A",
  red: "#B65B4B",
  white: "#FFFFFF",
};

const titleStyle = { fontSize: 54, bold: true, color: colors.ink };
const subtitleStyle = { fontSize: 24, color: colors.muted };
const labelStyle = { fontSize: 18, color: colors.muted };
const bodyStyle = { fontSize: 24, color: colors.ink };
const smallStyle = { fontSize: 15, color: colors.muted };

function addSlide(root) {
  const slide = presentation.slides.add();
  slide.compose(root, { frame: { left: 0, top: 0, width: 1920, height: 1080 }, baseUnit: 8 });
}

function header(title, subtitle) {
  return column({ name: "header", width: fill, height: hug, gap: 12 }, [
    text(title, { name: "slide-title", width: fill, height: hug, style: titleStyle }),
    subtitle
      ? text(subtitle, { name: "slide-subtitle", width: wrap(1280), height: hug, style: subtitleStyle })
      : rule({ name: "title-rule", width: fixed(180), stroke: colors.teal, weight: 5 }),
  ]);
}

function metric(value, label, accent = colors.blue) {
  return column({ name: `metric-${label}`, width: fill, height: hug, gap: 4 }, [
    text(value, { width: fill, height: hug, style: { fontSize: 58, bold: true, color: accent } }),
    text(label, { width: fill, height: hug, style: labelStyle }),
  ]);
}

function metricCompact(value, label, accent = colors.blue) {
  return column({ name: `metric-${label}`, width: fill, height: hug, gap: 2 }, [
    text(value, { width: fill, height: hug, style: { fontSize: 34, bold: true, color: accent } }),
    text(label, { width: fill, height: hug, style: { fontSize: 15, color: colors.muted } }),
  ]);
}

function bullet(items) {
  return column(
    { name: "bullets", width: fill, height: hug, gap: 16 },
    items.map((item) =>
      text(`- ${item}`, { width: fill, height: hug, style: { fontSize: 24, color: colors.ink } }),
    ),
  );
}

function rowItem(label, value, detail, accent = colors.blue) {
  return grid(
    { name: `row-${label}`, width: fill, height: hug, columns: [fixed(270), fixed(170), fr(1)], rows: [auto], columnGap: 18 },
    [
      text(label, { width: fill, height: hug, style: { fontSize: 22, bold: true, color: colors.ink } }),
      text(value, { width: fill, height: hug, style: { fontSize: 25, bold: true, color: accent } }),
      text(detail, { width: fill, height: hug, style: { fontSize: 19, color: colors.muted } }),
    ],
  );
}

function rowItemCompact(label, value, detail, accent = colors.blue) {
  return grid(
    { name: `row-${label}`, width: fill, height: hug, columns: [fixed(150), fixed(135), fr(1)], rows: [auto], columnGap: 14 },
    [
      text(label, { width: fill, height: hug, style: { fontSize: 19, bold: true, color: colors.ink } }),
      text(value, { width: fill, height: hug, style: { fontSize: 21, bold: true, color: accent } }),
      text(detail, { width: fill, height: hug, style: { fontSize: 17, color: colors.muted } }),
    ],
  );
}

function step(num, title, detail, accent = colors.blue) {
  return row({ name: `step-${num}`, width: fill, height: hug, gap: 18 }, [
    panel(
      { width: fixed(58), height: fixed(58), padding: 0, fill: accent, borderRadius: "rounded-full" },
      text(String(num), {
        width: fill,
        height: fill,
        style: { fontSize: 27, bold: true, color: colors.white, align: "center" },
      }),
    ),
    column({ width: fill, height: hug, gap: 3 }, [
      text(title, { width: fill, height: hug, style: { fontSize: 25, bold: true, color: colors.ink } }),
      text(detail, { width: fill, height: hug, style: { fontSize: 19, color: colors.muted } }),
    ]),
  ]);
}

// Slide 1
addSlide(
  grid(
    {
      name: "cover-root",
      width: fill,
      height: fill,
      columns: [fr(1.1), fr(0.9)],
      rows: [fr(1), auto],
      padding: { x: 90, y: 78 },
      columnGap: 72,
      rowGap: 28,
    },
    [
      column({ width: fill, height: fill, gap: 26 }, [
        text("Semantic News and Oil Price Database Project", {
          width: wrap(920),
          height: hug,
          style: { fontSize: 78, bold: true, color: colors.ink },
        }),
        rule({ width: fixed(260), stroke: colors.gold, weight: 7 }),
        text("This project turns oil-market, geopolitical risk, semantic news, and country exposure CSV files into a reproducible analytics workflow with MySQL tables, SQL views, and a transparent pricing model.", {
          width: wrap(860),
          height: hug,
          style: { fontSize: 28, color: colors.muted },
        }),
      ]),
      column({ width: fill, height: fill, gap: 38 }, [
        metric("4,047", "daily market rows", colors.blue),
        metric("66,660", "country-month GPR rows", colors.teal),
        metric("1.516", "test RMSE, USD", colors.gold),
      ]),
      text("DatabaseProject - MySQL - scikit-learn - May 2026", {
        columnSpan: 2,
        width: fill,
        height: hug,
        style: smallStyle,
      }),
    ],
  ),
);

// Slide 2
addSlide(
  column({ name: "s2-root", width: fill, height: fill, padding: { x: 84, y: 64 }, gap: 46 }, [
    header("One workflow, three outputs", "The source CSVs become a queryable database, analysis surfaces, and predictive model artifacts."),
    grid({ width: fill, height: fill, columns: [fr(1), fr(1)], rows: [fr(1), fr(1)], columnGap: 42, rowGap: 30 }, [
      step(1, "Datasets", "Semantic news, market prices, events, country impact"),
      step(2, "MySQL", "Operational source tables plus dim/fact reporting tables"),
      step(3, "Views", "Daily oil/news features and event/country analysis"),
      step(4, "Model", "StandardScaler -> Ridge for next-day Brent"),
    ]),
    text("Final handoff includes scripts, schema docs, trained model, report, PDF, and this deck.", {
      width: fill,
      height: hug,
      style: { fontSize: 26, bold: true, color: colors.ink },
    }),
  ]),
);

// Slide 3
addSlide(
  column({ name: "s3-root", width: fill, height: fill, padding: { x: 84, y: 64 }, gap: 34 }, [
    header("The data mixes market prices with semantic risk", "Coverage spans oil prices, GPR/news indices, events, and country-level exposure."),
    grid({ width: fill, height: fill, columns: [fr(1.18), fr(0.82)], rows: [fr(1)], columnGap: 54 }, [
      column({ width: fill, height: fill, gap: 19 }, [
        rowItem("ops_market_daily", "4,047", "Daily Brent/WTI, macro indicators, volatility, lags, and events", colors.blue),
        rowItem("ops_gpr_daily", "15,078", "Daily geopolitical risk/news index and article counts", colors.teal),
        rowItem("ops_gpr_country_monthly", "66,660", "Country-month geopolitical risk measures", colors.gold),
        rowItem("ops_events", "55", "Named geopolitical, sanctions, disaster, and market events", colors.red),
        rowItem("ops_countries", "18", "Country dimension for impact and petrol analysis", colors.blue),
      ]),
      column({ width: fill, height: fill, gap: 28 }, [
        metric("2010-2026", "market table date span", colors.blue),
        metric("1985-2026", "daily GPR date span", colors.teal),
        text("The modeling table already aligns Brent, WTI, volatility, GPR, event flags, and lag features for supervised learning.", {
          width: fill,
          height: hug,
          style: bodyStyle,
        }),
      ]),
    ]),
  ]),
);

// Slide 4
addSlide(
  column({ name: "s4-root", width: fill, height: fill, padding: { x: 84, y: 64 }, gap: 34 }, [
    header("Two database layers keep analysis flexible", "Operational tables preserve source shape; dimensional tables support BI-style reporting."),
    grid({ width: fill, height: fill, columns: [fr(1), fr(1)], rows: [fr(1), auto], columnGap: 46, rowGap: 26 }, [
      column({ width: fill, height: fill, gap: 18 }, [
        text("Operational layer", { width: fill, height: hug, style: { fontSize: 34, bold: true, color: colors.blue } }),
        bullet(["ops_market_daily", "ops_gpr_daily and monthly", "ops_events", "ops_countries and petrol snapshots"]),
      ]),
      column({ width: fill, height: fill, gap: 18 }, [
        text("Dimensional layer", { width: fill, height: hug, style: { fontSize: 34, bold: true, color: colors.teal } }),
        bullet(["dim_date, dim_country, dim_event", "fact_market_daily", "fact_gpr_daily and monthly", "fact_country_impact and petrol prices"]),
      ]),
      text("Primary joins: market_date = gpr_date, market_date = event_date, country_id, and iso3.", {
        columnSpan: 2,
        width: fill,
        height: hug,
        style: { fontSize: 28, bold: true, color: colors.ink },
      }),
    ]),
  ]),
);

// Slide 5
addSlide(
  column({ name: "s5-root", width: fill, height: fill, padding: { x: 84, y: 64 }, gap: 34 }, [
    header("Views turn raw tables into analysis surfaces", "Three MySQL views package the joins needed for modeling, event study, and country impact analysis."),
    grid({ width: fill, height: fill, columns: [fr(1), fr(1), fr(1)], rows: [fr(1)], columnGap: 34 }, [
      column({ width: fill, height: fill, gap: 18 }, [
        text("vw_daily_oil_news_features", { width: fill, height: hug, style: { fontSize: 30, bold: true, color: colors.blue } }),
        text("Daily Brent/WTI, GPR, event, and volatility features for analysis and model inputs.", { width: fill, height: hug, style: bodyStyle }),
      ]),
      column({ width: fill, height: fill, gap: 18 }, [
        text("vw_event_price_reaction", { width: fill, height: hug, style: { fontSize: 30, bold: true, color: colors.teal } }),
        text("Event-date oil price and risk indicators to inspect geopolitical market reactions.", { width: fill, height: hug, style: bodyStyle }),
      ]),
      column({ width: fill, height: fill, gap: 18 }, [
        text("vw_country_petrol_impact", { width: fill, height: hug, style: { fontSize: 30, bold: true, color: colors.gold } }),
        text("Country exposure, petrol price snapshots, and vulnerability indicators in one reporting surface.", { width: fill, height: hug, style: bodyStyle }),
      ]),
    ]),
  ]),
);

// Slide 6
addSlide(
  grid({ name: "s6-root", width: fill, height: fill, columns: [fr(0.9), fr(1.1)], rows: [auto, fr(1)], padding: { x: 84, y: 64 }, columnGap: 54, rowGap: 34 }, [
    header("A transparent scikit-learn model predicts next-day Brent", "The model is deliberately interpretable and reproducible for a database project showcase."),
    column({ width: fill, height: hug, gap: 10 }, [
      metricCompact("3,236", "training rows", colors.blue),
      metricCompact("810", "test rows", colors.teal),
      metricCompact("0.1", "Ridge alpha", colors.gold),
    ]),
    column({ width: fill, height: fill, gap: 26 }, [
      step(1, "Chronological split", "Train through 2022-12-21; test through 2026-03-12", colors.blue),
      step(2, "Standardize inputs", "Scale market, risk, volatility, lag, and event features", colors.teal),
      step(3, "Fit Ridge regression", "Regularized linear model saved as a joblib pipeline", colors.gold),
      step(4, "Export metrics", "JSON metadata and CSV test predictions for review", colors.red),
    ]),
  ]),
);

// Slide 7
addSlide(
  column({ name: "s7-root", width: fill, height: fill, padding: { x: 84, y: 64 }, gap: 34 }, [
    header("Features combine prices, risk, volatility, and events", "The model uses 20 fields from ops_market_daily.csv."),
    grid({ width: fill, height: fill, columns: [fr(1), fr(1), fr(1), fr(1)], rows: [fr(1), fr(1)], columnGap: 26, rowGap: 30 }, [
      metric("2", "current oil prices", colors.blue),
      metric("2", "macro indicators", colors.teal),
      metric("1", "GPR risk index", colors.gold),
      metric("2", "daily returns", colors.red),
      metric("6", "lagged prices", colors.blue),
      metric("4", "volatility fields", colors.teal),
      metric("1", "Brent-WTI spread", colors.gold),
      metric("2", "event features", colors.red),
    ]),
  ]),
);

// Slide 8
addSlide(
  grid({ name: "s8-root", width: fill, height: fill, columns: [fr(1.05), fr(0.95)], rows: [auto, fr(1)], padding: { x: 84, y: 64 }, columnGap: 50, rowGap: 28 }, [
    header("The model narrowly beats a strong baseline", "Next-day oil pricing is hard because yesterday's price is already an excellent predictor."),
    column({ width: fill, height: hug, gap: 14 }, [
      text("Test-set comparison", { width: fill, height: hug, style: { fontSize: 30, bold: true, color: colors.ink } }),
      rowItemCompact("MAE", "1.118", "Model error in USD; baseline is 1.113", colors.teal),
      rowItemCompact("RMSE", "1.516", "Model error in USD; baseline is 1.536", colors.blue),
      rowItemCompact("MAPE", "1.463%", "Percent error on chronological test rows", colors.gold),
      rowItemCompact("R-squared", "0.967", "Variance explained on future holdout rows", colors.red),
      text("Baseline = previous trading-day Brent price.", { width: fill, height: hug, style: smallStyle }),
    ]),
    column({ width: fill, height: fill, gap: 28 }, [
      metric("1.516", "test RMSE, USD", colors.blue),
      metric("1.118", "test MAE, USD", colors.teal),
      metric("0.967", "test R-squared", colors.gold),
      text("The result is credible because it is evaluated on a future holdout period, not a shuffled split.", {
        width: fill,
        height: hug,
        style: bodyStyle,
      }),
    ]),
  ]),
);

// Slide 9
addSlide(
  column({ name: "s9-root", width: fill, height: fill, padding: { x: 84, y: 64 }, gap: 36 }, [
    header("The handoff is reproducible", "Everything needed to load, train, predict, and present is in the workspace."),
    grid({ width: fill, height: fill, columns: [fr(1), fr(1)], rows: [fr(1), auto], columnGap: 50, rowGap: 24 }, [
      column({ width: fill, height: fill, gap: 18 }, [
        text("Run path", { width: fill, height: hug, style: { fontSize: 34, bold: true, color: colors.blue } }),
        bullet(["python -m pip install -r requirements.txt", "python src\\load_mysql.py --replace", "python src\\train_oil_model.py", "python src\\predict_oil_price.py"]),
      ]),
      column({ width: fill, height: fill, gap: 18 }, [
        text("Deliverables", { width: fill, height: hug, style: { fontSize: 34, bold: true, color: colors.teal } }),
        bullet(["MySQL loader and views", "Database schema guide", "Trained sklearn model", "Markdown/PDF report", "Editable deck source, PPTX, and PDF"]),
      ]),
      text("Next improvements: foreign keys and indexes, dashboarding from views, rolling-window validation, and model prediction tables in MySQL.", {
        columnSpan: 2,
        width: fill,
        height: hug,
        style: { fontSize: 25, bold: true, color: colors.ink },
      }),
    ]),
  ]),
);

const pptxBlob = await PresentationFile.exportPptx(presentation);
await pptxBlob.save(OUT);
console.log(OUT);
