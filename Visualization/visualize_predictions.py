import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates

def create_visualizations(predictions_csv_path, output_dir, forecast_csv_path=None, market_csv_path=None):
    """
    Reads the test predictions and generates graphs comparing actual vs predicted prices.
    Optionally also plots a forward forecast extending beyond actual data.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {predictions_csv_path}...")
    try:
        df = pd.read_csv(predictions_csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {predictions_csv_path}. Please ensure the model has been trained and output data exists.")
        return

    # Convert market_date to datetime
    df['market_date'] = pd.to_datetime(df['market_date'])
    
    # Sort by date just in case
    df = df.sort_values('market_date')

    # Read horizon from CSV if present, otherwise default to 5 trading days
    horizon_days = int(df['horizon_days'].iloc[0]) if 'horizon_days' in df.columns else 5
    horizon_label = f"{horizon_days} Trading Days (~1 Week) Ahead"

    # --- Detect column names (support both old and new schemas) ---
    actual_col = 'actual_brent_future' if 'actual_brent_future' in df.columns else 'actual_brent_next'
    predicted_col = 'predicted_brent_future' if 'predicted_brent_future' in df.columns else 'predicted_brent_next'

    # 1. Line plot of Actual vs Predicted over time
    plt.figure(figsize=(14, 7))
    plt.plot(df['market_date'], df[actual_col], label='Actual Brent Price', color='blue', alpha=0.7, linewidth=1.5)
    plt.plot(df['market_date'], df[predicted_col], label=f'Predicted Brent Price ({horizon_label})', color='orange', alpha=0.8, linewidth=1.5)
    
    plt.title(f'Brent Crude Oil Price: Actual vs Predicted {horizon_label} (Test Set)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gcf().autofmt_xdate()
    
    line_plot_path = os.path.join(output_dir, 'actual_vs_predicted_line.png')
    plt.savefig(line_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved line plot to {line_plot_path}")
    
    # 2. Scatter plot of Actual vs Predicted (to show correlation)
    plt.figure(figsize=(8, 8))
    plt.scatter(df[actual_col], df[predicted_col], alpha=0.5, color='green')
    
    # Add perfect prediction line (y=x)
    min_val = min(df[actual_col].min(), df[predicted_col].min())
    max_val = max(df[actual_col].max(), df[predicted_col].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction (y=x)')
    
    plt.title(f'Prediction Accuracy: Actual vs Predicted ({horizon_label})', fontsize=14)
    plt.xlabel('Actual Price (USD)', fontsize=12)
    plt.ylabel(f'Predicted Price (USD)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    scatter_plot_path = os.path.join(output_dir, 'actual_vs_predicted_scatter.png')
    plt.savefig(scatter_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved scatter plot to {scatter_plot_path}")
    
    # 3. Line plot including Baseline
    plt.figure(figsize=(14, 7))
    plt.plot(df['market_date'], df[actual_col], label='Actual Brent Price', color='blue', alpha=0.7, linewidth=1.5)
    plt.plot(df['market_date'], df[predicted_col], label=f'Predicted Brent ({horizon_label})', color='orange', alpha=0.8, linewidth=1.5)
    plt.plot(df['market_date'], df['baseline_previous_brent'], label='Baseline (Current Day Price)', color='gray', alpha=0.5, linestyle='--')
    
    plt.title(f'Model vs Baseline vs Actual Prices ({horizon_label})', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gcf().autofmt_xdate()
    
    baseline_plot_path = os.path.join(output_dir, 'model_vs_baseline.png')
    plt.savefig(baseline_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved baseline comparison plot to {baseline_plot_path}")

    # 4. Forward Forecast — extends predicted prices beyond actual data
    _plot_forward_forecast(forecast_csv_path, market_csv_path, output_dir, horizon_label)


def _plot_forward_forecast(forecast_csv_path, market_csv_path, output_dir, horizon_label):
    """Plot recent actual prices + forecasted future prices on one chart."""
    if forecast_csv_path is None or market_csv_path is None:
        print("Skipping forward forecast plot (no forecast CSV or market CSV provided).")
        return

    try:
        fc = pd.read_csv(forecast_csv_path)
        market = pd.read_csv(market_csv_path)
    except FileNotFoundError as e:
        print(f"Skipping forward forecast plot: {e}")
        return

    fc['forecast_date'] = pd.to_datetime(fc['forecast_date'])
    market['market_date'] = pd.to_datetime(market['market_date'])
    market = market.sort_values('market_date')

    # Show the last ~60 trading days of actual data for context
    recent = market.tail(60).copy()

    plt.figure(figsize=(14, 7))

    # Plot actual prices
    plt.plot(
        recent['market_date'], recent['brent_price_usd'].astype(float),
        label='Actual Brent Price', color='blue', linewidth=2, alpha=0.8,
    )

    # Bridge point: connect actual line to forecast line
    last_actual_date = recent['market_date'].iloc[-1]
    last_actual_price = float(recent['brent_price_usd'].iloc[-1])

    # Build forecast series with bridge
    fc_sorted = fc.sort_values('forecast_date')
    forecast_dates = pd.concat([pd.Series([last_actual_date]), fc_sorted['forecast_date']], ignore_index=True)
    forecast_prices = pd.concat(
        [pd.Series([last_actual_price]), fc_sorted['predicted_brent_usd'].astype(float)],
        ignore_index=True,
    )

    plt.plot(
        forecast_dates, forecast_prices,
        label='Forecasted Brent Price', color='red', linewidth=2.5,
        linestyle='--', marker='o', markersize=5, alpha=0.9,
    )

    # Shade the forecast region
    plt.axvspan(
        last_actual_date, fc_sorted['forecast_date'].max(),
        alpha=0.08, color='red', label='Forecast Window',
    )

    # Annotate forecast values
    for _, row in fc_sorted.iterrows():
        plt.annotate(
            f"${row['predicted_brent_usd']:.2f}",
            xy=(row['forecast_date'], row['predicted_brent_usd']),
            textcoords="offset points", xytext=(0, 12),
            ha='center', fontsize=8, color='red', fontweight='bold',
        )

    plt.title(f'Brent Crude Oil - Forward Price Forecast ({horizon_label})', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(True, alpha=0.3)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()

    forecast_plot_path = os.path.join(output_dir, 'forward_forecast.png')
    plt.savefig(forecast_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved forward forecast plot to {forecast_plot_path}")


if __name__ == "__main__":
    # Setup paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    predictions_file = os.path.join(project_root, 'model_artifacts', 'test_predictions.csv')
    forecast_file = os.path.join(project_root, 'model_artifacts', 'forward_forecast.csv')
    market_file = os.path.join(project_root, 'datasets', 'ops_market_daily.csv')
    
    # Output visualizations directly into the Visualization folder
    create_visualizations(predictions_file, script_dir, forecast_file, market_file)
