import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates

def create_visualizations(predictions_csv_path, output_dir):
    """
    Reads the test predictions and generates graphs comparing actual vs predicted prices.
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
    
    # 1. Line plot of Actual vs Predicted over time
    plt.figure(figsize=(14, 7))
    plt.plot(df['market_date'], df['actual_brent_next'], label='Actual Brent Price', color='blue', alpha=0.7, linewidth=1.5)
    plt.plot(df['market_date'], df['predicted_brent_next'], label='Predicted Brent Price', color='orange', alpha=0.8, linewidth=1.5)
    
    plt.title('Brent Crude Oil Price: Actual vs Predicted (Test Set)', fontsize=16)
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
    plt.scatter(df['actual_brent_next'], df['predicted_brent_next'], alpha=0.5, color='green')
    
    # Add perfect prediction line (y=x)
    min_val = min(df['actual_brent_next'].min(), df['predicted_brent_next'].min())
    max_val = max(df['actual_brent_next'].max(), df['predicted_brent_next'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction (y=x)')
    
    plt.title('Prediction Accuracy: Actual vs Predicted', fontsize=14)
    plt.xlabel('Actual Price (USD)', fontsize=12)
    plt.ylabel('Predicted Price (USD)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    scatter_plot_path = os.path.join(output_dir, 'actual_vs_predicted_scatter.png')
    plt.savefig(scatter_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved scatter plot to {scatter_plot_path}")
    
    # 3. Line plot including Baseline
    plt.figure(figsize=(14, 7))
    plt.plot(df['market_date'], df['actual_brent_next'], label='Actual Brent Price', color='blue', alpha=0.7, linewidth=1.5)
    plt.plot(df['market_date'], df['predicted_brent_next'], label='Predicted Brent Price', color='orange', alpha=0.8, linewidth=1.5)
    plt.plot(df['market_date'], df['baseline_previous_brent'], label='Baseline (Previous Day)', color='gray', alpha=0.5, linestyle='--')
    
    plt.title('Model vs Baseline vs Actual Prices', fontsize=16)
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

if __name__ == "__main__":
    # Setup paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    predictions_file = os.path.join(project_root, 'model_artifacts', 'test_predictions.csv')
    
    # Output visualizations directly into the Visualization folder
    create_visualizations(predictions_file, script_dir)
