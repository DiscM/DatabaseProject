import json
import os
from pathlib import Path

def create_markdown_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    }

def create_code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.split("\n")]
    }

def main():
    cells = []
    
    # 1. Title and Introduction
    cells.append(create_markdown_cell("# Oil News Project Demonstration\nThis notebook compiles the project scripts into a single run-through for easy demonstration."))
    
    # 2. Database Config
    db_config_code = Path("src/db_config.py").read_text(encoding="utf-8")
    cells.append(create_markdown_cell("## 1. Database Configuration\nFirst, we load the database configuration."))
    cells.append(create_code_cell(db_config_code))
    
    # 3. Load MySQL
    load_mysql_code = Path("src/load_mysql.py").read_text(encoding="utf-8")
    # Patch argparse to work in Jupyter
    load_mysql_code = load_mysql_code.replace("parser.parse_args()", "parser.parse_args([])")
    # Patch the relative path for the sql script
    load_mysql_code = load_mysql_code.replace('Path("sql") / "analytics_views.sql"', 'Path("sql/analytics_views.sql")')
    cells.append(create_markdown_cell("## 2. Load Datasets into MySQL\nLoad CSV datasets into the MySQL database using the config."))
    cells.append(create_code_cell(load_mysql_code))
    cells.append(create_code_cell("# Run the load_mysql main function\nmain()"))
    
    # 4. Train Model
    train_model_code = Path("src/train_oil_model.py").read_text(encoding="utf-8")
    train_model_code = train_model_code.replace("parser.parse_args()", "parser.parse_args([])")
    cells.append(create_markdown_cell("## 3. Train Oil Price Model\nTrain the predictive model for Brent crude oil prices."))
    cells.append(create_code_cell(train_model_code))
    cells.append(create_code_cell("# Run the training process\nmain()"))
    
    # 5. Predict Price
    predict_code = Path("src/predict_oil_price.py").read_text(encoding="utf-8")
    predict_code = predict_code.replace("parser.parse_args()", "parser.parse_args([])")
    cells.append(create_markdown_cell("## 4. Predict Next Trading-Day Price\nPredict the oil price based on the latest data."))
    cells.append(create_code_cell(predict_code))
    cells.append(create_code_cell("# Run the prediction\nmain()"))
    
    # 6. Visualizations
    viz_code = Path("Visualization/visualize_predictions.py").read_text(encoding="utf-8")
    # We remove the if __name__ == "__main__": block and provide custom code
    viz_main_custom = """
predictions_file = os.path.join('model_artifacts', 'test_predictions.csv')
# Make visualizations display inline instead of saving, or just let them save
%matplotlib inline
create_visualizations(predictions_file, 'Visualization')
"""
    # Just grab everything before the __main__ block
    viz_code = viz_code.split('if __name__ == "__main__":')[0]
    
    cells.append(create_markdown_cell("## 5. Visualizations\nFinally, generate visualizations of the model predictions vs actual prices."))
    cells.append(create_code_cell(viz_code + viz_main_custom))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open("oil_news_project_demo.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

if __name__ == "__main__":
    main()
