"""
run.py

Description:
This script serves as the execution point for setting up and running the machine learning pipeline defined in model.py. It is responsible for orchestrating the workflow that includes data loading, preprocessing, model training, evaluation, and visualization. The script leverages functionalities from model.py to train a model on user-specified data and outputs evaluation metrics and visualizations to assess the model's performance effectively.

Features:
- Data Loading: Loads data from a specified CSV file.
- Data Preprocessing: Uses the DataPreprocessor class from model.py to prepare the dataset for training.
- Model Training: Calls the train_model function to train the model using the preprocessed data.
- Evaluation: Evaluates the model's performance using the ROC AUC score and a classification report.
- Visualization: Utilizes the plot_roc_curve function to generate an ROC curve that helps in visual assessment of model performance.

Usage:
This script is intended to be run from the command line with an argument specifying the path to the dataset. 
"""

import argparse
import logging
import os
import sys
import time

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from hospital_readmission_model import DataPreprocessor, HospitalReadmissionModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class App:
    def __init__(self, file_name: str) -> None:
        self.file_name = file_name

    def run(self) -> None:
        """
        Main function that orchestrates the model training and evaluation process.
        """
        try:
            # Start time for logging execution time
            start = time.time()

            # Load data
            data = pd.read_csv(self.file_name)
            logging.info("Data loaded successfully from %s", self.file_name)

            # Preprocess data
            preprocessor = DataPreprocessor()
            X = preprocessor.fit_transform(data.drop('readmitted', axis=1))
            y = data['readmitted'].astype(int)

            # Validate processed data
            # Ensure all columns in the DataFrame are of numeric type
            if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
                logging.error(
                    "Data contains non-numeric values that could not be converted. Please clean your data."
                )
                return

            if X.isnull().values.any():
                logging.error(
                    "Data contains NaN values after preprocessing. Please clean your data."
                )
                return

            # Split data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            logging.info("Data split into training and test sets successfully.")

            # Train the model
            model = HospitalReadmissionModel()
            best_model = model.train_model(X_train, y_train)

            # Make predictions
            predictions = best_model.predict(X_test)
            y_scores = best_model.predict_proba(X_test)[:, 1]

            # Evaluate model
            report = classification_report(y_test, predictions)
            roc_score = roc_auc_score(y_test, y_scores)
            logging.info("Classification Report:\n%s", report)
            logging.info("ROC AUC Score: %f", roc_score)

            # Log execution time
            end = time.time()
            logging.info(
                "Model training and evaluation completed in %.2f seconds.", end - start
            )

            # Plot ROC curve
            model.plot_roc_curve(y_test, predictions, y_scores)

        except FileNotFoundError:
            logging.error("The file %s does not exist. Please provide a valid file path.", file_name)
        except pd.errors.EmptyDataError:
            logging.error("The file %s is empty. Please provide a valid CSV file.", file_name)
        except KeyError as e:
            logging.error("Column %s is missing in the dataset. Please ensure all required columns are present.", e)
        except Exception as e:
            logging.error("An error occurred during the process: %s", e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train and evaluate a model for hospital readmission."
    )
    parser.add_argument(
        'file_name', 
        type=str,
        help="Path to the CSV file containing the dataset."
    )
    args = parser.parse_args()
    
    if not args.file_name.lower().endswith('.csv'):
        logging.error("The provided file must be a CSV file.")
        sys.exit(1)

    if not os.path.isfile(args.file_name):
        logging.error("The file '%s' does not exist or is not accessible. Please provide a valid file path.", args.file_name)
        sys.exit(1)
    
    app = App(args.file_name)
    app.run()
