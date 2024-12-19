"""
model.py

Description:
This script is designed to facilitate the end-to-end process of training machine learning models specifically tailored for binary classification tasks. It incorporates several key functionalities:
1. Data Preprocessing: Handles missing values and encodes categorical variables to prepare data for modeling.
2. Model Training: Implements a machine learning pipeline using LightGBM through GridSearchCV for hyperparameter optimization. The script utilizes advanced techniques such as SMOTEENN for handling imbalanced datasets to enhance model performance.
3. Performance Visualization: Provides functions to plot ROC curves and calculate the accuracy specifically for 'Unadmitted' patients, helping in the evaluation of the model's performance on classification tasks.

The script is structured to be used with a dataset where users specify input features and target variables. It outputs optimal model parameters, evaluation metrics, and visualizations to assess the model's efficacy in predicting binary outcomes.
"""

import logging

import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImPipeline
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class BaseModel:
    """Base class for common functionalities in data preprocessing and modeling."""
    def __init__(self):
        logging.info(f"{self.__class__.__name__} initialized.")
    
    def log_message(self, message: str):
        """Logs a given message."""
        logging.info(message)

class DataPreprocessor(BaseModel):
    """Preprocesses data by filling missing values and encoding categorical variables."""
    def __init__(self):
        super().__init__()

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Identify numeric and categorical columns and transform the dataset.
        Fills missing values, encodes categorical variables, and checks for invalid data.
        """
        X = X.copy()
        
        # Fill missing values for numeric columns
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        for col in numeric_cols:
            X[col] = X[col].fillna(X[col].median())
        
        # Fill missing values for categorical columns and encode
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Convert all columns to numeric if possible
        X = X.apply(pd.to_numeric, errors='coerce')
        
        # Check for NaNs after conversion and handle them
        if X.isnull().values.any():
            self.log_message("Data contains NaN values after preprocessing. Replacing NaNs with 0.")
            X = X.fillna(0)
        
        return X

class HospitalReadmissionModel(BaseModel):
    """A machine learning model for predicting hospital readmissions."""
    def __init__(self):
        super().__init__()

    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Pipeline:
        """
        Trains a machine learning model using LightGBM with GridSearchCV for hyperparameter optimization.
        Utilizes SMOTEENN for handling imbalanced datasets and 'force_col_wise' optimization.
        """
        self.log_message("Starting model training with GridSearchCV using LightGBM with 'force_col_wise' optimization...")
        pipeline = ImPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTEENN(smote=SMOTE(sampling_strategy='auto'))),
            ('classifier', LGBMClassifier(
                random_state=42,
                n_jobs=-1,
                class_weight='balanced',
                force_col_wise=True
            ))
        ])

        param_grid = {
            'classifier__num_leaves': [31, 61],
            'classifier__max_depth': [-1, 10],
            'classifier__learning_rate': [0.05, 0.1],
            'classifier__n_estimators': [100, 200]
        }

        model_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=3,
            verbose=1
        )

        model_search.fit(X, y)
        self.log_message(f"Optimal Parameters Found: {model_search.best_params_}")
        return model_search.best_estimator_

    def plot_roc_curve(
        self, y_test: pd.Series,
        predictions: pd.Series, 
        y_scores: pd.Series
    ) -> None:
        """
        Plots ROC curve and accuracy for unadmitted patients.
        """
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        plt.subplot(1, 2, 2)
        unadmitted_accuracy = accuracy_score(y_test == 0, predictions == 0)
        plt.bar(['Unadmitted'], [unadmitted_accuracy], color='blue')
        plt.ylim([0, 1])
        plt.ylabel('Accuracy')
        plt.title('Accuracy for Unadmitted Patients')
        plt.show()
