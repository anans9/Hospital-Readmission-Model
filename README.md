# Hospital Readmission Prediction Model

## Overview

This is my project for predicting hospital readmissions using machine learning. The focus is on creating a binary classification model that is both effective and user-friendly. I used **LightGBM** as the main algorithm because it’s fast and performs well on a variety of datasets. To handle class imbalance, I included **SMOTEENN**, and for fine-tuning the model, I used **GridSearchCV**. There are also visualizations, like an **ROC curve**, to help understand the model's performance.

## What It Does

### Data Preprocessing
- It handles missing values by filling numeric ones with the median and encoding categorical data into dummy variables.
- Makes sure all data is numeric, replacing any leftover missing values with zeros.

### Model Training
- Trains a **LightGBM** model, which is optimized using **GridSearchCV**.
- Uses **SMOTEENN** to balance the dataset by over-sampling the minority class and cleaning noisy data.
- Tunes parameters like `num_leaves`, `max_depth`, and `learning_rate` to get the best results.

### Evaluation and Visualization
- Plots an **ROC curve** to show how well the model distinguishes between readmitted and non-readmitted patients.
- Shows accuracy for predicting 'Unadmitted' patients through a bar chart.

## How to Set It Up

1. Clone the repository.
2. Run the setup script:

   ```bash
   bash setup.bash

## Notes

- This project helped me practice working with machine learning, imbalanced datasets, and automating cross-platform setups. If you encounter any issues, re-running setup.bash should resolve missing dependencies. The project has been tested on macOS and Linux (Ubuntu).
  
