# Diabetes Prediction and Analysis with Machine Learning

## Overview
This project aims to predict diabetes occurrence using a dataset containing various health-related attributes. It employs a combination of data preprocessing, clustering, classification, and visualization techniques to analyze the dataset and build robust machine learning models.

## Features
- **Data Preprocessing**:
  - Handling missing values and outliers.
  - Standardizing features for improved model performance.
  - Addressing target class imbalance using SMOTE and other oversampling techniques.
  
- **Exploratory Data Analysis (EDA)**:
  - Correlation analysis and visualization.
  - Pairplots to understand feature interactions.
  - Distribution and count plots for target variable analysis.

- **Machine Learning Models**:
  - Logistic Regression, K-Nearest Neighbors (KNN), and Support Vector Machines (SVM).
  - Hyperparameter tuning using GridSearchCV.
  - Evaluation with metrics like accuracy, ROC-AUC, and classification reports.

- **Clustering**:
  - K-Means clustering to identify patterns in data.
  - Visualization of clusters with centroids.

- **Visualization**:
  - Correlation heatmaps, boxplots, confusion matrices, and ROC curves.
  - Automated saving of plots in a designated directory.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>

2. Navigate to the project directory:
    ```bash
    cd <repository-name>

3. Install the required Python packages:
    ```bash
    pip install -r requirements.txt

## Usage
1. Place your dataset file (diabetes_dataset.csv) in the project directory. Ensure the format and structure match the expected input.
2. Run the main script:
    ```bash
    python main.py
3. Outputs such as plots and model performance metrics will be saved in the plots directory.

## Project Structure
├── main.py                 # Main script for the project
├── plots/                  # Directory to save generated plots
├── requirements.txt        # Dependencies for the project
├── README.md               # Project documentation


## Dependencies
- Python 3.8 or later
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- imbalanced-learn