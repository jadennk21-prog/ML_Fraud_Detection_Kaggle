# Credit Card Fraud Detection Using Machine Learning

## Project Overview

This project focuses on developing a machine learning model to identify fraudulent credit card transactions using historical transaction data.

Credit card fraud detection is a binary classification problem with a highly imbalanced dataset, where legitimate transactions significantly outnumber fraudulent transactions.

The objective is to build and evaluate machine learning models that can effectively identify fraudulent transactions while balancing precision and recall.

## Objectives

- Explore and understand credit card transaction data through Exploratory Data Analysis (EDA).
- Identify patterns and characteristics associated with fraudulent transactions.
- Perform data preprocessing and feature scaling.
- Address class imbalance using appropriate techniques.
- Develop and compare machine learning classification models.
- Evaluate model performance using metrics suitable for imbalanced datasets.
- Improve fraud detection performance through model optimisation and threshold tuning.

## Dataset

**Source:** [Credit Card Fraud Detection – Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains transactions made by European cardholders over two days in September 2013.

Key dataset information:

- Total transactions: 284,807
- Legitimate transactions: 284,315
- Fraudulent transactions: 492
- Features: 30
- Target variable: `Class`

### Feature Description

| Feature | Description |
|---|---|
| Time | Seconds elapsed between the first transaction and each subsequent transaction |
| V1–V28 | Anonymised features obtained through Principal Component Analysis (PCA) |
| Amount | Transaction amount |
| Class | Target variable: 0 = Legitimate, 1 = Fraudulent |

## Project Structure

```text
ML_Fraud_Detection_Kaggle/
│
├── data/
│   └── creditcard.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│
├── reports/
│   └── figures/
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Technologies and Libraries

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Git and GitHub

## Methodology

### 1. Exploratory Data Analysis (EDA)

- Dataset inspection and data type verification.
- Missing value and duplicate record checks.
- Fraudulent versus legitimate transaction distribution.
- Transaction amount and time distribution analysis.
- Feature correlation analysis.

### 2. Data Preprocessing

- Separate independent features and target variable.
- Train-test split using stratified sampling.
- Feature scaling.
- Investigate class imbalance handling techniques, including SMOTE and class weighting.

### 3. Model Development

Planned classification algorithms:

- Logistic Regression
- Random Forest
- Additional classification models where appropriate.

### 4. Model Evaluation

Due to the highly imbalanced nature of the dataset, accuracy alone is insufficient to assess model performance.

The following evaluation metrics will be considered:

- Accuracy
- Precision
- Recall
- F1-score
- Precision-Recall AUC
- Confusion Matrix

Particular attention will be given to Recall, as detecting fraudulent transactions is a key objective.

## Initial Model Results

A baseline Logistic Regression model was developed and evaluated using an 80:20 stratified train-test split.

**Confusion Matrix**

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | 56,836 | 28 |
| Actual Fraud | 26 | 72 |

**Classification Performance**

| Metric | Score |
|---|---:|
| Accuracy | 99.91% |
| Fraud Precision | 72.00% |
| Fraud Recall | 73.47% |
| Fraud F1-score | 72.73% |

### Initial Findings

Although the baseline model achieved 99.91% accuracy, the dataset is highly imbalanced, making accuracy alone potentially misleading.

The model successfully detected 72 out of 98 fraudulent transactions but missed 26 fraudulent transactions.

Further improvements will focus on increasing fraud detection recall while maintaining an acceptable precision level.

## Future Improvements

- Experiment with SMOTE and class weighting.
- Compare Logistic Regression with Random Forest.
- Perform hyperparameter tuning.
- Optimise classification thresholds.
- Evaluate models using Precision-Recall AUC.
- Analyse feature importance and model interpretability.
- Develop a reusable machine learning pipeline.

## Disclaimer

This project is developed for educational and portfolio purposes. The dataset is anonymised and does not represent a live banking fraud detection system.