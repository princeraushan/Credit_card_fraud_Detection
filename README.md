# Credit Card Fraud Detection

A Machine Learning project that detects fraudulent credit card transactions using **Logistic Regression**. The project uses the Credit Card Fraud Detection dataset from Kaggle and applies data preprocessing, exploratory data analysis, and undersampling to address class imbalance.

## Project Overview

Credit card fraud detection is a binary classification problem in which financial transactions are classified into two categories:

- **0 – Legitimate Transaction:** A normal credit card transaction.
- **1 – Fraudulent Transaction:** A suspicious or fraudulent transaction.

Since fraudulent transactions represent a very small proportion of the overall dataset, the project uses **Under-Sampling** to balance the dataset before training the classification model.

## Objectives

- Analyze credit card transaction data.
- Identify patterns in legitimate and fraudulent transactions.
- Handle imbalanced datasets using undersampling.
- Train a Logistic Regression classification model.
- Evaluate model performance on training and testing datasets.
- Develop a prediction system for identifying fraudulent transactions.

## Technologies & Libraries

| Technology | Purpose |
|---|---|
| Python | Programming language |
| NumPy | Numerical computations |
| Pandas | Data manipulation and analysis |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine Learning and model evaluation |

### Dependencies

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

## Dataset

The project uses the Credit Card Fraud Detection dataset available on Kaggle.

**Dataset:** [Credit Card Fraud Detection – Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains transactions made by European cardholders in September 2013.

- Total transactions: 284,807
- Legitimate transactions: 284,315
- Fraudulent transactions: 492
- Target variable: `Class`

The dataset contains anonymized numerical features, including PCA-transformed variables, along with transaction amount and time.

## Project Workflow

The project follows these major steps:

1. Data Collection
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Handling Imbalanced Data
5. Data Visualization
6. Feature and Target Separation
7. Training and Testing Split
8. Logistic Regression Model Training
9. Model Evaluation
10. Prediction System

## 1. Data Collection

Import the dataset using Pandas.

```python
transaction_dataset = pd.read_csv("creditcard.csv")

transaction_dataset.head(10)
```

## 2. Exploratory Data Analysis

Analyze the dataset to understand its structure and characteristics.

The following operations are performed:

- `shape` – Determine the number of rows and columns.
- `info()` – Examine dataset structure and data types.
- `describe()` – Generate statistical summaries.
- `isnull()` – Identify missing values.
- `value_counts()` – Examine class distribution.
- `dtypes` – Inspect feature data types.

## 3. Separating Legitimate and Fraudulent Transactions

Separate transactions based on the target variable `Class`.

```python
legit = transaction_dataset[
    transaction_dataset.Class == 0
]

fraud = transaction_dataset[
    transaction_dataset.Class == 1
]
```

The original dataset contains a significant imbalance between legitimate and fraudulent transactions.

Compare the average feature values:

```python
transaction_dataset.groupby('Class').mean()
```

## 4. Handling Imbalanced Data – Under-Sampling

Since legitimate transactions significantly outnumber fraudulent transactions, undersampling is applied to create a balanced dataset.

The number of fraudulent transactions is 492.

A random sample of 492 legitimate transactions is selected to create a balanced dataset containing:

- 492 legitimate transactions.
- 492 fraudulent transactions.

This results in a dataset of 984 transactions.

The objective is to prevent the model from becoming biased toward the majority class.

## 5. Data Visualization

Correlation analysis is performed to understand relationships between different features.

A correlation heatmap is generated using Seaborn.

```python
plt.figure(figsize=(20, 11))

sns.heatmap(
    new_transaction_dataset2.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title(
    "Correlation Heatmap of Credit Card Transactions",
    fontsize=22
)

plt.show()
```

The heatmap helps visualize correlations between features and identify potential relationships in the dataset.

## 6. Feature and Target Separation

Separate independent features and the target variable.

```python
X = new_transaction_dataset2.drop(
    columns='Class',
    axis=1
)

Y = new_transaction_dataset2['Class']
```

- `X`: Independent features.
- `Y`: Target variable (`Class`).

## 7. Training and Testing Data Split

Split the dataset into training and testing subsets.

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    stratify=Y,
    random_state=2
)
```

- Training data: 80%
- Testing data: 20%
- Stratification: Maintains class proportions in both subsets.
- Random state: 2

## 8. Model Training – Logistic Regression

Logistic Regression is used because the problem involves binary classification.

```python
model = LogisticRegression()

model.fit(X_train, Y_train)
```

The model learns patterns from the training data to distinguish legitimate transactions from fraudulent ones.

## 9. Model Evaluation

Evaluate the trained model using accuracy score on both training and testing datasets.

```python
X_train_prediction = model.predict(X_train)

training_data_accuracy = accuracy_score(
    Y_train,
    X_train_prediction
)

X_test_prediction = model.predict(X_test)

test_data_accuracy = accuracy_score(
    Y_test,
    X_test_prediction
)

print("Accuracy on Training Data:", training_data_accuracy)

print("Accuracy on Testing Data:", test_data_accuracy)
```

The evaluation helps assess how well the model performs on unseen transactions.

## 10. Prediction System

The trained Logistic Regression model can be used to predict whether a transaction is legitimate or fraudulent.

The prediction output is:

- `0`: Legitimate Transaction
- `1`: Fraudulent Transaction

## Results

The model is evaluated using training and testing accuracy.

*Note: Actual accuracy values should be added after running the complete implementation.*

## Future Improvements

- Implement Random Forest and Decision Tree classifiers.
- Experiment with SMOTE for handling class imbalance.
- Evaluate Precision, Recall, F1-score and ROC-AUC.
- Improve fraud detection using hyperparameter tuning.
- Develop a web-based interface for real-time predictions.

## Repository

**GitHub:** [Credit Card Fraud Detection](https://github.com/princeraushan/Credit_card_fraud_Detection)

## Contributor

**Raushan Kumar Prince**

GitHub: [@princeraushan](https://github.com/princeraushan)

---

If you find this project useful, consider giving the repository a ⭐ on GitHub!