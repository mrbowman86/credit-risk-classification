# Credit Risk Analysis Report

## 📑 Table of Contents
- [Overview of the Analysis](#overview-of-the-analysis)
- [Results](#results)
- [Summary](#summary)
- [References](#references)

---

## Overview of the Analysis

The goal of this credit risk analysis was to determine whether a logistic regression model could effectively predict the likelihood that a loan is high-risk (defaulted) or healthy, based on borrower attributes and financial behavior. 

The dataset used (`lending_data.csv`) contains financial data on individual loans. The target variable, `loan_status`, is binary:
- `0`: Healthy loan
- `1`: High-risk loan (likely to default)

### Objective:
- Build and evaluate a **Logistic Regression** model using supervised machine learning.
- Predict loan risk status (`loan_status`) based on the input features.
- Evaluate model effectiveness using accuracy, precision, recall, and a confusion matrix.

### Steps Taken:
1. **Data Preprocessing**:
   - Loaded the dataset into a Pandas DataFrame.
   - Defined `x` (features) by dropping the `loan_status` column.
   - Defined `y` (labels) as the `loan_status` column.
   - Split the data into training and testing sets using `train_test_split`.

2. **Model Creation & Evaluation**:
   - Created a **Logistic Regression** model using Scikit-learn’s `LogisticRegression()`.
   - Fitted the model on the training data (`x_train`, `y_train`).
   - Generated predictions on the test data (`x_test`).
   - Evaluated model performance using:
     - Confusion matrix
     - Classification report (accuracy, precision, recall)

---

## Results

### Logistic Regression Model:

- **Accuracy**: `0.99`
- **Precision**:
  - Class 0 (Healthy): `1.00`
  - Class 1 (High-risk): `0.84`
- **Recall**:
  - Class 0 (Healthy): `0.99`
  - Class 1 (High-risk): `0.94`

> **Confusion Matrix**:
```
[[18655   110]
 [  36  583]]
```

These results show that the model performs **very well** at predicting healthy loans and **reasonably well** at predicting high-risk loans.

---

## Summary

The logistic regression model demonstrates strong overall accuracy (99%) and excels at identifying healthy loans with high precision and recall. 

However, for high-risk loans:
- The **precision** is slightly lower (84%), meaning that about 16% of predicted high-risk loans may be false positives.
- The **recall** is relatively strong (94%), suggesting that the model successfully captures most actual high-risk loans.

### Recommendation:
We recommend this logistic regression model for initial credit risk screening due to its high accuracy and solid recall for high-risk loans. While not perfect, it can assist financial institutions in flagging potentially risky applicants for further manual review or more sophisticated risk assessments.

If higher precision is needed for high-risk loan predictions (to avoid incorrectly flagging healthy loans), consider additional feature engineering, rebalancing techniques, or more advanced models (e.g., Random Forest, XGBoost).

---

## References

- Lending dataset provided by Data Analytics Bootcamp.
- Logistic Regression: [Scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- Model evaluation metrics: [Scikit-learn metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)
- This report was assisted by OpenAI's [ChatGPT](https://openai.com/chatgpt) for formatting and writing support.