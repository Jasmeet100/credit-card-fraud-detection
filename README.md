# Credit Card Fraud Detection

A machine learning project that detects potentially fraudulent credit card transactions using an imbalanced synthetic transaction dataset. The project compares Logistic Regression, Random Forest, and XGBoost, evaluates performance with precision-recall-focused metrics, and uses SHAP to interpret the XGBoost model.

## Project Overview

Fraud detection is an imbalanced classification problem: fraudulent transactions are much less common than legitimate transactions. Accuracy alone can therefore be misleading. This project focuses on fraud-class precision and recall, Average Precision (AP), and the trade-off between missed fraud and false alarms.

## Models

- Logistic Regression with class weighting
- Random Forest with class weighting
- XGBoost with `scale_pos_weight` and early stopping

## Cross-Validation Results

Five-fold Stratified Cross-Validation was run using Average Precision (AP).

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean AP | Std. dev. |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.3630 | 0.4205 | 0.4018 | 0.3866 | 0.4037 | 0.3951 | 0.0193 |
| Random Forest | 0.3704 | 0.4135 | 0.4821 | 0.4208 | 0.4202 | 0.4214 | 0.0356 |
| XGBoost | 0.4552 | 0.5145 | 0.4879 | 0.4541 | 0.5308 | **0.4885** | 0.0308 |

XGBoost achieved the highest mean Average Precision across the five folds.

**Evaluation note:** Encoding, imputation, and scaling were fitted on the full training split before cross-validation. Consequently, the cross-validation scores are supplementary comparisons and may be somewhat optimistic; they are not a fully fold-isolated estimate. The final test split was held out from model fitting and threshold selection.

## Final Test-Set Evaluation — XGBoost

The final XGBoost evaluation used a probability threshold of **0.80**.

| Metric | Fraud class (1) |
|---|---:|
| Precision | 0.3015 |
| Recall | 0.7567 |
| F1-score | 0.4311 |
| Support | 300 |

Additional metrics:

- **Average Precision:** 0.4870
- **ROC-AUC:** 0.9767
- **Accuracy:** 0.9700

Confusion matrix (rows = actual class, columns = predicted class):

| | Predicted legitimate | Predicted fraud |
|---|---:|---:|
| Actual legitimate | 19,174 | 526 |
| Actual fraud | 73 | 227 |

At this threshold, the model identified 227 of 300 fraudulent transactions and missed 73. It also flagged 526 legitimate transactions as potentially fraudulent. This illustrates the operational trade-off between catching fraud and generating false alerts.

## SHAP Explainability

The SHAP beeswarm plot summarizes how feature values influence the XGBoost model output. In the generated sample, the most influential features include:

1. `failed_attempts_last_24h`
2. `amount_to_customer_avg`
3. `merchant_risk_score`
4. `distance_from_home_km`
5. `device_trust_score`

Higher values of the first four features generally push predictions toward the fraud class in the displayed sample, while higher `device_trust_score` generally pushes predictions away from fraud. SHAP describes model behaviour; it does not establish causation.

[SHAP Summary Plot](shap_summary.png)

## Repository Structure

```text
credit-card-fraud-detection/
├── data/
│   ├── improved_synthetic_fraud_transactions.csv
│   └── .gitkeep
├── fraud_detection_backend.ipynb
├── README.md
├── data_README.txt
├── requirements.txt
└── shap_summary.png
```

## Setup and Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/Jasmeet100/credit-card-fraud-detection.git
   ```

2. Navigate to the project directory:

   ```bash
   cd credit-card-fraud-detection
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Open `fraud_detection_backend.ipynb` in Jupyter Notebook or VS Code.

5. Run the notebook cells.

**Dataset:** The synthetic dataset is included in the repository under the `data/` directory. No separate download is required.

## Tools and Libraries

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, SHAP, and Jupyter Notebook.
