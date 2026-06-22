# Retail Credit Risk Feature Engineering Pipeline

## Project Overview

This project demonstrates how raw financial transaction data can be transformed into a clean, model-ready feature matrix using a production-style scikit-learn pipeline.

Rather than focusing on predictive modelling or hyperparameter tuning, the primary objective is to showcase data engineering, feature engineering, missing value handling, and leakage-free preprocessing techniques commonly used in credit risk analytics.

The pipeline accepts a raw, unprocessed credit portfolio dataset and outputs a standardized numerical feature matrix suitable for downstream machine learning models.

---

## Business Scenario

A financial institution maintains historical records of customer credit card accounts, including balances, payments, income declarations, and credit limits.

The raw dataset contains several common real-world data quality problems:

* Missing reported income values
* Missing credit limit values
* Potential division-by-zero scenarios
* Raw transactional features that do not adequately describe customer behaviour
* Multiple months of historical account activity requiring temporal feature extraction

The goal is to transform these raw records into meaningful risk-oriented features while preventing data leakage and maintaining compatibility with scikit-learn pipelines.

---

## Dataset Structure

Each row represents a single customer account.

### Customer Information

| Feature         | Description                 |
| --------------- | --------------------------- |
| `customer_id`     | Unique customer identifier  |
| `reported_income` | Self-reported annual income |
| `credit_limit`    | Approved credit limit       |

### Historical Balances

| Feature         |
| --------------- |
| `balance_month_1` |
| `balance_month_2` |
| `balance_month_3` |

### Historical Payments

| Feature         |
| --------------- |
| `payment_month_1` |
| `payment_month_2` |
| `payment_month_3` |

### Additional Fields

| Feature       | Description                   |
| ------------- | ----------------------------- |
| `interim_bal_1` | Intermediate balance snapshot |
| `interim_bal_2` | Intermediate balance snapshot |

### Temporal Assumption

The project assumes:

```text
month_1 = oldest observation
month_3 = most recent observation
```

This assumption is used when calculating balance momentum and repayment trends.

---

## Data Quality Challenges

### Missing Reported Income

Missing income values are not treated as purely random omissions.

In consumer lending, failure to disclose income may itself contain behavioural information.

The pipeline therefore:

1. Creates a binary indicator:

```text
income_was_missing
```

2. Imputes missing income values using the training-set median.

This preserves both mathematical completeness and the potential predictive signal contained in the missingness itself.

---

### Missing Credit Limits

Traditional mean or median imputation can generate logically inconsistent credit limits.

For example:

```text
Credit Limit = $10,000
Historical Balance = $25,000
```

Such observations violate basic business constraints because a customer's outstanding balance cannot exceed their approved credit limit.

Instead, missing credit limits are imputed using:

```text
Maximum Historical Balance
```

across the three observed months.

This ensures the imputed limit remains economically plausible.

If all historical balances equal zero, the training-set median credit limit is used as a fallback value.

---

## Feature Engineering

### 1. Credit Utilization

Credit utilization measures borrowing intensity relative to available credit.

Formula:

```text
utilization_month_X =
balance_month_X / credit_limit
```

Higher utilization often signals increased financial stress and elevated credit risk.

Division-by-zero cases are handled safely using conditional logic.

---

### 2. Payment-to-Balance Ratio

Raw payment amounts are difficult to compare across customers with vastly different balances.

A normalized repayment metric is therefore created:

```text
pay_to_bal_ratio_month_X =
payment_month_X / balance_month_X
```

Interpretation:

* 1.0 → Full repayment
* 0.5 → Partial repayment
* 0.0 → No repayment

Accounts with zero balances are assigned a ratio of 1.0 to represent non-delinquent behaviour while avoiding division-by-zero errors.

---

### 3. Balance Momentum

Customer debt trajectories are often more informative than static balances.

The pipeline calculates:

```text
balance_trend =
balance_month_3 - balance_month_1
```

Interpretation:

* Positive values → Increasing debt burden
* Negative values → Declining debt burden

This feature captures the direction and velocity of debt accumulation over time.

---

## Leakage Prevention Strategy

A key design objective is preventing data leakage.

The dataset is split into training and validation sets before any preprocessing occurs.

```python
X_train, X_val = train_test_split(...)
```

During pipeline fitting:

* Training medians are learned exclusively from the training data
* Validation data never contributes to imputation statistics
* Validation records are transformed using previously learned parameters

This ensures realistic model evaluation and mirrors production deployment behaviour.

---

## Pipeline Architecture

The project follows a modular scikit-learn architecture.

```text
Raw Data
    │
    ▼
FinancialFeatureEngineer
    │
    ├── Missing Value Imputation
    ├── Missingness Indicators
    ├── Utilization Ratios
    ├── Repayment Ratios
    └── Balance Momentum
    │
    ▼
ColumnTransformer
    │
    ├── StandardScaler
    └── Binary Feature Passthrough
    │
    ▼
Model-Ready Feature Matrix
```

---

## Project Structure

```text
retail-credit-pipeline/
│
├── data/
│   └── raw_credit_data.csv
│
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   └── transformers.py
│
├── main.py
├── README.md
└── requirements.txt
```

---

## Final Feature Matrix

The final pipeline outputs sixteen features.

### Core Features

* reported_income
* credit_limit
* balance_month_1
* payment_month_1
* balance_month_2
* payment_month_2
* balance_month_3
* payment_month_3

### Engineered Features

* income_was_missing
* balance_trend
* utilization_month_1
* utilization_month_2
* utilization_month_3
* pay_to_bal_ratio_month_1
* pay_to_bal_ratio_month_2
* pay_to_bal_ratio_month_3

All continuous numerical features are standardized using `StandardScaler`.

The binary missingness indicator is passed through unchanged.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn

---

## Key Learning Outcomes

This project demonstrates:

* Custom scikit-learn transformers
* Production-style preprocessing pipelines
* Missing value engineering
* Domain-specific financial feature engineering
* Data leakage prevention
* ColumnTransformer usage
* Feature scaling workflows
* Clean project structuring for machine learning systems

The resulting pipeline can be integrated directly into downstream classification or credit risk modelling workflows.

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/kaushalya-dabare/retail-credit-pipeline.git
cd retail-credit-pipeline
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Execute the Pipeline

```bash
python main.py
```

### Expected Output

The pipeline will:

- Load the raw credit dataset
- Split the data into training and validation sets
- Apply leakage-safe feature engineering
- Standardize numerical features
- Generate the final model-ready feature matrix
- Display the resulting feature names and matrix dimensions

Example:

```text
[SUCCESS] Unified Data Pipeline execution complete.
Final Training Array Shape (400, 16)
Final Validation Array Shape (100, 16)

Final feature columns generated (in exact matrix order):
  [0] num_scaler__reported_income
  [1] num_scaler__credit_limit
  [2] num_scaler__balance_month_1
  [3] num_scaler__payment_month_1
  [4] num_scaler__balance_month_2
  [5] num_scaler__payment_month_2
  [6] num_scaler__balance_month_3
  [7] num_scaler__payment_month_3
  [8] num_scaler__balance_trend
  [9] num_scaler__pay_to_bal_ratio_month_1
  [10] num_scaler__utilization_month_1
  [11] num_scaler__pay_to_bal_ratio_month_2
  [12] num_scaler__utilization_month_2
  [13] num_scaler__pay_to_bal_ratio_month_3
  [14] num_scaler__utilization_month_3
  [15] remainder__income_was_missing
```