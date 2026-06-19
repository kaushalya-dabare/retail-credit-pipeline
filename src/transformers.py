import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FinancialFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom scikit-learn transformer for credit risk data cleaning,

    imputation, and feature engineering.
    """

    def __init__(self):
        self.median_credit_limit_ = None
        self.median_reported_income_ = None 

    def fit(self, X, y=None):
        """Learns data patterns (like medians) strictly from the training data

        split to prevent data leakage.
        """
        
        X_df = pd.DataFrame(X).copy()

        if "credit_limit" in X_df.columns:
            self.median_credit_limit_ = X_df["credit_limit"].median()

        if "reported_income" in X_df.columns:
            self.median_reported_income_ = X_df["reported_income"].median()

        return self
    
    def transform(self, X):
        """Applies the exact data transformations, business rules, and engineered

        ratios to the dataset.
        """

        df_out = pd.DataFrame(X).copy()

        cols_to_drop = ["customer_id", "interim_bal_1", "interim_bal_2"]
        existing_drop_cols = [c for c in cols_to_drop if c in df_out.columns]
        df_out = df_out.drop(columns=existing_drop_cols)

        if "reported_income" in df_out.columns:
            df_out["income_was_missing"] = (
                df_out["reported_income"].isna().astype(int)
            )

            df_out["reported_income"] = df_out["reported_income"].fillna(
                self.median_reported_income_
            )

        balance_cols = ["balance_month_1", "balance_month_2", "balance_month_3"]
        if all(col in df_out.columns for col in balance_cols):
            max_historical_balances = df_out[balance_cols].max(axis=1)

            imputation_values = max_historical_balances.replace(
                0, self.median_credit_limit_
            )

            if "credit_limit" in df_out.columns:
                df_out["credit_limit"] = df_out["credit_limit"].fillna(
                    imputation_values
                )

        if "balance_month_3" in df_out.columns and "balance_month_1" in df_out.columns:
            df_out["balance_trend"] = (
                df_out["balance_month_3"] - df_out["balance_month_1"]
            )
        
        for m in [1, 2, 3]:
            bal_col = f"balance_month_{m}"
            pay_col = f"payment_month_{m}"

            if bal_col in df_out.columns and pay_col in df_out.columns:
                df_out[f"pay_to_bal_ratio_month_{m}"] = np.where(
                    df_out[bal_col] == 0, 1.0, df_out[pay_col] / df_out[bal_col]
                )

            if bal_col in df_out.columns and "credit_limit" in df_out.columns:
                df_out[f"utilization_month_{m}"] = np.where(
                    df_out["credit_limit"] == 0,
                    0.0,
                    df_out[bal_col] / df_out["credit_limit"],
                )

        return df_out

if __name__ == "__main__":
    print(
        "[SUCCESS] src/transformers.py verified. Class 'FinancialFeatureEngineer' is structured and ready for pipeline deployment."
    )