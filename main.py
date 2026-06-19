import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.transformers import FinancialFeatureEngineer

def main():
    print('INFO: Starting Execution Pipeline')

    data_path = os.path.join('data', 'raw_credit_data.csv')

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f'Could not find the dataset at {data_path}. Ensure the file is placed correctly.'
        )
    
    print(f'INFO: Loading raw data from: {data_path}')
    df = pd.read_csv(data_path)

    print('INFO: Splitting data into 80% Training and 20% Validation sets')
    X_train, X_val = train_test_split(df, test_size=0.20, random_state=42)

    print(f'Training features shape:   {X_train.shape}')
    print(f'Validation features shape: {X_val.shape}')

    print('INFO: Assembling unified ColumnTransformer and Pipeline')
    numeric_selector = lambda df: [
        col for col in df.select_dtypes(include=['float64', 'int64']).columns 
        if col != 'income_was_missing'
    ]

    scaler_transformer = ColumnTransformer(
        transformers=[
            ('num_scaler', StandardScaler(), numeric_selector)
        ],
        remainder='passthrough'
    )

    data_pipeline = Pipeline(
        steps=[
            ('feature_engineering', FinancialFeatureEngineer()),
            ('scaling', scaler_transformer),
        ]
    )

    print('INFO: Executing pipeline.fit_transform() on Training Data')
    X_train_final = data_pipeline.fit_transform(X_train)

    print('INFO: Executing pipeline.transform() on Validation Data')
    X_val_final = data_pipeline.transform(X_val)

    print('\n[SUCCESS] Unified Data Pipeline execution complete.')
    print(f'Final Training Array Shape (NumPy 2D): {X_train_final.shape}')
    print(f'Final Validation Array Shape (NumPy 2D): {X_val_final.shape}')
    fitted_scaler = data_pipeline.named_steps['scaling']
    final_feature_names = fitted_scaler.get_feature_names_out()
    print('\nFinal feature columns generated (in exact matrix order):')
    for idx, col in enumerate(final_feature_names):
        print(f'  [{idx}] {col}')
    print('\nFirst row of fully scaled, model-ready data matrix:')
    print(X_train_final[0])

if __name__ == '__main__':
    main()