import pandas as pd
import os

def clean_data(input_path, output_path):
    print(f"Loading raw data from: {input_path}")
    df = pd.read_csv(input_path)
    
    # 1. Fix Data Types
    # TotalCharges is a string because of blank spaces. Force it to numeric.
    # errors='coerce' turns the blank spaces into actual NaN (missing) values.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # 2. Handle Missing Values
    missing_count = df['TotalCharges'].isnull().sum()
    print(f"Found {missing_count} missing values in TotalCharges. Dropping those rows...")
    df.dropna(inplace=True)
    
    # 3. Drop Useless Features
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
        print("Dropped 'customerID' column.")
        
    # 4. Save Processed Data
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Clean data saved to: {output_path}")
    print(f"New dataset shape: {df.shape}")

if __name__ == "__main__":
    # Remove the ../ so it reads exactly like this:
    RAW_DATA = 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    PROCESSED_DATA = 'data/processed/cleaned_churn_data.csv'
    
    clean_data(RAW_DATA, PROCESSED_DATA)