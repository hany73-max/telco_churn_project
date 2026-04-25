import pandas as pd
import os
from sklearn.model_selection import train_test_split

def encode_and_split(input_path, output_dir):
    print(f"Loading cleaned data from: {input_path}")
    df = pd.read_csv(input_path)
    
    # ==========================================
    # Step 3: Feature Encoding
    # ==========================================
    print("Encoding features...")
    
    # 1. Map target variable 'Churn' to 1 and 0
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # 2. Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # 3. Apply One-Hot Encoding
    # drop_first=True prevents multicollinearity (the dummy variable trap)
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # ==========================================
    # Step 4: Train / Test Split
    # ==========================================
    print("Splitting data into Train and Test sets...")
    
    # Separate features (X) and target (y)
    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']
    
    # Split the data (80% training, 20% testing)
    # stratify=y ensures both sets maintain the exact 73/27 churner ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # ==========================================
    # Save the splits
    # ==========================================
    os.makedirs(output_dir, exist_ok=True)
    
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    print(f"Success! Features built and splits saved to: {output_dir}")
    print(f"X_train shape: {X_train.shape} | X_test shape: {X_test.shape}")

if __name__ == "__main__":
    # Define input (from Step 2) and output directories
    CLEANED_DATA_PATH = 'data/processed/cleaned_churn_data.csv'
    OUTPUT_DIR = 'data/processed/'
    
    encode_and_split(CLEANED_DATA_PATH, OUTPUT_DIR)