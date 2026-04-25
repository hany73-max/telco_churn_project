import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

def tune_and_train_rf(x_train_path, y_train_path, x_test_path, y_test_path, model_output_path):
    print("Loading data...")
    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()

    # 1. Define the parameters for the Random Forest
    param_grid = {
        'n_estimators': [100, 200],               # Number of trees in the forest
        'max_depth': [5, 10, 15],                 # Depth of each tree
        'min_samples_split': [10, 50],
        'class_weight': ['balanced']              # Still helping with class imbalance
    }

    print("Running Grid Search for Random Forest (Testing multiple tree architectures)...")
    
    # 2. Initialize Random Forest & Grid Search
    # scoring='f1' forces the model to balance Precision and Recall, fixing our accuracy drop
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)
    
    # 3. Fit it
    grid_search.fit(X_train, y_train)
    
    # 4. Evaluate the best ensemble
    best_model = grid_search.best_estimator_
    print(f"\nBest Parameters Found:\n{grid_search.best_params_}")
    
    y_pred = best_model.predict(X_test)
    print("\n--- Random Forest Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    # 5. Save the upgraded model (keeping the same filename so explain.py still works)
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(best_model, model_output_path)
    print(f"\nUpgraded Random Forest saved successfully to: {model_output_path}")

if __name__ == "__main__":
    X_TRAIN = 'data/processed/X_train.csv'
    Y_TRAIN = 'data/processed/y_train.csv'
    X_TEST  = 'data/processed/X_test.csv'
    Y_TEST  = 'data/processed/y_test.csv'
    MODEL_OUTPUT = 'models/tuned_tree.pkl'
    
    tune_and_train_rf(X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, MODEL_OUTPUT)