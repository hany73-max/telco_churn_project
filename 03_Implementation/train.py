import pandas as pd
import os
import joblib
from sklearn.tree import DecisionTreeClassifier

def train_model(x_train_path, y_train_path, model_output_path):
    print("Loading training data...")
    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)
    
    # Initialize the model 
    # (random_state ensures we get the exact same tree every time we run it)
    print("Training the baseline Decision Tree...")
    clf = DecisionTreeClassifier(random_state=42)
    
    # Train (fit) the model
    clf.fit(X_train, y_train)
    
    # Save the trained model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(clf, model_output_path)
    
    print(f"Success! Model trained and saved to: {model_output_path}")

if __name__ == "__main__":
    # Define paths
    X_TRAIN_PATH = 'data/processed/X_train.csv'
    Y_TRAIN_PATH = 'data/processed/y_train.csv'
    MODEL_OUTPUT = 'models/baseline_tree.pkl'
    
    train_model(X_TRAIN_PATH, Y_TRAIN_PATH, MODEL_OUTPUT)