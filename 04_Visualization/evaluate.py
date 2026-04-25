import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import os

def evaluate_model(x_test_path, y_test_path, model_path, plot_output_path):
    print("Loading test data and trained model...")
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)
    clf = joblib.load(model_path)
    
    print("Making predictions on unseen data...")
    y_pred = clf.predict(X_test)
    
    # 1. Print Text Metrics
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    # 2. Visualize Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Stayed (0)', 'Churned (1)'])
    disp.plot(cmap='Blues', ax=plt.gca(), colorbar=False)
    
    plt.title('Baseline Decision Tree - Confusion Matrix')
    sns.despine()
    
    # Save the plot
    os.makedirs(os.path.dirname(plot_output_path), exist_ok=True)
    plt.savefig(plot_output_path, bbox_inches='tight')
    print(f"\nConfusion matrix plot saved to: {plot_output_path}")

if __name__ == "__main__":
    # Define paths
    X_TEST_PATH = 'data/processed/X_test.csv'
    Y_TEST_PATH = 'data/processed/y_test.csv'
    MODEL_PATH = 'models/baseline_tree.pkl'
    PLOT_OUTPUT = 'models/baseline_confusion_matrix.png'
    
    evaluate_model(X_TEST_PATH, Y_TEST_PATH, MODEL_PATH, PLOT_OUTPUT)