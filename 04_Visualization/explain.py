import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_feature_importance(x_train_path, model_path, plot_output_path):
    print("Loading model and feature names...")
    # We only need X_train to get the column (feature) names
    X_train = pd.read_csv(x_train_path)
    feature_names = X_train.columns
    
    # Load the tuned model
    clf = joblib.load(model_path)
    
    print("Extracting feature importances...")
    # Extract importances and map them to the column names
    importances = clf.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort them to find the top 10 most influential factors
    top_10_features = importance_df.sort_values(by='Importance', ascending=False).head(10)
    print("\n--- Top 10 Drivers of Customer Churn ---")
    print(top_10_features)
    
    # Plot the results
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Importance', y='Feature', data=top_10_features, hue='Feature', palette='viridis', legend=False)    
    plt.title('Top 10 Feature Importances - Tuned Decision Tree')
    plt.xlabel('Relative Importance')
    plt.ylabel('Feature')
    sns.despine()
    
    # Save the plot
    os.makedirs(os.path.dirname(plot_output_path), exist_ok=True)
    plt.savefig(plot_output_path, bbox_inches='tight')
    print(f"\nFeature importance plot saved to: {plot_output_path}")

if __name__ == "__main__":
    X_TRAIN_PATH = 'data/processed/X_train.csv'
    MODEL_PATH = 'models/tuned_tree.pkl'
    PLOT_OUTPUT = 'models/feature_importance.png'
    
    plot_feature_importance(X_TRAIN_PATH, MODEL_PATH, PLOT_OUTPUT)