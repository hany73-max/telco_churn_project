# Telco Customer Churn Prediction

An end-to-end machine learning pipeline and interactive dashboard designed to forecast customer attrition using a Random Forest ensemble.

## Architecture & Methodology

This repository is structured around a strict four-phase engineering methodology to ensure reproducibility and logical separation of concerns:

1. **`01_Theory/`**: Exploratory data analysis (EDA) and initial statistical evaluations to define the problem scope.
2. **`02_Math/`**: Data preprocessing pipelines, handling of missing values, and one-hot encoding for categorical variables.
3. **`03_Implementation/`**: Core model construction. Utilizes a GridSearchCV-optimized Random Forest classifier tailored for the F1-Score to balance precision and recall on imbalanced target data.
4. **`04_Visualization/`**: Generation of business-ready evaluation metrics, confusion matrices, and feature importance analyses.

## Directory Layout

```text
telco_churn_project/
├── 01_Theory/                     
├── 02_Math/                       
│   ├── data_prep.py                     
│   └── build_features.py                
├── 03_Implementation/             
│   └── train_tuned.py                   
├── 04_Visualization/              
│   └── explain.py                       
├── data/                          
│   ├── raw/                             # <-- Raw CSV data goes here
│   └── processed/                       
├── models/                              # Serialized .pkl models and visual outputs
├── main.py                              # Master pipeline orchestrator
└── app.py                               # Streamlit interactive dashboard
```

Setup & Execution
1. Prerequisites
Ensure you have Python 3.8+ installed. Install the required dependencies:

```Bash
pip install pandas scikit-learn matplotlib seaborn streamlit joblib
```

2. Dataset Setup
Download the dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv) and place it directly into the data/raw/ directory.

3. Pipeline Orchestration
The pipeline is designed for automated execution. Run the master orchestrator from the root directory to clean the data, engineer features, train the model, and generate visualizations:

```Bash
python main.py
```

4. Interactive Dashboard
To explore the dataset, view model architectures, and simulate real-time or batch predictions on unseen data, launch the Streamlit interface:

```Bash
streamlit run app.py
```