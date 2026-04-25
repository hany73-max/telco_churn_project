import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# --- Page Configuration ---
st.set_page_config(page_title="Churn Prediction App", layout="wide")

# --- Load Assets ---
@st.cache_data
def load_data():
    raw_df = pd.read_csv('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')
    return raw_df, X_test, y_test

@st.cache_resource
def load_model():
    return joblib.load('models/tuned_tree.pkl')

# Helper function to process new raw data so the model can understand it
def preprocess_input(input_df, expected_columns):
    # 1. Handle TotalCharges
    if 'TotalCharges' in input_df.columns:
        input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce').fillna(0)
    
    # 2. Drop ID if present
    if 'customerID' in input_df.columns:
        input_df = input_df.drop('customerID', axis=1)
        
    # 3. One-hot encode the same way we did in training
    cat_cols = input_df.select_dtypes(include=['object']).columns.tolist()
    encoded_df = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)
    
    # 4. ALIGNMENT: Ensure the new data has the exact same 30 columns as the training data
    # (If a category is missing in the new data, it fills the column with 0)
    encoded_df = encoded_df.reindex(columns=expected_columns, fill_value=0)
    
    return encoded_df

# Load everything
raw_df, X_test, y_test = load_data()
model = load_model()
expected_cols = X_test.columns # We need this to align new data

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", [
    "1. Data Explorer", 
    "2. Model Insights & Viz", 
    "3. Customer Simulator",
    "4. Testing Lab (New Data)"
])

# ==========================================
# PAGE 1: Data Explorer
# ==========================================
if page == "1. Data Explorer":
    st.title("📊 Raw Data Explorer")
    st.write("Explore the original, pre-processed Telco dataset.")
    st.dataframe(raw_df.head(100))
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Shape")
        st.write(f"Total Rows: {raw_df.shape[0]}")
        st.write(f"Total Columns: {raw_df.shape[1]}")
    with col2:
        st.subheader("Churn Distribution")
        st.bar_chart(raw_df['Churn'].value_counts(), color="#ff4b4b")

# ==========================================
# PAGE 2: Model Insights & Viz
# ==========================================
elif page == "2. Model Insights & Viz":
    st.title("🧠 Model Architecture & Visualizations")
    
    # Generate Predictions for the test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Probability Distribution"])
    
    with tab1:
        st.subheader("Top Churn Drivers")
        img_path = 'models/feature_importance.png'
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
            
    with tab2:
        st.subheader("Confusion Matrix (Test Data)")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=['Predicted Stay', 'Predicted Churn'],
                    yticklabels=['Actual Stay', 'Actual Churn'])
        st.pyplot(fig)
        
    with tab3:
        st.subheader("Model Confidence (Probability Distribution)")
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        sns.histplot(y_prob, bins=20, kde=True, color='purple', ax=ax2)
        ax2.set_title("Distribution of Churn Probabilities")
        ax2.set_xlabel("Probability of Churning")
        st.pyplot(fig2)

# ==========================================
# PAGE 3: Customer Simulator
# ==========================================
elif page == "3. Customer Simulator":
    st.title("🎯 Live Customer Prediction")
    st.write("Pull a random customer from our unseen test data and predict their churn risk.")
    
    if st.button("Simulate Random Customer"):
        sample_idx = X_test.sample(1).index[0]
        customer_data = X_test.iloc[[sample_idx]]
        actual_churn = y_test.iloc[sample_idx].values[0]
        
        prediction = model.predict(customer_data)[0]
        probability = model.predict_proba(customer_data)[0][1] * 100
        
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Actual Status", "Churned" if actual_churn == 1 else "Stayed")
        with col2:
            st.metric("Model Prediction", "Will Churn" if prediction == 1 else "Will Stay")
        with col3:
            st.metric("Churn Risk (Probability)", f"{probability:.1f}%")
            
        st.subheader("Customer Profile Snapshot")
        st.json({"Tenure (Months)": float(customer_data['tenure'].values[0]), 
                 "Monthly Charges ($)": float(customer_data['MonthlyCharges'].values[0]), 
                 "Total Charges ($)": float(customer_data['TotalCharges'].values[0])})

# ==========================================
# PAGE 4: Testing Lab (New Data)
# ==========================================
elif page == "4. Testing Lab (New Data)":
    st.title("🧪 Model Testing Lab")
    st.write("Test the model against entirely new data or manual scenarios.")
    
    test_type = st.radio("Choose testing method:", ["Manual Single Entry", "Upload CSV Batch"])
    
    # --- MANUAL ENTRY ---
    if test_type == "Manual Single Entry":
        st.subheader("Enter Customer Details")
        st.info("Adjust the top drivers to see how they impact the prediction.")
        
        with st.form("manual_input_form"):
            col1, col2 = st.columns(2)
            with col1:
                tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
                total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)
            with col2:
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
                payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            
            submit = st.form_submit_button("Predict Churn")
            
            if submit:
                # Create a raw dataframe from the inputs (filling missing fields with basic defaults)
                input_dict = {
                    'tenure': [tenure], 'MonthlyCharges': [monthly_charges], 'TotalCharges': [total_charges],
                    'Contract': [contract], 'InternetService': [internet], 'PaymentMethod': [payment],
                    'gender': ['Male'], 'SeniorCitizen': [0], 'Partner': ['No'], 'Dependents': ['No'],
                    'PhoneService': ['Yes'], 'MultipleLines': ['No'], 'OnlineSecurity': ['No'],
                    'OnlineBackup': ['No'], 'DeviceProtection': ['No'], 'TechSupport': ['No'],
                    'StreamingTV': ['No'], 'StreamingMovies': ['No'], 'PaperlessBilling': ['Yes']
                }
                custom_df = pd.DataFrame(input_dict)
                
                # Run the preprocessing pipeline
                processed_input = preprocess_input(custom_df, expected_cols)
                
                # Predict
                pred = model.predict(processed_input)[0]
                prob = model.predict_proba(processed_input)[0][1] * 100
                
                st.success(f"**Prediction:** {'Churn' if pred == 1 else 'Stay'}")
                st.warning(f"**Churn Risk:** {prob:.1f}%")

    # --- CSV BATCH UPLOAD ---
    elif test_type == "Upload CSV Batch":
        st.subheader("Upload a new dataset")
        uploaded_file = st.file_uploader("Upload a CSV formatted like the original dataset", type=['csv'])
        
        if uploaded_file is not None:
            new_data = pd.read_csv(uploaded_file)
            st.write(f"Loaded {new_data.shape[0]} rows.")
            
            if st.button("Run Batch Prediction"):
                # Run the preprocessing pipeline
                processed_data = preprocess_input(new_data, expected_cols)
                
                # Predict
                predictions = model.predict(processed_data)
                probabilities = model.predict_proba(processed_data)[:, 1]
                
                # Attach to results
                results_df = new_data.copy()
                results_df['Predicted_Churn'] = ['Yes' if p == 1 else 'No' for p in predictions]
                results_df['Churn_Probability'] = probabilities
                
                st.success("Predictions complete!")
                st.dataframe(results_df[['customerID', 'Predicted_Churn', 'Churn_Probability', 'Contract', 'tenure']].head(50))
                
                # Download button
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Full Predictions", csv, "batch_predictions.csv", "text/csv")