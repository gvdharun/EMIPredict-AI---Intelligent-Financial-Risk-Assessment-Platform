import streamlit as st
import pandas as pd
import numpy as np
import joblib
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.title("EMIPredict AI Platform")
page = st.sidebar.radio("Navigate", [
    "Home", 
    "Classification Prediction", 
    "Regression Prediction",
    "Data Exploration", 
    "Model Performance Dashboard", 
    "Admin Interface"
])

# Load models and preprocessors (update file paths)
clf = joblib.load('models/xgb_clf.pkl')
reg = joblib.load('models/gbr_model.pkl')

# ----------------------------
# Home
# ----------------------------
if page == "Home":
    st.title("Welcome to EMIPredict AI")
    st.markdown("Comprehensive financial risk assessment platform with ML and real-time analytics.")

# ----------------------------
# Classification Prediction
# ----------------------------
elif page == "Classification Prediction":
    st.header("Predict EMI Eligibility")
    # Collect inputs
    gender = st.selectbox('Gender', ['Female', 'Male', 'female', 'male', 
                                     'M', 'MALE', 'F', 'FEMALE'])
    marital_status = st.selectbox('Marital Status', ['Married', 'Single'])
    education = st.selectbox('Education', ['Professional', 'Graduate', 'High School', 'Post Graduate'])
    employment_type = st.selectbox('Employment Type', ['Private', 'Government', 'Self-employed'])
    company_type = st.selectbox('Company Type', ['Mid-size', 'MNC', 'Startup', 'Large Indian', 'Small'])
    house_type = st.selectbox('House Type', ['Rented', 'Family', 'Own'])
    emi_scenario = st.selectbox('EMI Scenario', ['Personal Loan EMI', 
                                'E-commerce Shopping EMI', 'Education EMI',
                                'Vehicle EMI', 'Home Appliances EMI'])
    credit_risk_score = st.selectbox('Credit Risk Score', ['1', '2', '3', '4'])
    # ... add all relevant inputs
    
    # Prepare input DataFrame for encoding/scaling
    input_dict = {'gender': [gender], 'marital_status': [marital_status], 
                  'education': [education], 'employment_type': [employment_type], 
                  'company_type': [company_type], 'house_type': [house_type], 
                  'emi_scenario': [emi_scenario], 'credit_risk_score': [credit_risk_score] }  # add all used features
    input_df = pd.DataFrame(input_dict)
    
    # Preprocess features
    # Initialize encoders and scalers
    encoder = OneHotEncoder(sparse=False, drop='first')  # drop='first' to avoid dummy variable trap
    scaler = StandardScaler()

    # Fit and transform categorical features
    categorical_cols = ['gender', 'marital_status', 'education', 'employment_type', 
                        'company_type', 'house_type', 'emi_scenario']

    cat_encoded = encoder.fit_transform(input_df[categorical_cols])
    encoded_cat_df = pd.DataFrame(cat_encoded, 
                                  columns=encoder.get_feature_names_out(categorical_cols))
    
    num_scaled = scaler.fit_transform(input_df[['credit_risk_score']])
    scaled_num_df = pd.DataFrame(num_scaled, columns=['credit_risk_score'])

    features = np.hstack([scaled_num_df.values, encoded_cat_df.values])
    
    # Make prediction
    if st.button("Predict EMI Eligibility"):
        st.write("Predicting...")
        result = clf.predict(input_df)
        st.success(f"Prediction: {result[0]}")

# ----------------------------
# Regression Prediction
# ----------------------------
elif page == "Regression Prediction":
    st.header("Predict Max Monthly EMI")

    # Collect inputs
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox('Gender', ['Female', 'Male', 'female', 'male', 
                                     'M', 'MALE', 'F', 'FEMALE'])
    marital_status = st.selectbox('Marital Status', ['Married', 'Single'])
    education = st.selectbox('Education', ['Professional', 'Graduate', 'High School', 'Post Graduate'])
    employment_type = st.selectbox('Employment Type', ['Private', 'Government', 'Self-employed'])
    years_of_employment = st.number_input("Years of Employment", min_value=0.0, max_value=50.0, value=1.0)
    employment_stability_score = st.selectbox("Employment Stability Score", ['1', '2', '3', '4'])
    company_type = st.selectbox('Company Type', ['Mid-size', 'MNC', 'Startup', 'Large Indian', 'Small'])
    house_type = st.selectbox('House Type', ['Rented', 'Family', 'Own'])
    emi_scenario = st.selectbox('EMI Scenario', ['Personal Loan EMI', 
                                'E-commerce Shopping EMI', 'Education EMI',
                                'Vehicle EMI', 'Home Appliances EMI'])
    school_fee = st.number_input("School Fee", min_value=0)
    credit_risk_score = st.selectbox('Credit Risk Score', ['1', '2', '3', '4'])

    # Prepare and preprocess input_df
    input_dict = {'age': [age], 'gender': [gender], 'marital_status': [marital_status], 
                  'education': [education], 'employment_type': [employment_type], 
                  'years_of_employment': [years_of_employment], 'employment_stability_score': [employment_stability_score], 
                  'company_type': [company_type], 'house_type': [house_type], 
                  'emi_scenario': [emi_scenario], 'school_fee': [school_fee], 'credit_risk_score': [credit_risk_score] }
    input_df = pd.DataFrame(input_dict)

    # Preprocess features
    # Initialize encoders and scalers
    encoder = OneHotEncoder(sparse=False, drop='first')  # drop='first' to avoid dummy variable trap
    scaler = StandardScaler()

    # Fit and transform categorical features
    cat_encoded = encoder.fit_transform(input_df.select_dtypes('object'))
    num_scaled = scaler.fit_transform(input_df.select_dtypes(np.number))
    features = np.hstack([num_scaled, cat_encoded])

    # Make prediction with reg.predict(features)
    if st.button("Predict Max Monthly EMI"):
        result = reg.predict(features)
        st.success(f"Prediction: {result[0]}")
    # Display result

# ----------------------------
# Data Exploration
# ----------------------------
elif page == "Data Exploration":
    st.header("Interactive Data Exploration")
    df = pd.read_csv('data/emi_prediction_dataset.csv')  # example path
    st.write("Data Overview:", df.head())
    # Plot EDA charts interactively
    col = st.selectbox('Select column to visualize', df.columns)
    fig, ax = plt.subplots()
    sns.histplot(df[col], ax=ax)
    st.pyplot(fig)

# ----------------------------
# Model Performance Dashboard
# ----------------------------
elif page == "Model Performance Dashboard":
    st.header("Performance Monitoring (MLflow)")
    st.markdown("Experiments, metrics, and comparison charts from MLflow runs.")
    # Optional: Use MLflow API (mlflow.search_runs etc.) to fetch and display charts and metrics

    # Specify your experiment name
    EXPERIMENT_NAME = st.selectbox('Select Experiment', ['EMI Eligibility Prediction', 'Maximum EMI Amount Prediction'])

    # Fetch experiment and runs using MLflow client
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    st.write(f"Selected Experiment: {EXPERIMENT_NAME}")
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])

    # Extract metrics and parameters from the MLflow runs
    metrics_list = []
    for run in runs:
        metrics = run.data.metrics
        params = run.data.params
        run_id = run.info.run_id
        metrics["run_id"] = run_id
        metrics.update(params)
        metrics_list.append(metrics)

    runs_df = pd.DataFrame(metrics_list)

    # Streamlit UI
    st.title("MLflow Model Performance Dashboard")

    # Show metrics table
    st.subheader("Experiment Runs - Metrics & Parameters")
    st.dataframe(runs_df)

    # Select metric to plot
    metric_to_plot = st.selectbox("Select metric for comparison", [m for m in runs_df.columns if m not in ["run_id", "model_type"]])
    if st.button("Show Metric Chart"):
        fig, ax = plt.subplots()
        runs_df.sort_values(by=metric_to_plot, inplace=True)
        ax.bar(runs_df['run_id'], runs_df[metric_to_plot])
        ax.set_xlabel("Run ID")
        ax.set_ylabel(metric_to_plot)
        ax.set_title(f"{metric_to_plot} Comparison across Runs")
        plt.xticks(rotation=90)
        st.pyplot(fig)

    # Optionally add filtering by model type, parameters, etc.
    model_types = runs_df['model_type'].unique() if 'model_type' in runs_df else []
    if model_types:
        selected_model = st.selectbox("Filter by Model Type", model_types)
        filtered_df = runs_df[runs_df['model_type'] == selected_model]
        st.dataframe(filtered_df)



# ----------------------------
# Admin Interface
# ----------------------------
elif page == "Admin Interface":
    st.header("Admin Dashboard - Data Management")
    if st.button("Refresh Data"):
        st.success("Data refreshed.")
    st.markdown("Upload new batch, validate entries, display summary stats, and perform cleaning operations here.")

    # --- Data Upload ---
    st.subheader("Upload New Data Batch")
    uploaded_file = st.file_uploader("Choose a .csv or .xlsx file", type=["csv", "xlsx"])
    if uploaded_file:
        if 'csv' in uploaded_file.name:
            data = pd.read_csv(uploaded_file)
        elif 'xlsx' in uploaded_file.name:
            data = pd.read_excel(uploaded_file)
        st.success("File uploaded successfully!")
        st.write("Preview of uploaded data:", data.head())

        # Option to Save/Replace Existing Database
        if st.button("Save to database"):
            # Example: Save file action; replace with your DB or file writing logic
            data.to_csv("data/latest_data.csv", index=False)
            st.success("Data saved/updated.")

    # --- Data Table Display ---
    st.subheader("Current Data Snapshot")
    try:
        current_df = pd.read_csv("data/latest_data.csv")
        st.dataframe(current_df.head(20))
    except Exception as e:
        st.warning("No current data found. Upload above to initialize.")

    # --- Summary Statistics and Validation ---
    st.subheader("Summary Statistics")
    if 'current_df' in locals():
        st.write(current_df.describe())
        st.write("Missing Values in each column:")
        st.write(current_df.isnull().sum())

    # --- Data Operations ---
    st.subheader("Data Operations")
    if 'current_df' in locals():
        col_to_clean = st.selectbox("Select column to fill missing values", current_df.columns)
        fill_method = st.radio("Fill method", ["Median", "Mode", "Zero"])
        if st.button("Fill Missing Values"):
            if fill_method == "Median":
                median_val = current_df[col_to_clean].median()
                current_df[col_to_clean].fillna(median_val, inplace=True)
            elif fill_method == "Mode":
                mode_val = current_df[col_to_clean].mode()[0]
                current_df[col_to_clean].fillna(mode_val, inplace=True)
            else:
                current_df[col_to_clean].fillna(0, inplace=True)
            # Save updates
            current_df.to_csv("data/latest_data.csv", index=False)
            st.success(f"{col_to_clean} updated with {fill_method}.")

# Note:
# - Fill in regression prediction code and real feature lists for your model.
# - MLflow integration can use `mlflow.get_experiment_by_name`, `mlflow.search_runs`, etc. for live metrics.
# - Use st.experimental_rerun(), st.cache_data, etc. for workflow improvement.
# ----------------------------