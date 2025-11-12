# EMIPredict AI - Financial Risk Assessment Platform  📊

---

## 🚀 Project Overview

EMIPredict AI is a comprehensive Streamlit-based web application for financial risk assessment featuring machine learning models for loan eligibility classification and EMI amount regression. The platform provides real-time prediction, interactive data exploration, model monitoring with MLflow, and administrative data management.

---

## 🔑 Features

- 🔹 **Multi-page Streamlit Application** for an intuitive, user-friendly interface  
- 🔹 **Real-time Predictions** for classification (EMI eligibility) and regression (max monthly EMI) tasks  
- 🔹 **Interactive Data Visualization** using Seaborn and Matplotlib  
- 🔹 **Model Performance Dashboard** integrated with MLflow for experiment tracking  
- 🔹 **Admin Interface** for batch data uploads, cleaning, and management  
- 🔹 **Cloud Deployment Ready**: Supports Streamlit Cloud with automated GitHub CI/CD pipeline  
- 🔹 **Responsive Design** for cross-platform device accessibility  

---

## 🛠️ Technologies & Tools

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,streamlit,pandas,sklearn,mlflow,git,github" alt="Technologies" />
</p>

---

## 📁 Repository Structure

```
├── data/                       # Dataset storage
├── models/                     # Serialized ML models and preprocessing objects
├── mlflow results/             # mlflow visualizaton
├── Streamlit Output/           # Streamlit interface
├── EMI_prediction_app.py       # Main Streamlit multi-page app
├── EMI Prediction.ipynb/       # Data preprocessing, training scripts
├── mlruns/                     # MLflow tracking directory
└── README.md                   # Project documentation
```


---

## ⚙️ Getting Started

1. Clone the repository  
  `git clone https://github.com/yourusername/emipredict-ai.git`
  `cd emipredict-ai`

2. Run the Streamlit app  
  `streamlit run EMI_prediction_app.py`

3. Access MLflow UI for experiment tracking (optional)  
  `mlflow ui`

---

## 📊 Model Development & Monitoring

- Models explored: Linear Regression, Random Forest Regressor, Gradient Boosting Regressor  
- Best performing: Gradient Boosting Regressor with RMSE 0.8691, MAE 0.6218, R² 0.2381  
- MLflow integration enables experiment tracking, version control, and performance comparison  

---

## 🛡️ Error Handling & User Feedback

- Comprehensive validation of user inputs  
- Graceful error management with descriptive feedback  
- Real-time updates during prediction and data operations  

---

## ☁️ Deployment

- Deployed on Streamlit Cloud with automated CI/CD from GitHub  
- Responsive and mobile-friendly design  
- Easy scalability with zero-config cloud hosting  

---

## Conclusion ✨

Thank you for exploring the EMIPredict AI project! 🚀

This platform combines advanced machine learning, interactive data visualization, and modern deployment technologies into an end-to-end financial risk assessment tool. The multi-page Streamlit app empowers users with real-time predictions, rich exploratory analytics, and admin controls for seamless data management.  

Integration with MLflow provides transparency and controlled experiment tracking, ensuring reproducibility and performance monitoring. Cloud deployment ensures that the application is accessible, scalable, and responsive across devices.

© 2025 EMIPredict AI Project  
