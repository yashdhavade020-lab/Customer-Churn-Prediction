# Customer Churn Prediction using Machine Learning and Streamlit

## Project Overview

Customer churn is one of the most important business challenges faced by subscription-based companies. This project predicts whether a customer is likely to leave a service using Machine Learning techniques.

The model is trained on customer demographic and service usage data and deployed through an interactive Streamlit web application.

---

## Features

- Data Cleaning and Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Class Imbalance Handling using SMOTE
- Machine Learning Model Training
- Hyperparameter Tuning
- Customer Churn Prediction
- Interactive Streamlit Interface

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Imbalanced-Learn (SMOTE)
- Matplotlib
- Seaborn
- Streamlit

---

## Dataset Features

| Feature | Description |
|----------|------------|
| Age | Customer age |
| Gender | Male/Female |
| Tenure | Duration of service |
| MonthlyCharges | Monthly bill amount |
| TotalCharges | Total amount spent |
| InternetService | Service type |
| ContractType | Contract category |
| Churn | Target Variable |

---

## Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Encoding
5. SMOTE Oversampling
6. Train-Test Split
7. Model Training
8. Hyperparameter Tuning
9. Model Evaluation
10. Deployment using Streamlit

---

## Models Evaluated

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

### Best Performing Model

Random Forest Classifier

---

## Project Structure

```text
Customer-Churn-Prediction/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── customer_churn_data.csv
│
├── model/
│   └── randomforest_churn_model.pkl
│
├── notebooks/
│   └── new_nb.ipynb
│
└── images/
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yashdhavade020-lab/Customer-Churn-Prediction.git
```

Navigate into project:

```bash
cd Customer-Churn-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit App:

```bash
streamlit run app.py
```

---

## Future Improvements

- XGBoost Integration
- LightGBM Integration
- Explainable AI (SHAP)
- Cloud Deployment
- Real-Time Prediction API

---

## Author

Yash Dhavade
