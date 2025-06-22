Waze User Churn Prediction & Retention Recommendation System

Project Overview

This project builds an end-to-end churn prediction and retention recommendation pipeline using behavioral data from a navigation app similar to Waze. It combines exploratory data analysis, preprocessing, feature engineering, and machine learning modeling with deployment via Streamlit and tracking using MLflow. The goal is to proactively identify users at risk of churning and recommend actions that could retain them.

Business Problem

In navigation apps like Waze, user engagement and retention are critical. A drop in active users not only affects revenue but also the effectiveness of crowdsourced traffic data. The key business questions were:

Who is likely to churn?

What drives user churn?

What retention strategies are most appropriate for different segments?

Dataset Description

A sample dataset released as part of Google's Advanced Data Analytics certification was used. It contains user-level metrics such as:

Label: Retained or Churned (binary target)

Sessions: App sessions count

Drives: Number of drive sessions

Activity Days: Active days on the app

Driving Days: Days user actually drove using Waze

Fav Navigations: Navigations to saved locations

Distance and Duration: Total distance and driving time

Device: Android or iPhone

Days Since Onboarding: Time since sign-up

Analysis Methodology

Step 1: Exploratory Data Analysis

Found 17.7% churn rate

Imbalance addressed using SMOTE

Key early insight: Users with fewer than 17 activity days per month are more likely to churn

Heavy but inconsistent users often churned, contradicting the idea that more usage always signals loyalty

Step 2: Data Preprocessing

Feature Engineering: Custom ratios like drives per day, distance per session, etc.

Outlier Handling: Applied IQR clipping

Encoding: Label and binary encoding

Scaling: Standard scaling of numeric features

Multicollinearity Reduction: Removed highly correlated variables with low variance

Modular pipeline was built using custom transformer classes

Step 3: Model Training

Models Trained: Logistic Regression (baseline), Random Forest, LightGBM

Metric Used: ROC-AUC (to address imbalance and threshold flexibility)

Best Model: Logistic Regression with ROC-AUC = 0.74

Feature Importance (via SHAP):

Days since onboarding

Activity ratio last month

Navigation patterns

Step 4: Prediction & Deployment

Built an interactive Streamlit app that:

Accepts user data as JSON

Predicts churn risk

Displays top churn drivers

Suggests personalized retention actions

MLflow used to log models, metrics, and artifacts

Key Findings

Churned users often had high intensity but low consistency in usage

Device type was not a significant factor in churn

Retained users averaged 17 monthly active days, churned averaged just 8 days

Primary location usage was more predictive than secondary

Recommendation

Use the model in marketing workflows to:

Intervene early for high-risk users with personalized push notifications

Monitor medium-risk users with targeted content

Invite low-risk users to beta programs or rewards

Technical Implementation

Language: Python 3.9+

Libraries: pandas, numpy, scikit-learn, lightgbm, SHAP, streamlit, mlflow, seaborn, matplotlib

Deployment: Docker + Streamlit

Experiment Tracking: MLflow with SQLite backend

Repository Structure

waze-churn-predictor/
├── data/
│   ├── waze_data.csv
│   ├── raw_feature_names.csv
│   ├── feature_mapping.csv
│   ├── X_train.csv, y_train.csv
│   ├── X_test.csv, y_test.csv
├── models/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_churn_prediction_recom_app.ipynb
├── app.py
├── churn_predictor.py
├── train_model.py
├── Dockerfile
├── requirements.txt
└── README.md

Getting Started

Clone the repo:

git clone https://github.com/yourusername/waze-churn-predictor.git
cd waze-churn-predictor

Install dependencies:

pip install -r requirements.txt

Train the model:

python train_model.py

Launch the app:

streamlit run app.py

Limitations

Dataset covers only ~1 month of activity

40% of sessions occurred in the last month, despite onboarding median being 3.5+ years

Retention patterns could be better understood with longer user timelines

Broader Applications

This churn prediction framework can be adapted for:

E-commerce: customer drop-off

Streaming: viewer disengagement

Healthcare: patient churn

Try It Yourself

Run the Streamlit app locally or deploy via Docker. Test with user-level JSON input and explore churn risk predictions and suggested actions. Feedback welcome!

