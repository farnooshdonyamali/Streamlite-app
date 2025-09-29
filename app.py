import streamlit as st 
import pandas as pd
import numpy as np
import plotly.express as px 
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selction import train_test_slpit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(
    page_title = "🔍 Fraud Detection System",
    page_icon = "🔍",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

st.markdown("""
<style>
.main_header{
    font_size: 3 rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)


class FraudDetectionApp:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
    
    def generate_sample_data(self, n_samples= 1000):
        np.random.seed(42) 
        
        #fiancial features
        amount = np.random.lognormal(3,2, n_samples)  
        amount = np.clip(amount, 1, 1000)

        #date features
        hour = np.random.randint(0, 24, n_samples)
        day_of_week = np.random.randint(0, 7, n_samples)

        #account features
        account_age = np.random.exponential(365, n_samples)
        prev_transactions = np.random.poission(5, n_samples)

        #local features
        merchant_risk_score = np.random.beta(2, 5, n_samples)

        #transactional features
        is_weekend = (day_of_week >= 5).astype(int)
        is_night = ((hour >= 22) | (hour <= 6)).astype(int)

        #night and weekends transactions have the most probability for fraud
        fraud_prob = (
            0.02 + #base probability
            0.05 * (amount > np.percentile(amount, 95)) +
            0.03 * is_night + 
            0.02 * is_weekend +
            0.04 * (merchant_risk_score > 0.8) + 
            0.03 * (account_age < 30)

        )
        is_fraud = np.random.binomial(1, fraud_prob, n_samples)

        df = pd.DataFrame({
            'amount': amount,
            "hour": hour,
            "day_of_week": day_of_week,
            "account_age": account_age,
            'prev_transactions': prev_transactions,
            'merchant_risk_score': merchant_risk_score,
            'is_weekend': is_weekend,
            'is_night': is_night,
            'is_fraud': is_fraud

        })
        return df
    def train_model(self, data):
        #training our model
        X = data.drop("is_fraud", axis= 1)
        y = data["is_fraud"]

        X_train, X_test, y_train, y_test = train_test_slpit(
            X, y, test_size= 0.2, random_state= 42, stratify = y 
        )

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # model training
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(X_train_scaled, y_train)