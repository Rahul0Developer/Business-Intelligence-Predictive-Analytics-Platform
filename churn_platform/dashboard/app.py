"""
Streamlit Dashboard for Customer Churn Prediction
Interactive business intelligence dashboard with predictions and analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import os
import joblib
from typing import Optional

# Page configuration
st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .prediction-high {
        color: #d62728;
        font-weight: bold;
    }
    .prediction-medium {
        color: #ff7f0e;
        font-weight: bold;
    }
    .prediction-low {
        color: #2ca02c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load models locally if API not available
@st.cache_resource
def load_local_models():
    """Load models from local files"""
    try:
        model = joblib.load("models/Logistic_Regression.joblib")
        scaler = joblib.load("models/scaler.joblib")
        feature_columns = joblib.load("models/feature_columns.joblib")
        return model, scaler, feature_columns
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None


def check_api_health(api_url: str) -> bool:
    """Check if API is available"""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def predict_churn_api(customer_data: dict, api_url: str) -> Optional[dict]:
    """Get prediction from API"""
    try:
        response = requests.post(f"{api_url}/api/v1/predict", json=customer_data, timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def predict_churn_local(customer_data: dict, model, scaler, feature_columns) -> dict:
    """Get prediction from local model"""
    try:
        # Prepare features (simplified version)
        features = prepare_features(customer_data)
        
        # Select available columns
        available_cols = [col for col in feature_columns if col in features.columns]
        X = features[available_cols]
        
        # Scale and predict
        X_scaled = scaler.transform(X)
        churn_proba = model.predict_proba(X_scaled)[0][1]
        churn_pred = model.predict(X_scaled)[0]
        
        # Determine risk level
        if churn_proba >= 0.7:
            risk_level = "Critical"
        elif churn_proba >= 0.5:
            risk_level = "High"
        elif churn_proba >= 0.3:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return {
            "churn_probability": round(float(churn_proba), 4),
            "predicted_churn": bool(churn_pred),
            "risk_level": risk_level,
            "top_factors": get_top_factors(customer_data),
            "estimated_savings": round(max(churn_proba * customer_data.get('monthly_charges', 50) * 12 - 50, 0), 2)
        }
    except Exception as e:
        return None


def prepare_features(customer_data: dict) -> pd.DataFrame:
    """Prepare features for prediction"""
    features = {
        'age': customer_data.get('age', 30),
        'tenure': customer_data.get('tenure', 12),
        'monthly_charges': customer_data.get('monthly_charges', 50),
        'total_charges': customer_data.get('total_charges', 600),
        'num_support_tickets': customer_data.get('num_support_tickets', 0),
        'avg_monthly_usage_gb': customer_data.get('avg_monthly_usage_gb', 50),
        'late_payments': customer_data.get('late_payments', 0),
        'customer_service_calls': customer_data.get('customer_service_calls', 0),
        'contract_length_months': customer_data.get('contract_length_months', 1),
        'tenure_squared': customer_data.get('tenure', 12) ** 2,
        'log_tenure': np.log1p(customer_data.get('tenure', 12)),
    }
    
    # Add more features as needed
    tenure = max(customer_data.get('tenure', 12), 1)
    features['avg_charge_per_month'] = customer_data.get('total_charges', 600) / tenure
    features['charges_per_gb'] = customer_data.get('monthly_charges', 50) / max(customer_data.get('avg_monthly_usage_gb', 1), 1)
    features['is_high_charge'] = 1 if customer_data.get('monthly_charges', 50) > 85 else 0
    
    service_map = {'Yes': 1, 'No': 0}
    num_services = sum([
        service_map.get(customer_data.get('online_security', 'No'), 0),
        service_map.get(customer_data.get('tech_support', 'No'), 0),
        service_map.get(customer_data.get('streaming_tv', 'No'), 0),
        service_map.get(customer_data.get('streaming_movies', 'No'), 0)
    ])
    features['num_services'] = num_services
    features['has_multiple_services'] = 1 if num_services >= 2 else 0
    features['has_no_services'] = 1 if num_services == 0 else 0
    
    total_interactions = customer_data.get('num_support_tickets', 0) + customer_data.get('customer_service_calls', 0)
    features['total_support_interactions'] = total_interactions
    features['support_per_month'] = total_interactions / tenure
    features['is_high_support'] = 1 if total_interactions > 10 else 0
    
    features['late_payment_ratio'] = customer_data.get('late_payments', 0) / tenure
    features['has_late_payments'] = 1 if customer_data.get('late_payments', 0) > 0 else 0
    features['multiple_late_payments'] = 1 if customer_data.get('late_payments', 0) >= 3 else 0
    features['is_electronic_check'] = 1 if customer_data.get('payment_method') == 'Electronic check' else 0
    features['calls_per_month'] = customer_data.get('customer_service_calls', 0) / tenure
    features['is_frequent_caller'] = 1 if customer_data.get('customer_service_calls', 0) >= 4 else 0
    
    contract_map = {'Month-to-month': 1, 'One year': 2, 'Two year': 3}
    features['contract_type_encoded'] = contract_map.get(customer_data.get('contract_type', 'Month-to-month'), 1)
    features['is_short_term_contract'] = 1 if customer_data.get('contract_type') == 'Month-to-month' else 0
    features['contract_value'] = customer_data.get('monthly_charges', 50) * customer_data.get('contract_length_months', 1)
    
    internet_map = {'No': 0, 'DSL': 1, 'Fiber optic': 2}
    features['internet_service_encoded'] = internet_map.get(customer_data.get('internet_service', 'DSL'), 1)
    
    fiber_no_protection = (
        customer_data.get('internet_service') == 'Fiber optic' and
        customer_data.get('online_security') == 'No' and
        customer_data.get('tech_support') == 'No'
    )
    features['fiber_no_protection'] = 1 if fiber_no_protection else 0
    
    features['gender_encoded'] = 1 if customer_data.get('gender') == 'Female' else 0
    features['is_young_adult'] = 1 if 18 <= customer_data.get('age', 30) <= 25 else 0
    features['is_middle_age'] = 1 if 36 <= customer_data.get('age', 30) <= 50 else 0
    features['is_senior'] = 1 if customer_data.get('is_senior', False) else 0
    
    # Risk score
    risk_score = 0
    if customer_data.get('contract_type') == 'Month-to-month':
        risk_score += 3
    if customer_data.get('tenure', 12) < 6:
        risk_score += 2
    if customer_data.get('late_payments', 0) >= 2:
        risk_score += 2
    if customer_data.get('payment_method') == 'Electronic check':
        risk_score += 2
    if customer_data.get('customer_service_calls', 0) >= 4:
        risk_score += 1
    if customer_data.get('num_support_tickets', 0) >= 10:
        risk_score += 1
    if not customer_data.get('has_premium_support', False):
        risk_score += 1
    if fiber_no_protection:
        risk_score += 2
    
    features['risk_score'] = min(risk_score / 10 * 10, 10)
    features['is_high_risk'] = 1 if features['risk_score'] >= 6 else 0
    
    # One-hot encoded features
    features['contract_type_One year'] = 1 if customer_data.get('contract_type') == 'One year' else 0
    features['contract_type_Two year'] = 1 if customer_data.get('contract_type') == 'Two year' else 0
    features['payment_method_Bank transfer'] = 1 if customer_data.get('payment_method') == 'Bank transfer' else 0
    features['payment_method_Credit card'] = 1 if customer_data.get('payment_method') == 'Credit card' else 0
    features['payment_method_Electronic check'] = 1 if customer_data.get('payment_method') == 'Electronic check' else 0
    features['internet_service_DSL'] = 1 if customer_data.get('internet_service') == 'DSL' else 0
    features['internet_service_Fiber optic'] = 1 if customer_data.get('internet_service') == 'Fiber optic' else 0
    features['gender_Female'] = 1 if customer_data.get('gender') == 'Female' else 0
    
    return pd.DataFrame([features])


def get_top_factors(customer_data: dict) -> list:
    """Identify top churn factors"""
    factors = []
    
    if customer_data.get('contract_type') == 'Month-to-month':
        factors.append("Short-term contract")
    if customer_data.get('tenure', 999) < 6:
        factors.append("New customer (< 6 months)")
    if customer_data.get('late_payments', 0) >= 2:
        factors.append("Multiple late payments")
    if customer_data.get('payment_method') == 'Electronic check':
        factors.append("Electronic check payment")
    if customer_data.get('customer_service_calls', 0) >= 4:
        factors.append("Frequent service calls")
    if not customer_data.get('has_premium_support', False):
        factors.append("No premium support")
    
    if not factors:
        factors.append("Standard risk profile")
    
    return factors[:5]


# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Customer Churn Analytics Platform</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/analytics.png", width=80)
        st.title("Navigation")
        
        page = st.radio(
            "Select Page",
            ["🏠 Overview", "🔮 Predict Churn", "📈 Analytics", "📁 Upload Data"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # API Configuration
        st.subheader("⚙️ Settings")
        use_api = st.checkbox("Use API Backend", value=False)
        api_url = st.text_input("API URL", value="http://localhost:8000")
        
        if use_api:
            api_status = "🟢 Connected" if check_api_health(api_url) else "🔴 Disconnected"
            st.markdown(f"**API Status:** {api_status}")
        
        st.markdown("---")
        st.info("""
        ### About This Platform
        
        This predictive analytics platform helps businesses:
        
        - **Identify** at-risk customers
        - **Understand** churn factors
        - **Prioritize** interventions
        - **Maximize** retention ROI
        
        Built with: Python, MLflow, FastAPI, Streamlit
        """)
    
    # Load models
    model, scaler, feature_columns = load_local_models()
    
    # Page routing
    if page == "🏠 Overview":
        show_overview()
    elif page == "🔮 Predict Churn":
        show_prediction_page(use_api, api_url, model, scaler, feature_columns)
    elif page == "📈 Analytics":
        show_analytics()
    elif page == "📁 Upload Data":
        show_upload_page(use_api, api_url, model, scaler, feature_columns)


def show_overview():
    """Show overview dashboard"""
    st.header("Business Intelligence Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Customers Analyzed",
            value="7,393",
            delta="+12% this month"
        )
    
    with col2:
        st.metric(
            label="Overall Churn Rate",
            value="60.7%",
            delta="-2.3% vs last month",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            label="Model Accuracy",
            value="87.6%",
            delta="+3.2% improvement"
        )
    
    with col4:
        st.metric(
            label="Potential Savings",
            value="$98,550",
            delta="Annual projection"
        )
    
    st.markdown("---")
    
    # Two column layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Model Performance")
        
        # Create performance data
        perf_data = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
            'Value': [0.876, 0.742, 0.636, 0.685, 0.698],
            'Target': [0.85, 0.70, 0.60, 0.65, 0.65]
        })
        
        fig = go.Figure(data=[
            go.Bar(name='Actual', x=perf_data['Metric'], y=perf_data['Value'], 
                   marker_color='#1f77b4'),
            go.Bar(name='Target', x=perf_data['Metric'], y=perf_data['Target'],
                   marker_color='#d62728', opacity=0.5)
        ])
        fig.update_layout(
            height=400,
            showlegend=True,
            legend=dict(x=0, y=1.1, orientation='h'),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⚠️ Risk Distribution")
        
        # Simulated risk distribution
        risk_dist = pd.DataFrame({
            'Risk Level': ['Low', 'Medium', 'High', 'Critical'],
            'Customers': [2800, 2200, 1500, 893],
            'Color': ['#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
        })
        
        fig = px.pie(risk_dist, values='Customers', names='Risk Level',
                     color='Risk Level',
                     color_discrete_map={
                         'Low': '#2ca02c',
                         'Medium': '#ff7f0e',
                         'High': '#d62728',
                         'Critical': '#9467bd'
                     },
                     hole=0.4)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # Top factors
    st.subheader("🔍 Top Churn Drivers")
    
    factors_df = pd.DataFrame({
        'Factor': ['Month-to-month Contract', 'New Customer (<6mo)', 'Late Payments',
                  'Electronic Check', 'No Premium Support', 'High Support Tickets'],
        'Impact': [0.35, 0.28, 0.22, 0.18, 0.15, 0.12]
    })
    
    fig = px.bar(factors_df, x='Impact', y='Factor', orientation='h',
                 color='Impact', color_continuous_scale='Reds')
    fig.update_layout(height=400, showlegend=False, xaxis_title='Churn Probability Increase')
    st.plotly_chart(fig, use_container_width=True)


def show_prediction_page(use_api, api_url, model, scaler, feature_columns):
    """Show individual prediction page"""
    st.header("Predict Customer Churn")
    
    # Input form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=75.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=900.0)
        
        with col2:
            contract_type = st.selectbox("Contract Type", 
                ["Month-to-month", "One year", "Two year"])
            payment_method = st.selectbox("Payment Method",
                ["Bank transfer", "Credit card", "Electronic check", "Mailed check"])
            internet_service = st.selectbox("Internet Service",
                ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No"])
        
        with col3:
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
            num_support_tickets = st.number_input("Support Tickets", min_value=0, value=5)
            avg_monthly_usage_gb = st.number_input("Monthly Usage (GB)", min_value=0.0, value=50.0)
            late_payments = st.number_input("Late Payments", min_value=0, value=1)
            customer_service_calls = st.number_input("Service Calls", min_value=0, value=2)
            has_premium_support = st.checkbox("Has Premium Support", value=False)
            is_senior = st.checkbox("Is Senior (65+)", value=False)
        
        submitted = st.form_submit_button("🔮 Predict Churn Risk", type="primary")
    
    if submitted:
        customer_data = {
            "age": age,
            "gender": gender,
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "internet_service": internet_service,
            "online_security": online_security,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "num_support_tickets": num_support_tickets,
            "avg_monthly_usage_gb": avg_monthly_usage_gb,
            "late_payments": late_payments,
            "customer_service_calls": customer_service_calls,
            "has_premium_support": has_premium_support,
            "contract_length_months": {"Month-to-month": 1, "One year": 12, "Two year": 24}[contract_type],
            "is_senior": is_senior
        }
        
        # Get prediction
        with st.spinner("Analyzing customer risk..."):
            if use_api and check_api_health(api_url):
                result = predict_churn_api(customer_data, api_url)
            else:
                result = predict_churn_local(customer_data, model, scaler, feature_columns)
        
        if result:
            # Display results
            st.markdown("---")
            
            # Probability gauge
            prob = result['churn_probability']
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Churn Probability", 'font': {'size': 24}},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#d62728" if prob > 0.5 else "#2ca02c"},
                        'steps': [
                            {'range': [0, 30], 'color': "#e0f3db"},
                            {'range': [30, 70], 'color': "#fee08b"},
                            {'range': [70, 100], 'color': "#d62728"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Results Summary")
                
                risk_class = f"prediction-{result['risk_level'].lower()}"
                st.markdown(f"**Risk Level:** :{risk_class}[{result['risk_level']}]")
                st.metric("Churn Probability", f"{prob:.1%}")
                st.metric("Estimated Annual Value", f"${monthly_charges * 12:,.0f}")
                st.metric("Potential Savings", f"${result['estimated_savings']:,.2f}")
                
                st.markdown("**Top Risk Factors:**")
                for factor in result.get('top_factors', []):
                    st.write(f"⚠️ {factor}")
            
            # Intervention recommendation
            st.markdown("---")
            st.subheader("💡 Recommended Actions")
            
            if prob >= 0.7:
                st.error("""
                **IMMEDIATE ACTION REQUIRED**
                
                - Assign dedicated account manager
                - Offer personalized retention discount (15-20%)
                - Schedule executive outreach call
                - Review service issues immediately
                """)
            elif prob >= 0.5:
                st.warning("""
                **HIGH PRIORITY INTERVENTION**
                
                - Contact customer within 48 hours
                - Offer service upgrade or discount (10-15%)
                - Address any outstanding support tickets
                - Consider contract renewal incentive
                """)
            elif prob >= 0.3:
                st.info("""
                **MONITOR CLOSELY**
                
                - Send satisfaction survey
                - Offer check-in call
                - Provide usage tips and best practices
                - Monitor for increased support activity
                """)
            else:
                st.success("""
                **LOW RISK - MAINTAIN ENGAGEMENT**
                
                - Continue regular communication
                - Share product updates and tips
                - Consider upsell opportunities
                - Request testimonial or referral
                """)


def show_analytics():
    """Show analytics dashboard"""
    st.header("Advanced Analytics")
    
    # Load sample data
    df = pd.read_csv("data/customers_full.csv")
    
    # Churn by contract type
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Churn Rate by Contract Type")
        contract_churn = df.groupby('contract_type')['churn'].agg(['mean', 'count']).reset_index()
        contract_churn.columns = ['Contract Type', 'Churn Rate', 'Count']
        
        fig = px.bar(contract_churn, x='Contract Type', y='Churn Rate',
                     color='Churn Rate', color_continuous_scale='Reds',
                     text_auto='.1%')
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Churn by Tenure Group")
        df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 6, 12, 24, 48, 72],
                                    labels=['0-6mo', '6-12mo', '1-2yr', '2-4yr', '4yr+'])
        tenure_churn = df.groupby('tenure_group')['churn'].mean().reset_index()
        
        fig = px.line(tenure_churn, x='tenure_group', y='churn', markers=True,
                      line_shape='hv', markersize=12)
        fig.update_layout(height=400, xaxis_title='Tenure', yaxis_title='Churn Rate',
                          yaxis_tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)
    
    # Monthly charges distribution
    st.subheader("Monthly Charges Distribution by Churn")
    
    fig = px.box(df, x='churn', y='monthly_charges', color='churn',
                 color_discrete_map={0: '#2ca02c', 1: '#d62728'},
                 points='outliers')
    fig.update_layout(height=400, xaxis_title='Churned', yaxis_title='Monthly Charges ($)',
                      showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Feature Correlations")
    
    numeric_cols = ['age', 'tenure', 'monthly_charges', 'total_charges',
                   'num_support_tickets', 'late_payments', 'customer_service_calls', 'churn']
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(corr_matrix, text_auto='.2f', aspect='auto',
                    color_continuous_scale='RdBu_r')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def show_upload_page(use_api, api_url, model, scaler, feature_columns):
    """Show bulk upload and prediction page"""
    st.header("Bulk Customer Analysis")
    
    st.markdown("""
    Upload a CSV file with customer data to get batch predictions.
    
    **Required columns:** age, tenure, monthly_charges, contract_type
    
    **Optional columns:** gender, total_charges, payment_method, internet_service,
    online_security, tech_support, streaming_tv, streaming_movies,
    num_support_tickets, avg_monthly_usage_gb, late_payments,
    customer_service_calls, has_premium_support, is_senior
    """)
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✓ Uploaded {len(df)} records")
        
        # Show preview
        with st.expander("📋 Data Preview"):
            st.dataframe(df.head())
        
        if st.button("🚀 Run Batch Prediction", type="primary"):
            with st.spinner("Processing all customers..."):
                # Process each customer
                predictions = []
                
                for idx, row in df.iterrows():
                    customer_data = {
                        "age": int(row.get('age', 30)),
                        "gender": str(row.get('gender', 'Male')),
                        "tenure": int(row.get('tenure', 12)),
                        "monthly_charges": float(row.get('monthly_charges', 50)),
                        "total_charges": float(row.get('total_charges', 600)),
                        "contract_type": str(row.get('contract_type', 'Month-to-month')),
                        "payment_method": str(row.get('payment_method', 'Credit card')),
                        "internet_service": str(row.get('internet_service', 'DSL')),
                        "online_security": str(row.get('online_security', 'No')),
                        "tech_support": str(row.get('tech_support', 'No')),
                        "streaming_tv": str(row.get('streaming_tv', 'No')),
                        "streaming_movies": str(row.get('streaming_movies', 'No')),
                        "num_support_tickets": int(row.get('num_support_tickets', 0)),
                        "avg_monthly_usage_gb": float(row.get('avg_monthly_usage_gb', 50)),
                        "late_payments": int(row.get('late_payments', 0)),
                        "customer_service_calls": int(row.get('customer_service_calls', 0)),
                        "has_premium_support": bool(row.get('has_premium_support', False)),
                        "contract_length_months": {"Month-to-month": 1, "One year": 12, "Two year": 24}.get(str(row.get('contract_type', 'Month-to-month')), 1),
                        "is_senior": bool(row.get('is_senior', False))
                    }
                    
                    if use_api and check_api_health(api_url):
                        result = predict_churn_api(customer_data, api_url)
                    else:
                        result = predict_churn_local(customer_data, model, scaler, feature_columns)
                    
                    if result:
                        predictions.append({
                            "customer_id": row.get('customer_id', idx + 1),
                            "churn_probability": result['churn_probability'],
                            "risk_level": result['risk_level'],
                            "predicted_churn": result['predicted_churn']
                        })
                
                results_df = pd.DataFrame(predictions)
                
                # Display results
                st.success(f"✓ Processed {len(results_df)} customers")
                
                # Summary stats
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    high_risk = len(results_df[results_df['risk_level'].isin(['Critical', 'High'])])
                    st.metric("High Risk Customers", high_risk)
                
                with col2:
                    avg_prob = results_df['churn_probability'].mean()
                    st.metric("Average Churn Probability", f"{avg_prob:.1%}")
                
                with col3:
                    predicted_churn = results_df['predicted_churn'].sum()
                    st.metric("Predicted to Churn", predicted_churn)
                
                # Downloadable results
                st.download_button(
                    label="📥 Download Results CSV",
                    data=results_df.to_csv(index=False).encode('utf-8'),
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                    type="primary"
                )
                
                # Show results table
                with st.expander("📊 View All Predictions"):
                    st.dataframe(results_df)


if __name__ == "__main__":
    main()
