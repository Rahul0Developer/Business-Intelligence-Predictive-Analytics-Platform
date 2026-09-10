# Predictive Business Intelligence & ML Operations Platform

## 📊 Customer Churn Prediction System

A complete end-to-end machine learning platform for predicting customer churn, designed specifically for production deployment and business impact.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.5+-orange.svg)

---

## 🎯 Business Problem

This platform answers critical business questions:
- **Which customers are likely to churn?** → Probability scores for each customer
- **What factors influence churn?** → Interpretable risk factor analysis
- **How can the business prioritize intervention?** → Risk-based segmentation with ROI calculations

---

## 🏗️ Architecture

```
                RAW DATA (CSV/PostgreSQL)
                   │
                   ▼
          ┌─────────────────┐
          │ Data Ingestion  │
          │   Generator     │
          └────────┬────────┘
                   ▼
          ┌─────────────────┐
          │ Data Cleaning   │
          │ & Validation    │
          └────────┬────────┘
                   ▼
          ┌─────────────────┐
          │ Feature         │
          │ Engineering     │
          │ (30+ features)  │
          └────────┬────────┘
                   ▼
          ┌─────────────────┐
          │ ML Training     │
          │ Scikit-learn    │
          └────────┬────────┘
                   ▼
          ┌─────────────────┐
          │ MLflow Tracking │
          │ Experiments     │
          └────────┬────────┘
                   ▼
          ┌─────────────────┐
          │ FastAPI         │
          │ Prediction API  │
          └────────┬────────┘
                   ▼
             Docker
                   │
                   ▼
        ┌──────────┴──────────┐
        ▼                     ▼
   Streamlit           Monitoring
   Dashboard            + Drift
```

---

## ✨ Key Features

### 🔬 Machine Learning
- **3 Trained Models**: Logistic Regression, Random Forest, Gradient Boosting
- **Best Model**: Logistic Regression with 87.6% accuracy
- **30+ Engineered Features**: Behavioral, engagement, and risk-based features
- **MLflow Integration**: Full experiment tracking and model registry

### 🚀 Production API
- **6 REST Endpoints**: Health checks, predictions, batch processing, file upload
- **Business Metrics**: Risk scoring, factor identification, cost calculations
- **Auto-documentation**: Interactive Swagger UI at `/docs`

### 📊 Interactive Dashboard
- **4 Pages**: Overview, Predictions, Analytics, Bulk Upload
- **Visualizations**: Plotly charts, gauges, heatmaps
- **Actionable Insights**: Intervention recommendations by risk level

### 💰 Business Impact
- **Potential Savings**: $98,550 identified through targeted interventions
- **Risk Prioritization**: Critical/High/Medium/Low segmentation
- **ROI Calculation**: Estimated savings per customer retention

---

## 📁 Project Structure

```
churn_platform/
├── data/                      # Datasets
│   ├── customers_full.csv     # 7,393 customers (full dataset)
│   ├── customers_cleaned.csv  # Cleaned data
│   ├── customers_engineered.csv # With 30+ features
│   └── customers_sample.csv   # Sample for testing
│
├── src/                       # Core Python modules
│   ├── data_generator.py      # Synthetic data generation
│   ├── data_cleaning.py       # Cleaning & validation
│   ├── feature_engineering.py # Feature creation
│   └── model_training.py      # ML training with MLflow
│
├── api/                       # FastAPI backend
│   └── main.py                # REST API with 6 endpoints
│
├── dashboard/                 # Streamlit frontend
│   └── app.py                 # Interactive dashboard
│
├── models/                    # Trained models
│   ├── Logistic_Regression.joblib
│   ├── Random_Forest.joblib
│   ├── Gradient_Boosting.joblib
│   ├── scaler.joblib
│   └── feature_columns.joblib
│
├── configs/                   # Configuration files
├── mlruns/                    # MLflow experiments
├── Dockerfile                 # Container configuration
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip
```

### Installation
```bash
# Clone repository
cd churn_platform

# Install dependencies
pip install -r requirements.txt
```

### Run Complete Pipeline
```bash
# 1. Generate data
python src/data_generator.py

# 2. Clean data
python src/data_cleaning.py

# 3. Engineer features
python src/feature_engineering.py

# 4. Train models
python src/model_training.py
```

### Start API Server
```bash
cd churn_platform
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Access API docs at: `http://localhost:8000/docs`

### Start Dashboard
```bash
streamlit run dashboard/app.py
```

Access dashboard at: `http://localhost:8501`

---

## 📊 Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Customers | 7,393 |
| Churn Rate | 60.7% |
| Features Engineered | 30+ |
| Contract Types | 3 |
| Payment Methods | 4 |
| Internet Services | 3 |

---

## 🎯 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Net Savings |
|-------|----------|-----------|--------|----------|---------|-------------|
| **Logistic Regression** | 87.6% | 74.2% | 63.6% | 68.5% | 69.8% | $275,550 |
| Random Forest | 86.3% | 73.5% | 61.7% | 67.1% | 69.0% | $267,000 |
| Gradient Boosting | 86.7% | 70.4% | 77.7% | 73.9% | 68.6% | $334,350 |

---

## 🔍 Top Churn Drivers

1. **Month-to-month Contract** (+35% churn probability)
2. **New Customer (<6 months)** (+28%)
3. **Multiple Late Payments** (+22%)
4. **Electronic Check Payment** (+18%)
5. **No Premium Support** (+15%)
6. **High Support Ticket Volume** (+12%)

---

## 🛠️ Technologies Used

### Core Stack
- **Python 3.8+**: Primary language
- **Pandas/NumPy**: Data manipulation
- **Scikit-learn**: Machine learning
- **MLflow**: Experiment tracking
- **FastAPI**: REST API
- **Streamlit**: Dashboard
- **Plotly**: Visualizations

### Deployment Ready
- **Docker**: Containerization
- **PostgreSQL**: Database support
- **Azure/AWS**: Cloud deployment ready
- **PySpark**: Big data processing (extensible)

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/api/v1/model-info` | GET | Model metadata |
| `/api/v1/predict` | POST | Single prediction |
| `/api/v1/predict/batch` | POST | Batch predictions |
| `/api/v1/upload-predict` | POST | CSV upload |
| `/api/v1/statistics` | GET | Analytics data |

---

## 💼 Business Use Cases

### 1. Customer Retention Team
- Identify high-risk customers daily
- Prioritize outreach based on risk level
- Track intervention effectiveness

### 2. Product Management
- Understand features driving churn
- A/B test retention strategies
- Monitor segment-level metrics

### 3. Executive Leadership
- View overall churn trends
- Calculate ROI of retention programs
- Make data-driven decisions

---

## 📈 Future Enhancements

- [ ] Real-time data streaming with Kafka
- [ ] Automated retraining pipeline
- [ ] Data drift detection
- [ ] A/B testing framework
- [ ] Integration with CRM systems
- [ ] PySpark for big data processing
- [ ] Cloud deployment (Azure/AWS)

---

## 👨‍💻 Author

Built as a comprehensive demonstration of end-to-end ML operations for business intelligence.

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🎓 Interview Preparation

This project demonstrates competency in:

✅ **Data Engineering**: ETL pipelines, data cleaning, feature engineering  
✅ **Machine Learning**: Model training, evaluation, selection  
✅ **MLOps**: MLflow tracking, model versioning, deployment  
✅ **Backend Development**: FastAPI, REST APIs, documentation  
✅ **Frontend Development**: Streamlit dashboards, interactive visualizations  
✅ **Business Acumen**: ROI calculation, prioritization frameworks  
✅ **Cloud Readiness**: Docker containerization, deployment patterns  

Perfect for roles in: Data Science, ML Engineering, Business Intelligence, Analytics Engineering
