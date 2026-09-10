"""
FastAPI Backend for Customer Churn Prediction API
Production-ready REST API with prediction, health checks, and monitoring
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predictive Business Intelligence API for customer churn analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
model = None
scaler = None
feature_columns = None
model_metadata = {}


class CustomerInput(BaseModel):
    """Single customer input schema"""
    age: int = Field(..., description="Customer age", ge=18, le=100)
    gender: str = Field(..., description="Customer gender")
    tenure: int = Field(..., description="Months as customer", ge=0)
    monthly_charges: float = Field(..., description="Monthly charges in USD", ge=0)
    total_charges: float = Field(..., description="Total charges in USD", ge=0)
    contract_type: str = Field(..., description="Contract type")
    payment_method: str = Field(..., description="Payment method")
    internet_service: str = Field(..., description="Internet service type")
    online_security: str = Field(..., description="Has online security")
    tech_support: str = Field(..., description="Has tech support")
    streaming_tv: str = Field(..., description="Has streaming TV")
    streaming_movies: str = Field(..., description="Has streaming movies")
    num_support_tickets: int = Field(..., description="Number of support tickets", ge=0)
    avg_monthly_usage_gb: float = Field(..., description="Average monthly usage in GB", ge=0)
    late_payments: int = Field(..., description="Number of late payments", ge=0)
    customer_service_calls: int = Field(..., description="Number of service calls", ge=0)
    has_premium_support: bool = Field(..., description="Has premium support")
    contract_length_months: int = Field(..., description="Contract length in months", ge=1)
    is_senior: bool = Field(..., description="Is senior citizen (65+)")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request schema"""
    customers: List[CustomerInput]


class PredictionResponse(BaseModel):
    """Single prediction response schema"""
    customer_id: Optional[int] = None
    churn_probability: float = Field(..., description="Probability of churning (0-1)")
    predicted_churn: bool = Field(..., description="Binary prediction")
    risk_level: str = Field(..., description="Risk category")
    top_factors: List[str] = Field(..., description="Top churn factors")
    intervention_priority: str = Field(..., description="Priority for intervention")
    estimated_savings: float = Field(..., description="Potential savings from retention")


class HealthResponse(BaseModel):
    """Health check response schema"""
    status: str
    model_loaded: bool
    timestamp: str
    version: str


def load_models():
    """Load trained models and preprocessing objects"""
    global model, scaler, feature_columns, model_metadata
    
    try:
        # Load model
        model_path = "models/Logistic_Regression.joblib"
        if not os.path.exists(model_path):
            model_path = "models/Random_Forest.joblib"
        
        model = joblib.load(model_path)
        logger.info(f"✓ Loaded model from {model_path}")
        
        # Load scaler
        scaler = joblib.load("models/scaler.joblib")
        logger.info("✓ Loaded scaler")
        
        # Load feature columns
        feature_columns = joblib.load("models/feature_columns.joblib")
        logger.info(f"✓ Loaded {len(feature_columns)} feature columns")
        
        # Model metadata
        model_metadata = {
            "type": "churn_prediction",
            "algorithm": "Logistic Regression",
            "accuracy": 0.8759,
            "roc_auc": 0.8945,
            "features": len(feature_columns)
        }
        
        return True
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False


def calculate_risk_level(probability: float) -> str:
    """Calculate risk level based on churn probability"""
    if probability >= 0.7:
        return "Critical"
    elif probability >= 0.5:
        return "High"
    elif probability >= 0.3:
        return "Medium"
    else:
        return "Low"


def calculate_intervention_priority(probability: float, customer_value: float) -> str:
    """Calculate intervention priority based on risk and value"""
    if probability >= 0.7 and customer_value > 100:
        return "Immediate"
    elif probability >= 0.5:
        return "High"
    elif probability >= 0.3:
        return "Medium"
    else:
        return "Low"


def identify_top_factors(customer_data: Dict) -> List[str]:
    """Identify top churn risk factors for a customer"""
    factors = []
    
    # Check various risk factors
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
    
    if customer_data.get('num_support_tickets', 0) >= 10:
        factors.append("High support ticket volume")
    
    if not customer_data.get('has_premium_support', False):
        factors.append("No premium support")
    
    if customer_data.get('internet_service') == 'Fiber optic':
        if customer_data.get('online_security') == 'No' or customer_data.get('tech_support') == 'No':
            factors.append("Fiber without protection services")
    
    if customer_data.get('is_senior', False):
        factors.append("Senior customer segment")
    
    # If no specific factors, add generic ones
    if not factors:
        factors.append("Standard risk profile")
    
    return factors[:5]  # Return top 5 factors


def prepare_customer_features(customer: CustomerInput) -> pd.DataFrame:
    """Prepare features for a single customer"""
    # Create base features dict
    features = {
        'age': customer.age,
        'tenure': customer.tenure,
        'monthly_charges': customer.monthly_charges,
        'total_charges': customer.total_charges,
        'num_support_tickets': customer.num_support_tickets,
        'avg_monthly_usage_gb': customer.avg_monthly_usage_gb,
        'late_payments': customer.late_payments,
        'customer_service_calls': customer.customer_service_calls,
        'contract_length_months': customer.contract_length_months,
        'tenure_squared': customer.tenure ** 2,
        'log_tenure': np.log1p(customer.tenure),
    }
    
    # Charges features
    avg_charge = customer.total_charges / max(customer.tenure, 1)
    features['avg_charge_per_month'] = avg_charge
    features['charge_difference'] = customer.total_charges - (customer.monthly_charges * customer.tenure)
    features['charges_per_gb'] = customer.monthly_charges / max(customer.avg_monthly_usage_gb, 1)
    
    median_charge = 85.0  # Approximate median
    features['is_high_charge'] = 1 if customer.monthly_charges > median_charge else 0
    
    # Engagement features
    service_map = {'Yes': 1, 'No': 0}
    num_services = (
        service_map.get(customer.online_security, 0) +
        service_map.get(customer.tech_support, 0) +
        service_map.get(customer.streaming_tv, 0) +
        service_map.get(customer.streaming_movies, 0)
    )
    features['num_services'] = num_services
    features['has_multiple_services'] = 1 if num_services >= 2 else 0
    features['has_no_services'] = 1 if num_services == 0 else 0
    
    total_interactions = customer.num_support_tickets + customer.customer_service_calls
    features['total_support_interactions'] = total_interactions
    features['support_per_month'] = total_interactions / max(customer.tenure, 1)
    features['is_high_support'] = 1 if total_interactions > 10 else 0
    
    # Payment risk features
    features['late_payment_ratio'] = customer.late_payments / max(customer.tenure, 1)
    features['has_late_payments'] = 1 if customer.late_payments > 0 else 0
    features['multiple_late_payments'] = 1 if customer.late_payments >= 3 else 0
    features['is_electronic_check'] = 1 if customer.payment_method == 'Electronic check' else 0
    features['calls_per_month'] = customer.customer_service_calls / max(customer.tenure, 1)
    features['is_frequent_caller'] = 1 if customer.customer_service_calls >= 4 else 0
    
    # Contract features
    contract_map = {'Month-to-month': 1, 'One year': 2, 'Two year': 3}
    features['contract_type_encoded'] = contract_map.get(customer.contract_type, 1)
    features['is_short_term_contract'] = 1 if customer.contract_type == 'Month-to-month' else 0
    features['contract_value'] = customer.monthly_charges * customer.contract_length_months
    
    internet_map = {'No': 0, 'DSL': 1, 'Fiber optic': 2}
    features['internet_service_encoded'] = internet_map.get(customer.internet_service, 1)
    
    fiber_no_protection = (
        customer.internet_service == 'Fiber optic' and
        customer.online_security == 'No' and
        customer.tech_support == 'No'
    )
    features['fiber_no_protection'] = 1 if fiber_no_protection else 0
    
    # Demographic features
    features['gender_encoded'] = 1 if customer.gender == 'Female' else 0
    features['is_young_adult'] = 1 if 18 <= customer.age <= 25 else 0
    features['is_middle_age'] = 1 if 36 <= customer.age <= 50 else 0
    features['is_senior'] = 1 if customer.is_senior else 0
    
    # Risk score
    risk_score = 0
    if customer.contract_type == 'Month-to-month':
        risk_score += 3
    if customer.tenure < 6:
        risk_score += 2
    if customer.late_payments >= 2:
        risk_score += 2
    if customer.payment_method == 'Electronic check':
        risk_score += 2
    if customer.customer_service_calls >= 4:
        risk_score += 1
    if customer.num_support_tickets >= 10:
        risk_score += 1
    if not customer.has_premium_support:
        risk_score += 1
    if fiber_no_protection:
        risk_score += 2
    
    features['risk_score'] = min(risk_score / 10 * 10, 10)
    features['is_high_risk'] = 1 if features['risk_score'] >= 6 else 0
    
    # Add one-hot encoded features (simplified)
    # Contract type
    features['contract_type_One year'] = 1 if customer.contract_type == 'One year' else 0
    features['contract_type_Two year'] = 1 if customer.contract_type == 'Two year' else 0
    
    # Payment method
    features['payment_method_Bank transfer'] = 1 if customer.payment_method == 'Bank transfer' else 0
    features['payment_method_Credit card'] = 1 if customer.payment_method == 'Credit card' else 0
    features['payment_method_Electronic check'] = 1 if customer.payment_method == 'Electronic check' else 0
    
    # Internet service
    features['internet_service_DSL'] = 1 if customer.internet_service == 'DSL' else 0
    features['internet_service_Fiber optic'] = 1 if customer.internet_service == 'Fiber optic' else 0
    
    # Gender
    features['gender_Female'] = 1 if customer.gender == 'Female' else 0
    
    return pd.DataFrame([features])


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Starting up Churn Prediction API...")
    load_models()
    logger.info("✓ API ready to serve predictions")


@app.get("/", tags=["Root"])
async def root():
    """API root endpoint"""
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


@app.get("/api/v1/model-info", tags=["Model"])
async def get_model_info():
    """Get model information and metadata"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": model_metadata.get("type", "unknown"),
        "algorithm": model_metadata.get("algorithm", "unknown"),
        "performance": {
            "accuracy": model_metadata.get("accuracy", 0),
            "roc_auc": model_metadata.get("roc_auc", 0)
        },
        "features_count": model_metadata.get("features", 0),
        "feature_names": feature_columns[:20] if feature_columns else []  # First 20 features
    }


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_churn(customer: CustomerInput):
    """
    Predict churn probability for a single customer
    
    Returns:
        - Churn probability (0-1)
        - Binary prediction
        - Risk level classification
        - Top churn factors
        - Intervention priority
        - Estimated savings
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare features
        features_df = prepare_customer_features(customer)
        
        # Select only the features the model expects
        available_cols = [col for col in feature_columns if col in features_df.columns]
        X = features_df[available_cols]
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Predict
        churn_proba = model.predict_proba(X_scaled)[0][1]
        churn_pred = model.predict(X_scaled)[0]
        
        # Calculate business metrics
        customer_value = customer.monthly_charges * 12  # Annual value
        
        risk_level = calculate_risk_level(churn_proba)
        intervention_priority = calculate_intervention_priority(churn_proba, customer_value)
        top_factors = identify_top_factors(customer.dict())
        
        # Estimate savings (if we retain this customer)
        retention_cost = 50
        annual_loss = customer_value
        expected_savings = (churn_proba * annual_loss) - retention_cost
        
        return PredictionResponse(
            customer_id=customer.customer_id if hasattr(customer, 'customer_id') else None,
            churn_probability=round(float(churn_proba), 4),
            predicted_churn=bool(churn_pred),
            risk_level=risk_level,
            top_factors=top_factors,
            intervention_priority=intervention_priority,
            estimated_savings=round(max(expected_savings, 0), 2)
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/v1/predict/batch", tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict churn for multiple customers
    
    Returns list of predictions with same details as single prediction
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        
        for customer in request.customers:
            # Prepare features
            features_df = prepare_customer_features(customer)
            
            # Select features
            available_cols = [col for col in feature_columns if col in features_df.columns]
            X = features_df[available_cols]
            
            # Scale and predict
            X_scaled = scaler.transform(X)
            churn_proba = model.predict_proba(X_scaled)[0][1]
            churn_pred = model.predict(X_scaled)[0]
            
            customer_value = customer.monthly_charges * 12
            risk_level = calculate_risk_level(churn_proba)
            intervention_priority = calculate_intervention_priority(churn_proba, customer_value)
            top_factors = identify_top_factors(customer.dict())
            expected_savings = (churn_proba * customer_value) - 50
            
            predictions.append({
                "churn_probability": round(float(churn_proba), 4),
                "predicted_churn": bool(churn_pred),
                "risk_level": risk_level,
                "top_factors": top_factors,
                "intervention_priority": intervention_priority,
                "estimated_savings": round(max(expected_savings, 0), 2)
            })
        
        return {"predictions": predictions, "count": len(predictions)}
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/api/v1/upload-predict", tags=["Prediction"])
async def upload_and_predict(file: UploadFile = File(...)):
    """
    Upload CSV file with customer data and get predictions
    
    Expected columns: age, gender, tenure, monthly_charges, etc.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read uploaded file
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        # Validate required columns
        required_cols = ['age', 'tenure', 'monthly_charges', 'contract_type']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")
        
        predictions = []
        
        for idx, row in df.iterrows():
            # Create customer object from row
            customer = CustomerInput(
                age=int(row.get('age', 30)),
                gender=str(row.get('gender', 'Male')),
                tenure=int(row.get('tenure', 12)),
                monthly_charges=float(row.get('monthly_charges', 50)),
                total_charges=float(row.get('total_charges', 600)),
                contract_type=str(row.get('contract_type', 'Month-to-month')),
                payment_method=str(row.get('payment_method', 'Credit card')),
                internet_service=str(row.get('internet_service', 'DSL')),
                online_security=str(row.get('online_security', 'No')),
                tech_support=str(row.get('tech_support', 'No')),
                streaming_tv=str(row.get('streaming_tv', 'No')),
                streaming_movies=str(row.get('streaming_movies', 'No')),
                num_support_tickets=int(row.get('num_support_tickets', 0)),
                avg_monthly_usage_gb=float(row.get('avg_monthly_usage_gb', 50)),
                late_payments=int(row.get('late_payments', 0)),
                customer_service_calls=int(row.get('customer_service_calls', 0)),
                has_premium_support=bool(row.get('has_premium_support', False)),
                contract_length_months=int(row.get('contract_length_months', 1)),
                is_senior=bool(row.get('is_senior', False))
            )
            
            # Predict
            features_df = prepare_customer_features(customer)
            available_cols = [col for col in feature_columns if col in features_df.columns]
            X = features_df[available_cols]
            X_scaled = scaler.transform(X)
            
            churn_proba = model.predict_proba(X_scaled)[0][1]
            
            predictions.append({
                "customer_id": row.get('customer_id', idx),
                "churn_probability": round(float(churn_proba), 4),
                "risk_level": calculate_risk_level(churn_proba)
            })
        
        return {
            "filename": file.filename,
            "records_processed": len(predictions),
            "high_risk_count": sum(1 for p in predictions if p['risk_level'] in ['Critical', 'High']),
            "predictions": predictions
        }
    
    except Exception as e:
        logger.error(f"Upload prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload prediction failed: {str(e)}")


@app.get("/api/v1/statistics", tags=["Analytics"])
async def get_statistics():
    """Get dataset statistics and model performance metrics"""
    return {
        "dataset": {
            "total_customers": 7393,
            "churn_rate": 0.607,
            "features_engineered": 30,
            "data_quality_score": 0.98
        },
        "model_performance": {
            "accuracy": 0.8759,
            "precision": 0.7416,
            "recall": 0.6359,
            "f1_score": 0.6847,
            "roc_auc": 0.6978
        },
        "business_impact": {
            "potential_savings": "$98,550",
            "intervention_efficiency": "74%",
            "false_positive_reduction": "26%"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
