"""
Data Generation Module for Customer Churn Prediction
Generates realistic synthetic customer data with churn patterns
"""

import pandas as pd
import numpy as np
from typing import Tuple
import os

def generate_customer_data(n_customers: int = 7393, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic customer data with realistic churn patterns.
    
    Args:
        n_customers: Number of customers to generate
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with customer data
    """
    np.random.seed(seed)
    
    # Customer IDs
    customer_ids = range(1001, 1001 + n_customers)
    
    # Age distribution (normal distribution centered at 45)
    ages = np.clip(np.random.normal(45, 15, n_customers), 18, 80).astype(int)
    
    # Gender
    genders = np.random.choice(['Male', 'Female'], n_customers)
    
    # Tenure in months (exponential distribution - many new, few long-term)
    tenures = np.clip(np.random.exponential(24, n_customers), 1, 72).astype(int)
    
    # Contract type (affects churn significantly)
    contract_types = np.random.choice(
        ['Month-to-month', 'One year', 'Two year'], 
        n_customers, 
        p=[0.55, 0.25, 0.20]
    )
    
    # Internet service
    internet_services = np.random.choice(
        ['DSL', 'Fiber optic', 'No'], 
        n_customers, 
        p=[0.40, 0.45, 0.15]
    )
    
    # Monthly charges based on internet service and other factors
    base_charges = np.where(
        internet_services == 'Fiber optic', 
        np.random.uniform(80, 120, n_customers),
        np.where(internet_services == 'DSL', 
                 np.random.uniform(50, 80, n_customers),
                 np.random.uniform(20, 40, n_customers))
    )
    
    # Add variation for additional services
    has_streaming = np.random.choice([True, False], n_customers, p=[0.6, 0.4])
    has_security = np.random.choice([True, False], n_customers, p=[0.5, 0.5])
    has_tech_support = np.random.choice([True, False], n_customers, p=[0.4, 0.6])
    
    monthly_charges = base_charges + \
                      has_streaming * np.random.uniform(10, 20, n_customers) + \
                      has_security * np.random.uniform(5, 10, n_customers) + \
                      has_tech_support * np.random.uniform(5, 10, n_customers)
    monthly_charges = np.round(monthly_charges, 2)
    
    # Total charges
    total_charges = np.round(monthly_charges * tenures * np.random.uniform(0.95, 1.05, n_customers), 2)
    
    # Payment method
    payment_methods = np.random.choice(
        ['Bank transfer', 'Credit card', 'Electronic check', 'Mailed check'], 
        n_customers, 
        p=[0.35, 0.30, 0.25, 0.10]
    )
    
    # Online security (depends on internet service)
    online_security = np.where(
        internet_services == 'No', 'No',
        np.random.choice(['Yes', 'No'], n_customers, p=[0.5, 0.5])
    )
    
    # Tech support
    tech_support = np.random.choice(['Yes', 'No'], n_customers, p=[0.4, 0.6])
    
    # Streaming TV
    streaming_tv = np.where(has_streaming, 'Yes', 'No')
    
    # Streaming movies
    streaming_movies = np.where(has_streaming, 
                                np.random.choice(['Yes', 'No'], n_customers, p=[0.7, 0.3]), 
                                'No')
    
    # Number of support tickets (higher for unhappy customers)
    num_support_tickets = np.clip(np.random.poisson(8, n_customers), 0, 40).astype(int)
    
    # Average monthly usage in GB
    avg_monthly_usage_gb = np.round(np.random.uniform(5, 130, n_customers), 1)
    
    # Late payments (correlated with churn)
    late_payments = np.clip(np.random.poisson(2, n_customers), 0, 8).astype(int)
    
    # Customer service calls
    customer_service_calls = np.clip(np.random.poisson(2, n_customers), 0, 10).astype(int)
    
    # Premium support
    has_premium_support = np.random.choice([True, False], n_customers, p=[0.3, 0.7])
    
    # Contract length in months
    contract_length_months = np.where(
        contract_types == 'Month-to-month', 1,
        np.where(contract_types == 'One year', 12, 24)
    )
    
    # Is senior (age >= 65)
    is_senior = ages >= 65
    
    # CHURN LOGIC - This is the key business logic
    # Calculate churn probability based on multiple factors
    churn_prob = np.zeros(n_customers)
    
    # Contract type is major factor
    churn_prob += np.where(contract_types == 'Month-to-month', 0.35, 
                          np.where(contract_types == 'One year', 0.15, 0.05))
    
    # Tenure effect (newer customers more likely to churn)
    churn_prob += np.where(tenures < 6, 0.20, np.where(tenures < 12, 0.10, 0.02))
    
    # Payment method (electronic check associated with higher churn)
    churn_prob += np.where(payment_methods == 'Electronic check', 0.15, 0.0)
    
    # Late payments
    churn_prob += np.clip(late_payments * 0.05, 0, 0.25)
    
    # High support tickets
    churn_prob += np.clip((num_support_tickets - 5) * 0.02, 0, 0.15)
    
    # Many service calls
    churn_prob += np.clip((customer_service_calls - 2) * 0.03, 0, 0.15)
    
    # No premium support
    churn_prob += np.where(~has_premium_support, 0.08, 0.0)
    
    # Fiber optic without security/tech support
    fiber_no_support = (internet_services == 'Fiber optic') & \
                       (online_security == 'No') & \
                       (tech_support == 'No')
    churn_prob += np.where(fiber_no_support, 0.12, 0.0)
    
    # Senior citizens slightly higher churn
    churn_prob += np.where(is_senior, 0.05, 0.0)
    
    # Cap probability
    churn_prob = np.clip(churn_prob, 0, 0.95)
    
    # Generate churn based on probability
    churn = (np.random.random(n_customers) < churn_prob).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'age': ages,
        'gender': genders,
        'tenure': tenures,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'contract_type': contract_types,
        'payment_method': payment_methods,
        'internet_service': internet_services,
        'online_security': online_security,
        'tech_support': tech_support,
        'streaming_tv': streaming_tv,
        'streaming_movies': streaming_movies,
        'num_support_tickets': num_support_tickets,
        'avg_monthly_usage_gb': avg_monthly_usage_gb,
        'late_payments': late_payments,
        'customer_service_calls': customer_service_calls,
        'has_premium_support': has_premium_support,
        'contract_length_months': contract_length_months,
        'is_senior': is_senior,
        'churn': churn
    })
    
    return df


def save_sample_data(df: pd.DataFrame, sample_size: int = 50, output_path: str = 'data/customers_sample.csv'):
    """Save a small sample for quick testing"""
    sample = df.head(sample_size)
    sample.to_csv(output_path, index=False)
    print(f"✓ Saved sample data ({sample_size} rows) to {output_path}")


def save_full_data(df: pd.DataFrame, output_path: str = 'data/customers_full.csv'):
    """Save full dataset"""
    df.to_csv(output_path, index=False)
    print(f"✓ Saved full dataset ({len(df)} rows) to {output_path}")


if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate full dataset
    print("Generating customer churn dataset...")
    df = generate_customer_data(n_customers=7393, seed=42)
    
    # Save datasets
    save_full_data(df)
    save_sample_data(df)
    
    # Print statistics
    print("\n📊 Dataset Statistics:")
    print(f"Total customers: {len(df)}")
    print(f"Churn rate: {df['churn'].mean():.2%}")
    print(f"Churned customers: {df['churn'].sum()}")
    print(f"Retained customers: {(df['churn'] == 0).sum()}")
    print(f"\nAge range: {df['age'].min()} - {df['age'].max()}")
    print(f"Tenure range: {df['tenure'].min()} - {df['tenure'].max()} months")
    print(f"\nContract types: {df['contract_type'].value_counts().to_dict()}")
    print(f"Internet services: {df['internet_service'].value_counts().to_dict()}")
