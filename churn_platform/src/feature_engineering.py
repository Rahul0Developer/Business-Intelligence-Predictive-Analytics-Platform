"""
Feature Engineering Module for Customer Churn Prediction
Creates behavioral, engagement, and risk-based features
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates predictive features for churn modeling"""
    
    def __init__(self):
        self.feature_names = []
        self.categorical_features = []
        self.numerical_features = []
        
    def create_tenure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on customer tenure"""
        df_feat = df.copy()
        
        # Tenure groups
        df_feat['tenure_group'] = pd.cut(
            df_feat['tenure'],
            bins=[0, 6, 12, 24, 48, 72],
            labels=['0-6mo', '6-12mo', '1-2yr', '2-4yr', '4yr+']
        )
        
        # Is new customer (less than 6 months)
        df_feat['is_new_customer'] = (df_feat['tenure'] < 6).astype(int)
        
        # Tenure squared (for non-linear relationships)
        df_feat['tenure_squared'] = df_feat['tenure'] ** 2
        
        # Log tenure (handle 0 by adding 1)
        df_feat['log_tenure'] = np.log1p(df_feat['tenure'])
        
        logger.info("Created tenure-based features")
        return df_feat
    
    def create_charges_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on charges and billing"""
        df_feat = df.copy()
        
        # Average charge per month (total / tenure)
        df_feat['avg_charge_per_month'] = np.where(
            df_feat['tenure'] > 0,
            df_feat['total_charges'] / df_feat['tenure'],
            df_feat['monthly_charges']
        )
        
        # Charge difference (actual vs expected)
        df_feat['charge_difference'] = df_feat['total_charges'] - \
                                       (df_feat['monthly_charges'] * df_feat['tenure'])
        
        # Monthly charges per GB usage
        df_feat['charges_per_gb'] = np.where(
            df_feat['avg_monthly_usage_gb'] > 0,
            df_feat['monthly_charges'] / df_feat['avg_monthly_usage_gb'],
            df_feat['monthly_charges']
        )
        
        # High charges flag (above median)
        median_charge = df_feat['monthly_charges'].median()
        df_feat['is_high_charge'] = (df_feat['monthly_charges'] > median_charge).astype(int)
        
        logger.info("Created charges-based features")
        return df_feat
    
    def create_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on customer engagement"""
        df_feat = df.copy()
        
        # Number of services (count of Yes values in service columns)
        service_cols = ['online_security', 'tech_support', 'streaming_tv', 'streaming_movies']
        df_feat['num_services'] = sum(
            df_feat[col].map({'Yes': 1, 'No': 0}) 
            for col in service_cols if col in df_feat.columns
        )
        
        # Has multiple services
        df_feat['has_multiple_services'] = (df_feat['num_services'] >= 2).astype(int)
        
        # No services flag
        df_feat['has_no_services'] = (df_feat['num_services'] == 0).astype(int)
        
        # Support engagement (tickets + calls)
        df_feat['total_support_interactions'] = (
            df_feat['num_support_tickets'] + df_feat['customer_service_calls']
        )
        
        # Support intensity (interactions per month of tenure)
        df_feat['support_per_month'] = np.where(
            df_feat['tenure'] > 0,
            df_feat['total_support_interactions'] / df_feat['tenure'],
            df_feat['total_support_interactions']
        )
        
        # High support flag
        df_feat['is_high_support'] = (df_feat['total_support_interactions'] > 10).astype(int)
        
        logger.info("Created engagement features")
        return df_feat
    
    def create_payment_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features related to payment behavior and risk"""
        df_feat = df.copy()
        
        # Late payment ratio
        df_feat['late_payment_ratio'] = np.where(
            df_feat['tenure'] > 0,
            df_feat['late_payments'] / df_feat['tenure'],
            df_feat['late_payments']
        )
        
        # Has late payments flag
        df_feat['has_late_payments'] = (df_feat['late_payments'] > 0).astype(int)
        
        # Multiple late payments
        df_feat['multiple_late_payments'] = (df_feat['late_payments'] >= 3).astype(int)
        
        # Payment method risk (electronic check is higher risk)
        df_feat['is_electronic_check'] = (df_feat['payment_method'] == 'Electronic check').astype(int)
        
        # Service calls per month
        df_feat['calls_per_month'] = np.where(
            df_feat['tenure'] > 0,
            df_feat['customer_service_calls'] / df_feat['tenure'],
            df_feat['customer_service_calls']
        )
        
        # High call frequency
        df_feat['is_frequent_caller'] = (df_feat['customer_service_calls'] >= 4).astype(int)
        
        logger.info("Created payment risk features")
        return df_feat
    
    def create_contract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on contract characteristics"""
        df_feat = df.copy()
        
        # Contract type encoding (ordinal)
        contract_mapping = {
            'Month-to-month': 1,
            'One year': 2,
            'Two year': 3
        }
        df_feat['contract_type_encoded'] = df_feat['contract_type'].map(contract_mapping)
        
        # Is short-term contract
        df_feat['is_short_term_contract'] = (df_feat['contract_type'] == 'Month-to-month').astype(int)
        
        # Contract value (monthly * contract length)
        df_feat['contract_value'] = df_feat['monthly_charges'] * df_feat['contract_length_months']
        
        # Internet service type encoding
        internet_mapping = {
            'No': 0,
            'DSL': 1,
            'Fiber optic': 2
        }
        df_feat['internet_service_encoded'] = df_feat['internet_service'].map(internet_mapping)
        
        # Fiber without protection (high risk segment)
        df_feat['fiber_no_protection'] = (
            (df_feat['internet_service'] == 'Fiber optic') &
            (df_feat['online_security'] == 'No') &
            (df_feat['tech_support'] == 'No')
        ).astype(int)
        
        logger.info("Created contract features")
        return df_feat
    
    def create_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create demographic-based features"""
        df_feat = df.copy()
        
        # Age groups
        df_feat['age_group'] = pd.cut(
            df_feat['age'],
            bins=[0, 25, 35, 50, 65, 100],
            labels=['18-25', '26-35', '36-50', '51-65', '65+']
        )
        
        # Is young adult
        df_feat['is_young_adult'] = ((df_feat['age'] >= 18) & (df_feat['age'] <= 25)).astype(int)
        
        # Is middle age
        df_feat['is_middle_age'] = ((df_feat['age'] >= 36) & (df_feat['age'] <= 50)).astype(int)
        
        # Gender encoding
        df_feat['gender_encoded'] = df_feat['gender'].map({'Male': 0, 'Female': 1})
        
        logger.info("Created demographic features")
        return df_feat
    
    def create_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a composite risk score feature"""
        df_feat = df.copy()
        
        # Initialize risk score
        risk_score = np.zeros(len(df_feat))
        
        # Add risk points for various factors
        risk_score += (df_feat['contract_type'] == 'Month-to-month').astype(int) * 3
        risk_score += (df_feat['tenure'] < 6).astype(int) * 2
        risk_score += (df_feat['late_payments'] >= 2).astype(int) * 2
        risk_score += (df_feat['payment_method'] == 'Electronic check').astype(int) * 2
        risk_score += (df_feat['customer_service_calls'] >= 4).astype(int) * 1
        risk_score += (df_feat['num_support_tickets'] >= 10).astype(int) * 1
        risk_score += (~df_feat['has_premium_support']).astype(int) * 1
        risk_score += (df_feat['fiber_no_protection'] if 'fiber_no_protection' in df_feat.columns else 0) * 2
        
        # Normalize to 0-10 scale
        df_feat['risk_score'] = np.clip(risk_score / 10 * 10, 0, 10)
        
        # High risk flag
        df_feat['is_high_risk'] = (df_feat['risk_score'] >= 6).astype(int)
        
        logger.info("Created composite risk score")
        return df_feat
    
    def encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables for ML models"""
        df_encoded = df.copy()
        
        # Binary encoding for Yes/No columns
        binary_cols = ['online_security', 'tech_support', 'streaming_tv', 'streaming_movies']
        for col in binary_cols:
            if col in df_encoded.columns:
                df_encoded[f'{col}_encoded'] = df_encoded[col].map({'Yes': 1, 'No': 0})
        
        # One-hot encoding for categorical columns
        categorical_cols = ['contract_type', 'payment_method', 'internet_service', 
                           'gender', 'tenure_group', 'age_group']
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                # Get dummies
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                self.categorical_features.extend(dummies.columns.tolist())
        
        logger.info(f"Encoded {len(categorical_cols)} categorical variables")
        return df_encoded
    
    def get_feature_list(self) -> Dict:
        """Return list of engineered features"""
        return {
            'numerical': self.numerical_features,
            'categorical': self.categorical_features
        }
    
    def engineer_features(self, df: pd.DataFrame, include_encoding: bool = True) -> pd.DataFrame:
        """
        Complete feature engineering pipeline
        
        Args:
            df: Cleaned input DataFrame
            include_encoding: Whether to encode categorical variables
        
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting feature engineering pipeline...")
        
        # Apply all feature creation methods
        df_feat = self.create_tenure_features(df)
        df_feat = self.create_charges_features(df_feat)
        df_feat = self.create_engagement_features(df_feat)
        df_feat = self.create_payment_risk_features(df_feat)
        df_feat = self.create_contract_features(df_feat)
        df_feat = self.create_demographic_features(df_feat)
        df_feat = self.create_risk_score(df_feat)
        
        # Encode categorical variables
        if include_encoding:
            df_feat = self.encode_categorical_variables(df_feat)
        
        # Define feature lists
        self.numerical_features = [
            'age', 'tenure', 'monthly_charges', 'total_charges',
            'num_support_tickets', 'avg_monthly_usage_gb', 'late_payments',
            'customer_service_calls', 'contract_length_months',
            'tenure_squared', 'log_tenure', 'avg_charge_per_month',
            'charge_difference', 'charges_per_gb', 'num_services',
            'total_support_interactions', 'support_per_month',
            'late_payment_ratio', 'calls_per_month', 'contract_value',
            'risk_score', 'contract_type_encoded', 'internet_service_encoded',
            'gender_encoded', 'is_new_customer', 'is_high_charge',
            'has_multiple_services', 'has_no_services', 'is_high_support',
            'has_late_payments', 'multiple_late_payments', 'is_electronic_check',
            'is_frequent_caller', 'is_short_term_contract', 'fiber_no_protection',
            'is_young_adult', 'is_middle_age', 'is_senior', 'is_high_risk'
        ]
        
        # Add encoded features
        encoded_cols = [col for col in df_feat.columns if '_encoded' in col or col.startswith('contract_type_') or col.startswith('payment_method_') or col.startswith('internet_service_') or col.startswith('gender_') or col.startswith('tenure_group_') or col.startswith('age_group_')]
        self.numerical_features.extend([c for c in encoded_cols if c not in self.numerical_features])
        
        logger.info(f"Created {len(self.numerical_features)} numerical features")
        logger.info(f"Created {len(self.categorical_features)} one-hot encoded features")
        
        return df_feat


def engineer_customer_features(input_path: str, output_path: str = None,
                               include_encoding: bool = True) -> pd.DataFrame:
    """
    Main function to engineer features from cleaned data
    
    Args:
        input_path: Path to cleaned CSV file
        output_path: Path to save engineered features (optional)
        include_encoding: Whether to encode categorical variables
    
    Returns:
        DataFrame with engineered features
    """
    engineer = FeatureEngineer()
    
    # Load cleaned data
    logger.info(f"Loading cleaned data from {input_path}")
    df_clean = pd.read_csv(input_path)
    
    # Engineer features
    df_features = engineer.engineer_features(df_clean, include_encoding=include_encoding)
    
    # Save if output path provided
    if output_path:
        df_features.to_csv(output_path, index=False)
        logger.info(f"Saved engineered features to {output_path}")
    
    # Report
    feature_report = engineer.get_feature_list()
    logger.info(f"Total features created: {len(feature_report['numerical']) + len(feature_report['categorical'])}")
    
    return df_features


if __name__ == "__main__":
    print("=" * 60)
    print("FEATURE ENGINEERING MODULE")
    print("=" * 60)
    
    # Engineer features
    df_engineered = engineer_customer_features(
        input_path='data/customers_cleaned.csv',
        output_path='data/customers_engineered.csv',
        include_encoding=True
    )
    
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 60)
    print(f"Original columns: 21")
    print(f"Engineered columns: {len(df_engineered.columns)}")
    print(f"New features created: {len(df_engineered.columns) - 21}")
    
    print("\nKey engineered features:")
    key_features = [
        'tenure_group', 'is_new_customer', 'log_tenure',
        'avg_charge_per_month', 'charges_per_gb', 'is_high_charge',
        'num_services', 'total_support_interactions', 'support_per_month',
        'late_payment_ratio', 'is_electronic_check', 'is_frequent_caller',
        'contract_type_encoded', 'fiber_no_protection', 'risk_score', 'is_high_risk'
    ]
    for feat in key_features:
        if feat in df_engineered.columns:
            print(f"  ✓ {feat}")
    
    print(f"\nDataset shape: {df_engineered.shape}")
    print(f"\n✓ Feature engineering completed successfully!")
