"""
Data Cleaning and Validation Module
Handles missing values, data types, outliers, and data quality checks
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """Handles all data cleaning operations for customer churn data"""
    
    def __init__(self):
        self.cleaning_stats = {}
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file"""
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        return df
    
    def detect_missing_values(self, df: pd.DataFrame) -> Dict:
        """Detect and report missing values"""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        
        missing_info = pd.DataFrame({
            'missing_count': missing,
            'missing_pct': missing_pct
        })
        
        # Only return columns with missing values
        missing_info = missing_info[missing_info['missing_count'] > 0]
        
        return missing_info.to_dict('index')
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using appropriate strategies"""
        df_clean = df.copy()
        
        # Numeric columns - fill with median
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                logger.info(f"Filled missing values in '{col}' with median: {median_val}")
        
        # Categorical columns - fill with mode
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().any():
                mode_val = df_clean[col].mode()[0]
                df_clean[col].fillna(mode_val, inplace=True)
                logger.info(f"Filled missing values in '{col}' with mode: {mode_val}")
        
        return df_clean
    
    def fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure correct data types for all columns"""
        df_fixed = df.copy()
        
        # Customer ID should be integer or string (not used in modeling)
        if 'customer_id' in df_fixed.columns:
            df_fixed['customer_id'] = df_fixed['customer_id'].astype(int)
        
        # Boolean columns
        bool_columns = ['has_premium_support', 'is_senior']
        for col in bool_columns:
            if col in df_fixed.columns:
                # Handle various boolean representations
                if df_fixed[col].dtype == 'object':
                    df_fixed[col] = df_fixed[col].map({'True': True, 'False': False, 
                                                       'true': True, 'false': False,
                                                       'Yes': True, 'No': False,
                                                       'yes': True, 'no': False})
                df_fixed[col] = df_fixed[col].astype(bool)
        
        # Target variable
        if 'churn' in df_fixed.columns:
            df_fixed['churn'] = df_fixed['churn'].astype(int)
        
        # Numeric columns
        numeric_columns = ['age', 'tenure', 'monthly_charges', 'total_charges',
                          'num_support_tickets', 'avg_monthly_usage_gb', 
                          'late_payments', 'customer_service_calls', 
                          'contract_length_months']
        
        for col in numeric_columns:
            if col in df_fixed.columns:
                df_fixed[col] = pd.to_numeric(df_fixed[col], errors='coerce')
        
        logger.info("Fixed data types for all columns")
        return df_fixed
    
    def detect_outliers_iqr(self, df: pd.DataFrame, column: str) -> Tuple[float, float]:
        """Detect outliers using IQR method"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return lower_bound, upper_bound
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'clip') -> pd.DataFrame:
        """
        Handle outliers in numeric columns
        
        Args:
            df: Input DataFrame
            method: 'clip' to cap outliers, 'remove' to drop them
        """
        df_outlier = df.copy()
        
        numeric_cols = ['age', 'tenure', 'monthly_charges', 'total_charges',
                       'num_support_tickets', 'avg_monthly_usage_gb',
                       'late_payments', 'customer_service_calls']
        
        for col in numeric_cols:
            if col not in df_outlier.columns:
                continue
                
            lower, upper = self.detect_outliers_iqr(df_outlier, col)
            
            if method == 'clip':
                original_min = df_outlier[col].min()
                original_max = df_outlier[col].max()
                
                df_outlier[col] = df_outlier[col].clip(lower=lower, upper=upper)
                
                if original_min < lower or original_max > upper:
                    logger.info(f"Capped outliers in '{col}': [{lower:.2f}, {upper:.2f}]")
            
            elif method == 'remove':
                mask = (df_outlier[col] >= lower) & (df_outlier[col] <= upper)
                df_outlier = df_outlier[mask]
                logger.info(f"Removed {len(df) - len(df_outlier)} rows with outliers in '{col}'")
        
        return df_outlier
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """Comprehensive data quality validation"""
        validation_results = {
            'passed': True,
            'issues': [],
            'warnings': []
        }
        
        # Check for duplicate customer IDs
        if 'customer_id' in df.columns:
            duplicates = df['customer_id'].duplicated().sum()
            if duplicates > 0:
                validation_results['issues'].append(f"Found {duplicates} duplicate customer IDs")
                validation_results['passed'] = False
        
        # Check for negative values in numeric columns
        numeric_cols = ['age', 'tenure', 'monthly_charges', 'total_charges',
                       'num_support_tickets', 'late_payments', 'customer_service_calls']
        
        for col in numeric_cols:
            if col in df.columns:
                negatives = (df[col] < 0).sum()
                if negatives > 0:
                    validation_results['issues'].append(f"Found {negatives} negative values in '{col}'")
                    validation_results['passed'] = False
        
        # Check churn values (should be 0 or 1)
        if 'churn' in df.columns:
            unique_churn = df['churn'].unique()
            if not set(unique_churn).issubset({0, 1}):
                validation_results['issues'].append(f"Invalid churn values: {unique_churn}")
                validation_results['passed'] = False
        
        # Check tenure range (should be positive)
        if 'tenure' in df.columns:
            min_tenure = df['tenure'].min()
            if min_tenure < 1:
                validation_results['warnings'].append(f"Minimum tenure is {min_tenure} months")
        
        # Check for extremely high charges
        if 'monthly_charges' in df.columns:
            high_charges = (df['monthly_charges'] > 200).sum()
            if high_charges > 0:
                validation_results['warnings'].append(f"Found {high_charges} customers with monthly charges > $200")
        
        # Check class balance
        if 'churn' in df.columns:
            churn_rate = df['churn'].mean()
            if churn_rate < 0.1 or churn_rate > 0.9:
                validation_results['warnings'].append(
                    f"Severe class imbalance detected: {churn_rate:.2%} churn rate"
                )
        
        return validation_results
    
    def clean_pipeline(self, df: pd.DataFrame, remove_outliers: bool = False) -> pd.DataFrame:
        """
        Complete cleaning pipeline
        
        Args:
            df: Raw input DataFrame
            remove_outliers: If True, remove outlier rows; if False, clip them
        """
        logger.info("Starting data cleaning pipeline...")
        
        # Step 1: Fix data types
        df_clean = self.fix_data_types(df)
        
        # Step 2: Handle missing values
        df_clean = self.handle_missing_values(df_clean)
        
        # Step 3: Handle outliers
        outlier_method = 'remove' if remove_outliers else 'clip'
        df_clean = self.handle_outliers(df_clean, method=outlier_method)
        
        # Step 4: Validate cleaned data
        validation = self.validate_data_quality(df_clean)
        
        if not validation['passed']:
            logger.warning(f"Data quality issues found: {validation['issues']}")
        else:
            logger.info("✓ Data quality validation passed")
        
        if validation['warnings']:
            for warning in validation['warnings']:
                logger.warning(f"Warning: {warning}")
        
        # Store cleaning statistics
        self.cleaning_stats = {
            'original_rows': len(df),
            'cleaned_rows': len(df_clean),
            'rows_removed': len(df) - len(df_clean),
            'validation_passed': validation['passed'],
            'issues_found': len(validation['issues']),
            'warnings_found': len(validation['warnings'])
        }
        
        logger.info(f"Cleaning complete: {self.cleaning_stats['original_rows']} → "
                   f"{self.cleaning_stats['cleaned_rows']} rows")
        
        return df_clean
    
    def get_cleaning_report(self) -> Dict:
        """Return cleaning statistics report"""
        return self.cleaning_stats


def clean_customer_data(input_path: str, output_path: Optional[str] = None, 
                       remove_outliers: bool = False) -> pd.DataFrame:
    """
    Main function to clean customer data
    
    Args:
        input_path: Path to raw CSV file
        output_path: Path to save cleaned data (optional)
        remove_outliers: Whether to remove outlier rows
    
    Returns:
        Cleaned DataFrame
    """
    cleaner = DataCleaner()
    
    # Load data
    df_raw = cleaner.load_data(input_path)
    
    # Detect initial missing values
    missing_before = cleaner.detect_missing_values(df_raw)
    if missing_before:
        logger.info(f"Missing values before cleaning: {missing_before}")
    
    # Clean data
    df_clean = cleaner.clean_pipeline(df_raw, remove_outliers=remove_outliers)
    
    # Save if output path provided
    if output_path:
        df_clean.to_csv(output_path, index=False)
        logger.info(f"Saved cleaned data to {output_path}")
    
    # Return cleaning report
    report = cleaner.get_cleaning_report()
    logger.info(f"Cleaning report: {report}")
    
    return df_clean


if __name__ == "__main__":
    # Test the cleaning pipeline
    print("=" * 60)
    print("DATA CLEANING AND VALIDATION MODULE")
    print("=" * 60)
    
    # Clean the full dataset
    df_cleaned = clean_customer_data(
        input_path='data/customers_full.csv',
        output_path='data/customers_cleaned.csv',
        remove_outliers=False
    )
    
    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)
    print(f"Original rows: {len(pd.read_csv('data/customers_full.csv'))}")
    print(f"Cleaned rows: {len(df_cleaned)}")
    print(f"Churn rate: {df_cleaned['churn'].mean():.2%}")
    print(f"\nColumns: {list(df_cleaned.columns)}")
    print(f"\nData types:\n{df_cleaned.dtypes}")
    print(f"\n✓ Data cleaning completed successfully!")
