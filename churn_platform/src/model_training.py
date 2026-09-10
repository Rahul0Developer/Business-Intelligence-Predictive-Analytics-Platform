"""
ML Model Training Module for Customer Churn Prediction
Trains and evaluates multiple models with MLflow tracking
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnModelTrainer:
    """Handles ML model training and evaluation"""
    
    def __init__(self, mlflow_tracking_uri: str = "sqlite:///mlflow.db"):
        import os
        os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.best_model = None
        self.feature_columns = []
        
    def prepare_data(self, df: pd.DataFrame, 
                    target_col: str = 'churn',
                    exclude_cols: List[str] = None) -> Tuple:
        """
        Prepare data for modeling
        
        Args:
            df: Input DataFrame with features
            target_col: Name of target column
            exclude_cols: Columns to exclude from features
        
        Returns:
            X_train, X_test, y_train, y_test, feature_columns
        """
        # Define columns to exclude
        if exclude_cols is None:
            exclude_cols = ['customer_id']
        
        exclude_cols.append(target_col)
        
        # Get feature columns (exclude non-feature columns)
        exclude_patterns = ['customer_id', 'churn']
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols 
                       and not any(pat in col for pat in ['_group', '_encoded'])]
        
        # Add encoded features
        encoded_cols = [col for col in df.columns if '_encoded' in col or 
                       col.startswith('contract_type_') or col.startswith('payment_method_') or
                       col.startswith('internet_service_') or col.startswith('gender_')]
        feature_cols.extend(encoded_cols)
        
        # Remove duplicates while preserving order
        feature_cols = list(dict.fromkeys(feature_cols))
        
        # Filter to only existing columns
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        self.feature_columns = feature_cols
        logger.info(f"Using {len(feature_cols)} features for modeling")
        
        # Separate features and target
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle any remaining non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        self.feature_columns = list(numeric_cols)
        
        # Drop rows with NaN values
        mask = ~X.isna().any(axis=1) & ~y.isna()
        X = X[mask]
        y = y[mask]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Churn rate - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple:
        """Scale numerical features"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        return X_train_scaled, X_test_scaled
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_pred_proba: np.ndarray, model_name: str) -> Dict:
        """Comprehensive model evaluation"""
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        # Calculate business metrics
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Business cost assumptions
        intervention_cost = 50  # Cost to retain a customer
        churn_loss = 500  # Revenue lost when customer churns
        
        # Savings from correct predictions
        true_positive_savings = tp * churn_loss  # Saved customers we correctly identified
        false_positive_cost = fp * intervention_cost  # Wasted interventions
        false_negative_cost = fn * churn_loss  # Lost customers we missed
        
        net_savings = true_positive_savings - false_positive_cost
        metrics['net_savings'] = net_savings
        metrics['true_positives'] = int(tp)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_negatives'] = int(tn)
        
        return metrics
    
    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series,
                                  X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Train Logistic Regression model"""
        logger.info("Training Logistic Regression...")
        
        # Get or create experiment
        experiment_id = mlflow.get_experiment_by_name("churn_prediction")
        if experiment_id is None:
            experiment_id = mlflow.create_experiment("churn_prediction")
        else:
            experiment_id = experiment_id.experiment_id
        
        with mlflow.start_run(experiment_id=experiment_id, run_name="Logistic_Regression"):
            # Log parameters
            mlflow.log_param("model_type", "Logistic Regression")
            mlflow.log_param("C", 1.0)
            mlflow.log_param("max_iter", 1000)
            
            # Train model
            model = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight='balanced')
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Evaluate
            metrics = self.evaluate_model(y_test, y_pred, y_pred_proba, "Logistic Regression")
            
            # Log metrics
            mlflow.log_metric("accuracy", metrics['accuracy'])
            mlflow.log_metric("precision", metrics['precision'])
            mlflow.log_metric("recall", metrics['recall'])
            mlflow.log_metric("f1_score", metrics['f1_score'])
            mlflow.log_metric("roc_auc", metrics['roc_auc'])
            mlflow.log_metric("net_savings", metrics['net_savings'])
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Store results
            self.models['Logistic Regression'] = model
            self.results['Logistic Regression'] = metrics
            
            logger.info(f"LR Accuracy: {metrics['accuracy']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
            
        return metrics
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, y_test: pd.Series,
                           n_estimators: int = 100) -> Dict:
        """Train Random Forest model"""
        logger.info("Training Random Forest...")
        
        with mlflow.start_run(run_name="Random_Forest"):
            # Log parameters
            mlflow.log_param("model_type", "Random Forest")
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", 10)
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=n_estimators, 
                max_depth=10,
                random_state=42, 
                class_weight='balanced',
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Evaluate
            metrics = self.evaluate_model(y_test, y_pred, y_pred_proba, "Random Forest")
            
            # Log metrics
            mlflow.log_metric("accuracy", metrics['accuracy'])
            mlflow.log_metric("precision", metrics['precision'])
            mlflow.log_metric("recall", metrics['recall'])
            mlflow.log_metric("f1_score", metrics['f1_score'])
            mlflow.log_metric("roc_auc", metrics['roc_auc'])
            mlflow.log_metric("net_savings", metrics['net_savings'])
            
            # Log feature importances
            feature_importance = dict(zip(X_train.columns, model.feature_importances_))
            for feat, imp in sorted(feature_importance.items(), key=lambda x: -x[1])[:10]:
                mlflow.log_param(f"top_feature_{feat}", imp)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Store results
            self.models['Random Forest'] = model
            self.results['Random Forest'] = metrics
            
            logger.info(f"RF Accuracy: {metrics['accuracy']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
            
        return metrics
    
    def train_gradient_boosting(self, X_train: pd.DataFrame, y_train: pd.Series,
                                X_test: pd.DataFrame, y_test: pd.Series,
                                n_estimators: int = 100) -> Dict:
        """Train Gradient Boosting model"""
        logger.info("Training Gradient Boosting...")
        
        with mlflow.start_run(run_name="Gradient_Boosting"):
            # Log parameters
            mlflow.log_param("model_type", "Gradient Boosting")
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("learning_rate", 0.1)
            mlflow.log_param("max_depth", 5)
            
            # Train model
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Evaluate
            metrics = self.evaluate_model(y_test, y_pred, y_pred_proba, "Gradient Boosting")
            
            # Log metrics
            mlflow.log_metric("accuracy", metrics['accuracy'])
            mlflow.log_metric("precision", metrics['precision'])
            mlflow.log_metric("recall", metrics['recall'])
            mlflow.log_metric("f1_score", metrics['f1_score'])
            mlflow.log_metric("roc_auc", metrics['roc_auc'])
            mlflow.log_metric("net_savings", metrics['net_savings'])
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Store results
            self.models['Gradient Boosting'] = model
            self.results['Gradient Boosting'] = metrics
            
            logger.info(f"GB Accuracy: {metrics['accuracy']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
            
        return metrics
    
    def select_best_model(self, metric: str = 'roc_auc') -> str:
        """Select best model based on specified metric"""
        if not self.results:
            raise ValueError("No models trained yet")
        
        best_model_name = max(self.results.keys(), 
                             key=lambda k: self.results[k][metric])
        
        self.best_model = self.models[best_model_name]
        logger.info(f"Best model: {best_model_name} ({metric}: {self.results[best_model_name][metric]:.4f})")
        
        return best_model_name
    
    def get_feature_importance(self, model_name: str = None) -> pd.DataFrame:
        """Get feature importance from the best model"""
        if model_name is None:
            if self.best_model is None:
                self.select_best_model()
            model = self.best_model
        else:
            model = self.models.get(model_name)
        
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        
        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            return pd.DataFrame()
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_models(self, output_dir: str = 'models/'):
        """Save all trained models"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = f"{output_dir}/{name.replace(' ', '_')}.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Saved {name} to {model_path}")
        
        # Save scaler
        scaler_path = f"{output_dir}/scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")
        
        # Save feature columns
        feature_path = f"{output_dir}/feature_columns.joblib"
        joblib.dump(self.feature_columns, feature_path)
        logger.info(f"Saved feature columns to {feature_path}")
    
    def get_training_report(self) -> Dict:
        """Generate comprehensive training report"""
        if not self.results:
            return {}
        
        report = {
            'models_trained': len(self.results),
            'features_used': len(self.feature_columns),
            'results': self.results,
            'best_model': self.select_best_model() if self.best_model is None else None,
            'feature_importance': self.get_feature_importance().head(10).to_dict('records')
        }
        
        return report


def train_churn_models(data_path: str, mlflow_tracking_uri: str = "mlruns") -> ChurnModelTrainer:
    """
    Main function to train all churn prediction models
    
    Args:
        data_path: Path to engineered features CSV
        mlflow_tracking_uri: MLflow tracking URI
    
    Returns:
        Trained ChurnModelTrainer instance
    """
    logger.info("=" * 60)
    logger.info("ML MODEL TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # Initialize trainer
    trainer = ChurnModelTrainer(mlflow_tracking_uri=mlflow_tracking_uri)
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(df)
    
    # Scale features
    X_train_scaled, X_test_scaled = trainer.scale_features(X_train, X_test)
    
    # Train models
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING MODELS")
    logger.info("=" * 60)
    
    trainer.train_logistic_regression(X_train_scaled, y_train, X_test_scaled, y_test)
    trainer.train_random_forest(X_train_scaled, y_train, X_test_scaled, y_test, n_estimators=100)
    trainer.train_gradient_boosting(X_train_scaled, y_train, X_test_scaled, y_test, n_estimators=100)
    
    # Select best model
    logger.info("\n" + "=" * 60)
    logger.info("MODEL SELECTION")
    logger.info("=" * 60)
    
    best_model = trainer.select_best_model(metric='roc_auc')
    
    # Save models
    trainer.save_models()
    
    # Generate report
    report = trainer.get_training_report()
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    
    return trainer


if __name__ == "__main__":
    # Train all models
    trainer = train_churn_models(
        data_path='data/customers_engineered.csv',
        mlflow_tracking_uri='mlruns'
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 60)
    
    for model_name, metrics in trainer.results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  Net Savings: ${metrics['net_savings']:,.2f}")
    
    print("\n" + "=" * 60)
    print("TOP 10 FEATURE IMPORTANCES")
    print("=" * 60)
    
    importance_df = trainer.get_feature_importance()
    for idx, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    print(f"\n✓ Model training completed successfully!")
    print(f"✓ Models saved to models/ directory")
    print(f"✓ MLflow experiments logged to mlruns/")
