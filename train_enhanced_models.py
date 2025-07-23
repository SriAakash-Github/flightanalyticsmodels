import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import joblib
import os
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedBaggageMLPipeline:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = {}
        self.model_metrics = {}
        self.pipelines = {}
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('static/model_data', exist_ok=True)
    
    def load_and_preprocess_arrival_delivery_data(self):
        """Load and preprocess arrival delivery data for flight-level prediction"""
        logger.info("Loading arrival delivery data...")
        try:
            df = pd.read_csv('attached_assets/refined_arrival_delivery_data_1753200261261.csv')
            
            # Convert datetime columns
            datetime_cols = ['Flight_Arrival_Time', 'Baggage_Delivery_Time', 'Passenger_Arrival_Time']
            for col in datetime_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Create flight-level aggregation
            flight_data = df.groupby('Flight_Number').agg({
                'Delivery_Delay_Minutes': ['mean', 'max', 'std'],
                'Passenger_Count': 'first',
                'Baggage_Count': 'first',
                'Weather_Condition': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Clear',
                'Day_of_Week': 'first',
                'Is_International_Flight': 'first',
                'Aircraft_Type': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown',
                'Origin_Airport': 'first',
                'Destination_Airport': 'first'
            }).reset_index()
            
            # Flatten column names
            flight_data.columns = ['Flight_Number', 'Avg_Delivery_Delay', 'Max_Delivery_Delay', 'Std_Delivery_Delay',
                                  'Passenger_Count', 'Baggage_Count', 'Weather_Condition', 'Day_of_Week',
                                  'Is_International_Flight', 'Aircraft_Type', 'Origin_Airport', 'Destination_Airport']
            
            # Fill NaN values
            flight_data['Std_Delivery_Delay'] = flight_data['Std_Delivery_Delay'].fillna(0)
            
            # Create target variable (delayed if avg delay > 30 minutes)
            flight_data['Is_Delayed'] = (flight_data['Avg_Delivery_Delay'] > 30).astype(int)
            
            # Select features for training
            feature_cols = ['Passenger_Count', 'Baggage_Count', 'Weather_Condition', 'Day_of_Week',
                           'Is_International_Flight', 'Aircraft_Type', 'Origin_Airport', 'Destination_Airport',
                           'Max_Delivery_Delay', 'Std_Delivery_Delay']
            
            return flight_data[feature_cols], flight_data['Is_Delayed'], flight_data
            
        except Exception as e:
            logger.error(f"Error loading arrival delivery data: {e}")
            return None, None, None
    
    def create_preprocessing_pipeline(self, X):
        """Create preprocessing pipeline for mixed data types"""
        # Identify categorical and numerical columns
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Create preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
            ])
        
        return preprocessor, categorical_features, numerical_features
    
    def train_enhanced_model(self, X, y, model_name, use_grid_search=False):
        """Train enhanced model with preprocessing pipeline"""
        logger.info(f"Training enhanced {model_name} model...")
        
        if X is None or y is None:
            logger.error(f"No data available for {model_name} model")
            return None
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Create preprocessing pipeline
            preprocessor, cat_features, num_features = self.create_preprocessing_pipeline(X)
            
            # Create model pipeline
            if use_grid_search:
                # Use GridSearchCV for hyperparameter tuning
                model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
                param_grid = {
                    'classifier__n_estimators': [100, 200],
                    'classifier__max_depth': [3, 6, 9],
                    'classifier__learning_rate': [0.01, 0.1, 0.2]
                }
                
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', model)
                ])
                
                grid_search = GridSearchCV(
                    pipeline, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                best_pipeline = grid_search.best_estimator_
                
            else:
                # Use default parameters
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    eval_metric='logloss'
                )
                
                best_pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', model)
                ])
                best_pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred = best_pipeline.predict(X_test)
            y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                'feature_names': list(X.columns),
                'categorical_features': cat_features,
                'numerical_features': num_features
            }
            
            # Get feature importance if available
            if hasattr(best_pipeline.named_steps['classifier'], 'feature_importances_'):
                # Get feature names after preprocessing
                feature_names = (num_features + 
                               list(best_pipeline.named_steps['preprocessor']
                                   .named_transformers_['cat']
                                   .get_feature_names_out(cat_features)))
                metrics['feature_importance'] = best_pipeline.named_steps['classifier'].feature_importances_.tolist()
                metrics['processed_feature_names'] = feature_names
            
            # Store pipeline and metrics
            self.pipelines[model_name] = best_pipeline
            self.model_metrics[model_name] = metrics
            
            logger.info(f"{model_name} model - Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training {model_name} model: {e}")
            return None
    
    def train_all_enhanced_models(self):
        """Train all enhanced models with available datasets"""
        logger.info("Starting enhanced baggage ML pipeline...")
        
        # Train existing models (from original pipeline)
        try:
            from train_models import BaggageMLPipeline
            original_pipeline = BaggageMLPipeline()
            
            # Train damage prediction model
            try:
                X_damage, y_damage, damage_data = original_pipeline.load_and_preprocess_damage_data()
                self.train_enhanced_model(X_damage, y_damage, 'damage')
            except Exception as e:
                logger.error(f"Error training damage model: {e}")
            
            # Train downtime prediction model
            try:
                X_downtime, y_downtime, downtime_data = original_pipeline.load_and_preprocess_downtime_data()
                self.train_enhanced_model(X_downtime, y_downtime, 'downtime')
            except Exception as e:
                logger.error(f"Error training downtime model: {e}")
            
            # Train departure prediction model
            try:
                X_departure, y_departure, departure_data = original_pipeline.load_and_preprocess_departure_data()
                self.train_enhanced_model(X_departure, y_departure, 'departure')
            except Exception as e:
                logger.error(f"Error training departure model: {e}")
            
            # Train mishandled prediction model
            try:
                X_mishandled, y_mishandled, mishandled_data = original_pipeline.load_and_preprocess_mishandled_data()
                self.train_enhanced_model(X_mishandled, y_mishandled, 'mishandled')
            except Exception as e:
                logger.error(f"Error training mishandled model: {e}")
            
            # Train transfer SLA prediction model
            try:
                X_transfer, y_transfer, transfer_data = original_pipeline.load_and_preprocess_transfer_data()
                self.train_enhanced_model(X_transfer, y_transfer, 'transfer')
            except Exception as e:
                logger.error(f"Error training transfer model: {e}")
                
        except Exception as e:
            logger.error(f"Error with original pipeline: {e}")
        
        # Train new arrival delivery model
        try:
            X_arrival, y_arrival, arrival_data = self.load_and_preprocess_arrival_delivery_data()
            if X_arrival is not None:
                self.train_enhanced_model(X_arrival, y_arrival, 'arrival_delivery')
        except Exception as e:
            logger.error(f"Error training arrival delivery model: {e}")
        
        # Save all models
        self.save_enhanced_models()
        
        return self.model_metrics
    
    def save_enhanced_models(self):
        """Save all trained models and pipelines"""
        logger.info("Saving enhanced models...")
        
        # Save pipelines
        for name, pipeline in self.pipelines.items():
            joblib.dump(pipeline, f'models/{name}_pipeline.pkl')
        
        # Save metrics for web display
        joblib.dump(self.model_metrics, 'static/model_data/enhanced_metrics.pkl')
        
        logger.info("All enhanced models saved successfully!")

if __name__ == "__main__":
    pipeline = EnhancedBaggageMLPipeline()
    metrics = pipeline.train_all_enhanced_models()
    
    print("\n=== Enhanced Model Training Complete ===")
    for model_name, metric in metrics.items():
        if metric:
            print(f"{model_name.upper()} Model:")
            print(f"  Accuracy: {metric['accuracy']:.3f}")
            print(f"  F1 Score: {metric['f1']:.3f}")
            print(f"  Precision: {metric['precision']:.3f}")
            print(f"  Recall: {metric['recall']:.3f}")
            print()