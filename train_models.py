import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb
import joblib
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaggageMLPipeline:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = {}
        self.model_metrics = {}
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        os.makedirs('static/model_data', exist_ok=True)
    
    def load_and_preprocess_damage_data(self):
        """Load and preprocess baggage damage data for flight-level prediction"""
        logger.info("Loading baggage damage data...")
        df = pd.read_csv('attached_assets/refined_baggage_damage_rate_data_1753200261261.csv')
        
        # Convert datetime columns
        df['Damage_Reported_Time'] = pd.to_datetime(df['Damage_Reported_Time'])
        df['Flight_Arrival_Time'] = pd.to_datetime(df['Flight_Arrival_Time'])
        
        # Flight-level aggregation
        flight_data = df.groupby('Flight_Number').agg({
            'Is_Damaged_Bag': ['sum', 'count'],
            'Time_To_Report_Damage': 'mean',
            'Repeat_Zone_Incidents': 'mean',
            'Damage_Reported_Hour': 'mean',
            'Zone_of_Damage': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown',
            'Baggage_Type': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown',
            'Damage_Claim_Type': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
        }).reset_index()
        
        # Flatten column names
        flight_data.columns = ['Flight_Number', 'Total_Damaged_Bags', 'Total_Bags', 
                              'Avg_Report_Time', 'Avg_Repeat_Incidents', 'Avg_Report_Hour',
                              'Primary_Damage_Zone', 'Primary_Baggage_Type', 'Primary_Damage_Type']
        
        # Calculate damage rate
        flight_data['Damage_Rate'] = flight_data['Total_Damaged_Bags'] / flight_data['Total_Bags']
        
        # Create binary target for high damage rate (>10%)
        flight_data['High_Damage_Risk'] = (flight_data['Damage_Rate'] > 0.1).astype(int)
        
        # Encode categorical variables
        le_zone = LabelEncoder()
        le_baggage = LabelEncoder()
        le_damage = LabelEncoder()
        
        flight_data['Zone_Encoded'] = le_zone.fit_transform(flight_data['Primary_Damage_Zone'])
        flight_data['Baggage_Encoded'] = le_baggage.fit_transform(flight_data['Primary_Baggage_Type'])
        flight_data['Damage_Type_Encoded'] = le_damage.fit_transform(flight_data['Primary_Damage_Type'])
        
        # Store encoders
        self.encoders['damage'] = {
            'zone': le_zone,
            'baggage': le_baggage,
            'damage_type': le_damage
        }
        
        features = ['Total_Bags', 'Avg_Report_Time', 'Avg_Repeat_Incidents', 
                   'Avg_Report_Hour', 'Zone_Encoded', 'Baggage_Encoded', 'Damage_Type_Encoded']
        
        return flight_data[features], flight_data['High_Damage_Risk'], flight_data
    
    def load_and_preprocess_downtime_data(self):
        """Load and preprocess system downtime data"""
        logger.info("Loading system downtime data...")
        df = pd.read_csv('attached_assets/refined_baggage_system_downtime_data_1753200261261.csv')
        
        # Convert datetime columns
        df['Incident_Start_Time'] = pd.to_datetime(df['Incident_Start_Time'])
        df['Incident_End_Time'] = pd.to_datetime(df['Incident_End_Time'])
        
        # Aggregate by location and day
        df['Date'] = df['Incident_Start_Time'].dt.date
        daily_data = df.groupby(['Location_Affected', 'Date']).agg({
            'Incident_Duration_Minutes': ['sum', 'count', 'max'],
            'SLA_Compliance_Downtime': 'mean',
            'Maintenance_Team_Response_Time': 'mean',
            'Incident_Type': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown',
            'Root_Cause_Code': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
        }).reset_index()
        
        # Flatten column names
        daily_data.columns = ['Location', 'Date', 'Total_Downtime', 'Incident_Count', 
                             'Max_Incident_Duration', 'Avg_SLA_Compliance', 'Avg_Response_Time',
                             'Primary_Incident_Type', 'Primary_Root_Cause']
        
        # Create target for high downtime risk (>60 minutes total daily downtime)
        daily_data['High_Downtime_Risk'] = (daily_data['Total_Downtime'] > 60).astype(int)
        
        # Encode categorical variables
        le_location = LabelEncoder()
        le_incident = LabelEncoder()
        le_cause = LabelEncoder()
        
        daily_data['Location_Encoded'] = le_location.fit_transform(daily_data['Location'])
        daily_data['Incident_Type_Encoded'] = le_incident.fit_transform(daily_data['Primary_Incident_Type'])
        daily_data['Root_Cause_Encoded'] = le_cause.fit_transform(daily_data['Primary_Root_Cause'])
        
        self.encoders['downtime'] = {
            'location': le_location,
            'incident': le_incident,
            'cause': le_cause
        }
        
        features = ['Incident_Count', 'Max_Incident_Duration', 'Avg_SLA_Compliance',
                   'Avg_Response_Time', 'Location_Encoded', 'Incident_Type_Encoded', 'Root_Cause_Encoded']
        
        return daily_data[features], daily_data['High_Downtime_Risk'], daily_data
    
    def load_and_preprocess_departure_data(self):
        """Load and preprocess departure load data"""
        logger.info("Loading departure load data...")
        df = pd.read_csv('attached_assets/refined_departure_load_data_1753200261261.csv')
        
        # Convert datetime columns
        df['Checkin_Time'] = pd.to_datetime(df['Checkin_Time'])
        df['Bag_Loaded_on_Aircraft_Time'] = pd.to_datetime(df['Bag_Loaded_on_Aircraft_Time'])
        df['Flight_Scheduled_Departure_Time'] = pd.to_datetime(df['Flight_Scheduled_Departure_Time'])
        
        # Extract flight number from Bag_Tag_ID or use a synthetic approach
        df['Flight_ID'] = df['Bag_Tag_ID'].str[:6]  # Use first 6 chars as flight identifier
        
        # Flight-level aggregation
        flight_data = df.groupby('Flight_ID').agg({
            'Late_Load_Flag': ['sum', 'count'],
            'Bag_Load_Lead_Time_Minutes': 'mean',
            'Checkin_to_Load_Duration_Minutes': 'mean',
            'Trolley_Turnaround_Time': 'mean',
            'Load_Team_Assigned': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown',
            'Bag_Routing_Status': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown',
            'Bag_Screening_Status': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
        }).reset_index()
        
        # Flatten column names
        flight_data.columns = ['Flight_ID', 'Late_Load_Count', 'Total_Bags',
                              'Avg_Load_Lead_Time', 'Avg_Checkin_Duration', 'Avg_Turnaround_Time',
                              'Primary_Team', 'Primary_Routing', 'Primary_Screening']
        
        # Calculate late load rate
        flight_data['Late_Load_Rate'] = flight_data['Late_Load_Count'] / flight_data['Total_Bags']
        
        # Create target for high late load risk (>20%)
        flight_data['High_Late_Load_Risk'] = (flight_data['Late_Load_Rate'] > 0.2).astype(int)
        
        # Encode categorical variables
        le_team = LabelEncoder()
        le_routing = LabelEncoder()
        le_screening = LabelEncoder()
        
        flight_data['Team_Encoded'] = le_team.fit_transform(flight_data['Primary_Team'])
        flight_data['Routing_Encoded'] = le_routing.fit_transform(flight_data['Primary_Routing'])
        flight_data['Screening_Encoded'] = le_screening.fit_transform(flight_data['Primary_Screening'])
        
        self.encoders['departure'] = {
            'team': le_team,
            'routing': le_routing,
            'screening': le_screening
        }
        
        features = ['Total_Bags', 'Avg_Load_Lead_Time', 'Avg_Checkin_Duration',
                   'Avg_Turnaround_Time', 'Team_Encoded', 'Routing_Encoded', 'Screening_Encoded']
        
        return flight_data[features], flight_data['High_Late_Load_Risk'], flight_data
    
    def load_and_preprocess_mishandled_data(self):
        """Load and preprocess mishandled baggage data"""
        logger.info("Loading mishandled baggage data...")
        df = pd.read_csv('attached_assets/refined_mishandled_baggage_data__1753200261262.csv')
        
        # Convert datetime columns
        df['Passenger_Arrival_Time'] = pd.to_datetime(df['Passenger_Arrival_Time'])
        df['Final_Delivery_Time'] = pd.to_datetime(df['Final_Delivery_Time'])
        
        # Flight-level aggregation
        flight_data = df.groupby('Flight_Number').agg({
            'Is_Mishandled': ['sum', 'count'],
            'Delivery_Delay_Minutes': 'mean',
            'Transfer_Flight_Count': 'mean',
            'Transfer_Risk_Flag': 'sum',
            'Sensor_Tracking_Gap_Flag': 'sum',
            'Delivery_Status': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown',
            'Baggage_Tracking_Sensor_Data': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
        }).reset_index()
        
        # Flatten column names
        flight_data.columns = ['Flight_Number', 'Mishandled_Count', 'Total_Bags',
                              'Avg_Delivery_Delay', 'Avg_Transfer_Count', 'Transfer_Risk_Count',
                              'Sensor_Gap_Count', 'Primary_Delivery_Status', 'Primary_Sensor_Data']
        
        # Calculate mishandled rate
        flight_data['Mishandled_Rate'] = flight_data['Mishandled_Count'] / flight_data['Total_Bags']
        
        # Create target for high mishandled risk (>5%)
        flight_data['High_Mishandled_Risk'] = (flight_data['Mishandled_Rate'] > 0.05).astype(int)
        
        # Encode categorical variables
        le_delivery = LabelEncoder()
        le_sensor = LabelEncoder()
        
        flight_data['Delivery_Encoded'] = le_delivery.fit_transform(flight_data['Primary_Delivery_Status'])
        flight_data['Sensor_Encoded'] = le_sensor.fit_transform(flight_data['Primary_Sensor_Data'])
        
        self.encoders['mishandled'] = {
            'delivery': le_delivery,
            'sensor': le_sensor
        }
        
        features = ['Total_Bags', 'Avg_Delivery_Delay', 'Avg_Transfer_Count',
                   'Transfer_Risk_Count', 'Sensor_Gap_Count', 'Delivery_Encoded', 'Sensor_Encoded']
        
        return flight_data[features], flight_data['High_Mishandled_Risk'], flight_data
    
    def load_and_preprocess_transfer_data(self):
        """Load and preprocess transfer baggage SLA data"""
        logger.info("Loading transfer baggage data...")
        df = pd.read_csv('attached_assets/refined_transfer_baggage_sla_data_1753200261262.csv')
        
        # Convert datetime columns
        df['Inbound_Arrival_Time'] = pd.to_datetime(df['Inbound_Arrival_Time'])
        df['Outbound_Departure_Time'] = pd.to_datetime(df['Outbound_Departure_Time'])
        
        # Flight pair aggregation (inbound-outbound combination)
        flight_data = df.groupby(['Inbound_Flight_Number', 'Outbound_Flight_Number']).agg({
            'SLA_Compliance': ['mean', 'count'],
            'Transfer_Buffer_Minutes': 'mean',
            'Bag_Transfer_Time': 'mean',
            'Gate_to_Gate_Distance': 'mean',
            'Airport_MCT_for_Transfer': 'mean',
            'Transfer_Path_Type': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
        }).reset_index()
        
        # Flatten column names
        flight_data.columns = ['Inbound_Flight', 'Outbound_Flight', 'SLA_Compliance_Rate', 'Transfer_Count',
                              'Avg_Buffer_Minutes', 'Avg_Transfer_Time', 'Avg_Gate_Distance',
                              'Avg_MCT', 'Primary_Path_Type']
        
        # Create target for SLA compliance risk (<90% compliance)
        flight_data['SLA_Risk'] = (flight_data['SLA_Compliance_Rate'] < 0.9).astype(int)
        
        # Encode categorical variables
        le_path = LabelEncoder()
        flight_data['Path_Type_Encoded'] = le_path.fit_transform(flight_data['Primary_Path_Type'])
        
        self.encoders['transfer'] = {
            'path': le_path
        }
        
        features = ['Transfer_Count', 'Avg_Buffer_Minutes', 'Avg_Transfer_Time',
                   'Avg_Gate_Distance', 'Avg_MCT', 'Path_Type_Encoded']
        
        return flight_data[features], flight_data['SLA_Risk'], flight_data
    
    def train_model(self, X, y, model_name):
        """Train XGBoost model and return metrics"""
        logger.info(f"Training {model_name} model...")
        
        # Check if we have enough samples for stratification
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        except ValueError:
            # Fallback if stratification fails due to low sample count
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Calculate class distribution and base score for downtime model
        if model_name == 'downtime':
            positive_rate = y_train.mean()
            # Ensure base_score is within valid range for logistic loss
            base_score = max(0.01, min(0.99, positive_rate)) if positive_rate > 0 else 0.5
            
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                base_score=base_score,
                scale_pos_weight=(1 - positive_rate) / positive_rate if positive_rate > 0 else 1
            )
        else:
            # Standard XGBoost configuration for other models
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
        
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'feature_importance': model.feature_importances_.tolist(),
            'feature_names': X.columns.tolist()
        }
        
        # Store model and scaler
        self.models[model_name] = model
        self.scalers[model_name] = scaler
        self.feature_names[model_name] = X.columns.tolist()
        self.model_metrics[model_name] = metrics
        
        logger.info(f"{model_name} model - Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}")
        
        return metrics
    
    def save_models(self):
        """Save all trained models and preprocessors"""
        logger.info("Saving models...")
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, f'models/{name}_model.pkl')
            joblib.dump(self.scalers[name], f'models/{name}_scaler.pkl')
        
        # Save encoders
        joblib.dump(self.encoders, 'models/encoders.pkl')
        
        # Save feature names
        joblib.dump(self.feature_names, 'models/feature_names.pkl')
        
        # Save metrics for web display
        joblib.dump(self.model_metrics, 'static/model_data/metrics.pkl')
        
        logger.info("All models saved successfully!")
    
    def train_all_models(self):
        """Train all baggage prediction models"""
        logger.info("Starting baggage ML pipeline...")
        
        # Train damage prediction model
        try:
            X_damage, y_damage, damage_data = self.load_and_preprocess_damage_data()
            self.train_model(X_damage, y_damage, 'damage')
        except Exception as e:
            logger.error(f"Error training damage model: {e}")
        
        # Train downtime prediction model
        try:
            X_downtime, y_downtime, downtime_data = self.load_and_preprocess_downtime_data()
            self.train_model(X_downtime, y_downtime, 'downtime')
        except Exception as e:
            logger.error(f"Error training downtime model: {e}")
        
        # Train departure prediction model
        try:
            X_departure, y_departure, departure_data = self.load_and_preprocess_departure_data()
            self.train_model(X_departure, y_departure, 'departure')
        except Exception as e:
            logger.error(f"Error training departure model: {e}")
        
        # Train mishandled prediction model
        try:
            X_mishandled, y_mishandled, mishandled_data = self.load_and_preprocess_mishandled_data()
            self.train_model(X_mishandled, y_mishandled, 'mishandled')
        except Exception as e:
            logger.error(f"Error training mishandled model: {e}")
        
        # Train transfer SLA prediction model
        try:
            X_transfer, y_transfer, transfer_data = self.load_and_preprocess_transfer_data()
            self.train_model(X_transfer, y_transfer, 'transfer')
        except Exception as e:
            logger.error(f"Error training transfer model: {e}")
        
        # Save all models
        self.save_models()
        
        return self.model_metrics

if __name__ == "__main__":
    pipeline = BaggageMLPipeline()
    metrics = pipeline.train_all_models()
    
    print("\n=== Model Training Complete ===")
    for model_name, metric in metrics.items():
        print(f"{model_name.upper()} Model:")
        print(f"  Accuracy: {metric['accuracy']:.3f}")
        print(f"  F1 Score: {metric['f1']:.3f}")
        print(f"  Precision: {metric['precision']:.3f}")
        print(f"  Recall: {metric['recall']:.3f}")
        print()
