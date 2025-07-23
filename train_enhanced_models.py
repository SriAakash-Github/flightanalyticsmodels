#!/usr/bin/env python3
"""
Enhanced ML Model Training Script
Simplified for WebContainer environment
"""

import json
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

def create_sample_data():
    """Create sample training data for demonstration"""
    np.random.seed(42)
    
    # Sample data for flight delay prediction
    n_samples = 1000
    
    # Features: weather_score, traffic_density, aircraft_age, route_complexity
    X = np.random.rand(n_samples, 4)
    X[:, 0] = X[:, 0] * 10  # weather_score 0-10
    X[:, 1] = X[:, 1] * 100  # traffic_density 0-100
    X[:, 2] = X[:, 2] * 20  # aircraft_age 0-20 years
    X[:, 3] = X[:, 3] * 5   # route_complexity 0-5
    
    # Target: delay in minutes (0 means no delay)
    y_delay = np.random.exponential(scale=30, size=n_samples)
    y_delay = np.clip(y_delay, 0, 300)  # Cap at 5 hours
    
    # Binary classification: delayed or not
    y_binary = (y_delay > 15).astype(int)  # Delayed if > 15 minutes
    
    return X, y_binary, y_delay

def train_models():
    """Train all ML models"""
    print("Creating sample training data...")
    X, y_binary, y_delay = create_sample_data()
    
    # Split data
    X_train, X_test, y_bin_train, y_bin_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42
    )
    _, _, y_reg_train, y_reg_test = train_test_split(
        X, y_delay, test_size=0.2, random_state=42
    )
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    models_info = {}
    
    # 1. Flight Delay Classification Model
    print("Training Flight Delay Classification Model...")
    flight_delay_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    flight_delay_clf.fit(X_train, y_bin_train)
    
    # Save model
    with open('models/flight_delay_model.pkl', 'wb') as f:
        pickle.dump(flight_delay_clf, f)
    
    # Calculate accuracy
    y_pred = flight_delay_clf.predict(X_test)
    accuracy = accuracy_score(y_bin_test, y_pred)
    models_info['flight_delay'] = {'accuracy': accuracy, 'type': 'classification'}
    
    # 2. Flight Delay Regression Model (for delay time)
    print("Training Flight Delay Regression Model...")
    flight_delay_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    flight_delay_reg.fit(X_train, y_reg_train)
    
    with open('models/flight_delay_time_model.pkl', 'wb') as f:
        pickle.dump(flight_delay_reg, f)
    
    y_pred_reg = flight_delay_reg.predict(X_test)
    mse = mean_squared_error(y_reg_test, y_pred_reg)
    models_info['flight_delay_time'] = {'mse': mse, 'type': 'regression'}
    
    # 3. Damage Prediction Model
    print("Training Damage Prediction Model...")
    damage_model = RandomForestClassifier(n_estimators=100, random_state=42)
    y_damage = np.random.binomial(1, 0.1, len(X_train))  # 10% damage rate
    damage_model.fit(X_train, y_damage)
    
    with open('models/damage_model.pkl', 'wb') as f:
        pickle.dump(damage_model, f)
    
    models_info['damage'] = {'type': 'classification'}
    
    # 4. Downtime Prediction Model
    print("Training Downtime Prediction Model...")
    downtime_model = RandomForestRegressor(n_estimators=100, random_state=42)
    y_downtime = np.random.exponential(scale=60, size=len(X_train))  # Minutes
    downtime_model.fit(X_train, y_downtime)
    
    with open('models/downtime_model.pkl', 'wb') as f:
        pickle.dump(downtime_model, f)
    
    models_info['downtime'] = {'type': 'regression'}
    
    # 5. Departure Delay Model
    print("Training Departure Delay Model...")
    departure_model = RandomForestRegressor(n_estimators=100, random_state=42)
    y_departure = np.random.exponential(scale=20, size=len(X_train))
    departure_model.fit(X_train, y_departure)
    
    with open('models/departure_model.pkl', 'wb') as f:
        pickle.dump(departure_model, f)
    
    models_info['departure'] = {'type': 'regression'}
    
    # 6. Mishandled Baggage Model
    print("Training Mishandled Baggage Model...")
    mishandled_model = RandomForestClassifier(n_estimators=100, random_state=42)
    y_mishandled = np.random.binomial(1, 0.05, len(X_train))  # 5% mishandled rate
    mishandled_model.fit(X_train, y_mishandled)
    
    with open('models/mishandled_model.pkl', 'wb') as f:
        pickle.dump(mishandled_model, f)
    
    models_info['mishandled'] = {'type': 'classification'}
    
    # 7. Transfer SLA Model
    print("Training Transfer SLA Model...")
    transfer_model = RandomForestClassifier(n_estimators=100, random_state=42)
    y_transfer = np.random.binomial(1, 0.85, len(X_train))  # 85% meet SLA
    transfer_model.fit(X_train, y_transfer)
    
    with open('models/transfer_model.pkl', 'wb') as f:
        pickle.dump(transfer_model, f)
    
    models_info['transfer'] = {'type': 'classification'}
    
    # Save model info
    with open('models/models_info.json', 'w') as f:
        json.dump(models_info, f, indent=2)
    
    print("All models trained successfully!")
    print(f"Models saved in 'models/' directory")
    
    return models_info

if __name__ == "__main__":
    models_info = train_models()
    print("\nModel Training Summary:")
    for model_name, info in models_info.items():
        print(f"- {model_name}: {info}")