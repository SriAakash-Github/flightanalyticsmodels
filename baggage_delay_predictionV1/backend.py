from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load the trained pipeline
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
# Load the regression model
with open('model_reg.pkl', 'rb') as f:
    model_reg = pickle.load(f)

# List of expected raw input fields (remove actual_arrival_time, check_out_time)
RAW_FIELDS = [
    'origin_airport', 'destination_airport',
    'departure_time', 'actual_departure_time',
    'arrival_time', 'check_in_time',
    'weather_condition', 'number_of_bags', 'bag_weight_kg',
    'day_of_week', 'is_international', 'extra_baggage'
]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    # Parse only the time (HH:MM) for all time fields
    for col in ['departure_time', 'actual_departure_time', 'arrival_time', 'check_in_time']:
        df[col] = pd.to_datetime(df[col].astype(str).str[-5:], format='%H:%M', errors='coerce')
        df[f'{col}_hour'] = df[col].dt.hour
        df[f'{col}_minute'] = df[col].dt.minute
    df['route'] = df['origin_airport'] + '-' + df['destination_airport']
    # df['membership_tier'] = df['membership_tier'].replace('None', 'Non-member')
    df['day_of_week'] = df['day_of_week'].astype(str)
    FEATURES = [
    'origin_airport', 'destination_airport', 'route',
    'weather_condition', 'number_of_bags', 'bag_weight_kg',
    'day_of_week', 'is_international', 'extra_baggage',
    'departure_time_hour', 'departure_time_minute',
    'actual_departure_time_hour', 'actual_departure_time_minute',
    'arrival_time_hour', 'arrival_time_minute',
    'check_in_time_hour', 'check_in_time_minute',
    'travel_duration_min'
]
    X = df[FEATURES]
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]
    return jsonify({
        'prediction': int(pred),
        'probability': float(proba)
    })

@app.route('/predict_reg', methods=['POST'])
def predict_reg():
    data = request.json
    df = pd.DataFrame([data])
    # Parse only the time (HH:MM) for all time fields
    for col in ['departure_time', 'actual_departure_time', 'arrival_time', 'check_in_time']:
        df[col] = pd.to_datetime(df[col].astype(str).str[-5:], format='%H:%M', errors='coerce')
        df[f'{col}_hour'] = df[col].dt.hour
        df[f'{col}_minute'] = df[col].dt.minute
    df['route'] = df['origin_airport'] + '-' + df['destination_airport']
    df['day_of_week'] = df['day_of_week'].astype(str)
    FEATURES = [
        'origin_airport', 'destination_airport', 'route',
        'weather_condition', 'number_of_bags', 'bag_weight_kg',
        'day_of_week', 'is_international', 'extra_baggage',
        'departure_time_hour', 'departure_time_minute',
        'actual_departure_time_hour', 'actual_departure_time_minute',
        'arrival_time_hour', 'arrival_time_minute',
        'check_in_time_hour', 'check_in_time_minute',
        'travel_duration_min'
    ]
    X = df[FEATURES]
    log_delay_pred = model_reg.predict(X)[0]
    delay_minutes = float(np.expm1(log_delay_pred))
    return jsonify({
        'predicted_delay_minutes': delay_minutes
    })

if __name__ == '__main__':
    app.run(debug=True) 