from flask import render_template, request, jsonify, flash, redirect, url_for
import joblib
import numpy as np
import pandas as pd
import os
import logging
from app import app

logger = logging.getLogger(__name__)

class ModelPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = {}
        self.metrics = {}
        self.load_models()
    
    def load_models(self):
        """Load all trained models and preprocessors"""
        try:
            model_names = ['damage', 'downtime', 'departure', 'mishandled', 'transfer']
            
            for name in model_names:
                model_path = f'models/{name}_model.pkl'
                scaler_path = f'models/{name}_scaler.pkl'
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    self.models[name] = joblib.load(model_path)
                    self.scalers[name] = joblib.load(scaler_path)
                    logger.info(f"Loaded {name} model successfully")
                else:
                    logger.warning(f"Model files not found for {name}")
            
            # Load encoders and feature names
            if os.path.exists('models/encoders.pkl'):
                self.encoders = joblib.load('models/encoders.pkl')
            
            if os.path.exists('models/feature_names.pkl'):
                self.feature_names = joblib.load('models/feature_names.pkl')
            
            # Load metrics
            if os.path.exists('static/model_data/metrics.pkl'):
                self.metrics = joblib.load('static/model_data/metrics.pkl')
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")

# Initialize predictor
predictor = ModelPredictor()

@app.route('/')
def index():
    """Main dashboard showing all available predictions"""
    return render_template('index.html', metrics=predictor.metrics)

@app.route('/damage_prediction', methods=['GET', 'POST'])
def damage_prediction():
    """Baggage damage risk prediction"""
    if request.method == 'POST':
        try:
            # Get form data
            total_bags = float(request.form['total_bags'])
            avg_report_time = float(request.form['avg_report_time'])
            avg_repeat_incidents = float(request.form['avg_repeat_incidents'])
            avg_report_hour = float(request.form['avg_report_hour'])
            damage_zone = request.form['damage_zone']
            baggage_type = request.form['baggage_type']
            damage_type = request.form['damage_type']
            
            # Encode categorical variables
            zone_encoded = 0
            baggage_encoded = 0
            damage_type_encoded = 0
            
            if 'damage' in predictor.encoders:
                try:
                    zone_encoded = predictor.encoders['damage']['zone'].transform([damage_zone])[0]
                    baggage_encoded = predictor.encoders['damage']['baggage'].transform([baggage_type])[0]
                    damage_type_encoded = predictor.encoders['damage']['damage_type'].transform([damage_type])[0]
                except:
                    # Handle unknown categories
                    pass
            
            # Prepare features
            features = np.array([[total_bags, avg_report_time, avg_repeat_incidents, 
                                avg_report_hour, zone_encoded, baggage_encoded, damage_type_encoded]])
            
            # Make prediction
            if 'damage' in predictor.models:
                # Scale features
                features_scaled = predictor.scalers['damage'].transform(features)
                
                # Predict
                prediction = predictor.models['damage'].predict(features_scaled)[0]
                probability = predictor.models['damage'].predict_proba(features_scaled)[0]
                
                result = {
                    'prediction': int(prediction),
                    'risk_level': 'High' if prediction == 1 else 'Low',
                    'confidence': float(max(probability)) * 100,
                    'low_risk_prob': float(probability[0]) * 100,
                    'high_risk_prob': float(probability[1]) * 100 if len(probability) > 1 else 0
                }
                
                return render_template('damage_prediction.html', result=result, 
                                     form_data=request.form)
            else:
                flash('Damage prediction model not available', 'error')
                
        except Exception as e:
            logger.error(f"Error in damage prediction: {e}")
            flash(f'Error making prediction: {str(e)}', 'error')
    
    return render_template('damage_prediction.html')

@app.route('/downtime_prediction', methods=['GET', 'POST'])
def downtime_prediction():
    """System downtime risk prediction"""
    if request.method == 'POST':
        try:
            # Get form data
            incident_count = int(request.form['incident_count'])
            max_incident_duration = float(request.form['max_incident_duration'])
            avg_sla_compliance = float(request.form['avg_sla_compliance'])
            avg_response_time = float(request.form['avg_response_time'])
            location = request.form['location']
            incident_type = request.form['incident_type']
            root_cause = request.form['root_cause']
            
            # Encode categorical variables
            location_encoded = 0
            incident_encoded = 0
            cause_encoded = 0
            
            if 'downtime' in predictor.encoders:
                try:
                    location_encoded = predictor.encoders['downtime']['location'].transform([location])[0]
                    incident_encoded = predictor.encoders['downtime']['incident'].transform([incident_type])[0]
                    cause_encoded = predictor.encoders['downtime']['cause'].transform([root_cause])[0]
                except:
                    pass
            
            # Prepare features
            features = np.array([[incident_count, max_incident_duration, avg_sla_compliance,
                                avg_response_time, location_encoded, incident_encoded, cause_encoded]])
            
            # Make prediction
            if 'downtime' in predictor.models:
                features_scaled = predictor.scalers['downtime'].transform(features)
                prediction = predictor.models['downtime'].predict(features_scaled)[0]
                probability = predictor.models['downtime'].predict_proba(features_scaled)[0]
                
                result = {
                    'prediction': int(prediction),
                    'risk_level': 'High' if prediction == 1 else 'Low',
                    'confidence': float(max(probability)) * 100,
                    'low_risk_prob': float(probability[0]) * 100,
                    'high_risk_prob': float(probability[1]) * 100 if len(probability) > 1 else 0
                }
                
                return render_template('downtime_prediction.html', result=result,
                                     form_data=request.form)
            else:
                flash('Downtime prediction model not available', 'error')
                
        except Exception as e:
            logger.error(f"Error in downtime prediction: {e}")
            flash(f'Error making prediction: {str(e)}', 'error')
    
    return render_template('downtime_prediction.html')

@app.route('/departure_prediction', methods=['GET', 'POST'])
def departure_prediction():
    """Departure delay risk prediction"""
    if request.method == 'POST':
        try:
            # Get form data
            total_bags = int(request.form['total_bags'])
            avg_load_lead_time = float(request.form['avg_load_lead_time'])
            avg_checkin_duration = float(request.form['avg_checkin_duration'])
            avg_turnaround_time = float(request.form['avg_turnaround_time'])
            team = request.form['team']
            routing_status = request.form['routing_status']
            screening_status = request.form['screening_status']
            
            # Encode categorical variables
            team_encoded = 0
            routing_encoded = 0
            screening_encoded = 0
            
            if 'departure' in predictor.encoders:
                try:
                    team_encoded = predictor.encoders['departure']['team'].transform([team])[0]
                    routing_encoded = predictor.encoders['departure']['routing'].transform([routing_status])[0]
                    screening_encoded = predictor.encoders['departure']['screening'].transform([screening_status])[0]
                except:
                    pass
            
            # Prepare features
            features = np.array([[total_bags, avg_load_lead_time, avg_checkin_duration,
                                avg_turnaround_time, team_encoded, routing_encoded, screening_encoded]])
            
            # Make prediction
            if 'departure' in predictor.models:
                features_scaled = predictor.scalers['departure'].transform(features)
                prediction = predictor.models['departure'].predict(features_scaled)[0]
                probability = predictor.models['departure'].predict_proba(features_scaled)[0]
                
                result = {
                    'prediction': int(prediction),
                    'risk_level': 'High' if prediction == 1 else 'Low',
                    'confidence': float(max(probability)) * 100,
                    'low_risk_prob': float(probability[0]) * 100,
                    'high_risk_prob': float(probability[1]) * 100 if len(probability) > 1 else 0
                }
                
                return render_template('departure_prediction.html', result=result,
                                     form_data=request.form)
            else:
                flash('Departure prediction model not available', 'error')
                
        except Exception as e:
            logger.error(f"Error in departure prediction: {e}")
            flash(f'Error making prediction: {str(e)}', 'error')
    
    return render_template('departure_prediction.html')

@app.route('/mishandled_prediction', methods=['GET', 'POST'])
def mishandled_prediction():
    """Mishandled baggage risk prediction"""
    if request.method == 'POST':
        try:
            # Get form data
            total_bags = int(request.form['total_bags'])
            avg_delivery_delay = float(request.form['avg_delivery_delay'])
            avg_transfer_count = float(request.form['avg_transfer_count'])
            transfer_risk_count = int(request.form['transfer_risk_count'])
            sensor_gap_count = int(request.form['sensor_gap_count'])
            delivery_status = request.form['delivery_status']
            sensor_data = request.form['sensor_data']
            
            # Encode categorical variables
            delivery_encoded = 0
            sensor_encoded = 0
            
            if 'mishandled' in predictor.encoders:
                try:
                    delivery_encoded = predictor.encoders['mishandled']['delivery'].transform([delivery_status])[0]
                    sensor_encoded = predictor.encoders['mishandled']['sensor'].transform([sensor_data])[0]
                except:
                    pass
            
            # Prepare features
            features = np.array([[total_bags, avg_delivery_delay, avg_transfer_count,
                                transfer_risk_count, sensor_gap_count, delivery_encoded, sensor_encoded]])
            
            # Make prediction
            if 'mishandled' in predictor.models:
                features_scaled = predictor.scalers['mishandled'].transform(features)
                prediction = predictor.models['mishandled'].predict(features_scaled)[0]
                probability = predictor.models['mishandled'].predict_proba(features_scaled)[0]
                
                result = {
                    'prediction': int(prediction),
                    'risk_level': 'High' if prediction == 1 else 'Low',
                    'confidence': float(max(probability)) * 100,
                    'low_risk_prob': float(probability[0]) * 100,
                    'high_risk_prob': float(probability[1]) * 100 if len(probability) > 1 else 0
                }
                
                return render_template('mishandled_prediction.html', result=result,
                                     form_data=request.form)
            else:
                flash('Mishandled prediction model not available', 'error')
                
        except Exception as e:
            logger.error(f"Error in mishandled prediction: {e}")
            flash(f'Error making prediction: {str(e)}', 'error')
    
    return render_template('mishandled_prediction.html')

@app.route('/transfer_prediction', methods=['GET', 'POST'])
def transfer_prediction():
    """Transfer SLA compliance prediction"""
    if request.method == 'POST':
        try:
            # Get form data
            transfer_count = int(request.form['transfer_count'])
            avg_buffer_minutes = float(request.form['avg_buffer_minutes'])
            avg_transfer_time = float(request.form['avg_transfer_time'])
            avg_gate_distance = float(request.form['avg_gate_distance'])
            avg_mct = float(request.form['avg_mct'])
            path_type = request.form['path_type']
            
            # Encode categorical variables
            path_encoded = 0
            
            if 'transfer' in predictor.encoders:
                try:
                    path_encoded = predictor.encoders['transfer']['path'].transform([path_type])[0]
                except:
                    pass
            
            # Prepare features
            features = np.array([[transfer_count, avg_buffer_minutes, avg_transfer_time,
                                avg_gate_distance, avg_mct, path_encoded]])
            
            # Make prediction
            if 'transfer' in predictor.models:
                features_scaled = predictor.scalers['transfer'].transform(features)
                prediction = predictor.models['transfer'].predict(features_scaled)[0]
                probability = predictor.models['transfer'].predict_proba(features_scaled)[0]
                
                result = {
                    'prediction': int(prediction),
                    'risk_level': 'High SLA Risk' if prediction == 1 else 'Low SLA Risk',
                    'confidence': float(max(probability)) * 100,
                    'low_risk_prob': float(probability[0]) * 100,
                    'high_risk_prob': float(probability[1]) * 100 if len(probability) > 1 else 0
                }
                
                return render_template('transfer_prediction.html', result=result,
                                     form_data=request.form)
            else:
                flash('Transfer prediction model not available', 'error')
                
        except Exception as e:
            logger.error(f"Error in transfer prediction: {e}")
            flash(f'Error making prediction: {str(e)}', 'error')
    
    return render_template('transfer_prediction.html')

@app.route('/model_performance')
def model_performance():
    """Display model performance metrics and feature importance"""
    return render_template('model_performance.html', metrics=predictor.metrics)

@app.route('/api/feature_importance/<model_name>')
def get_feature_importance(model_name):
    """API endpoint to get feature importance data for charts"""
    if model_name in predictor.metrics:
        metrics = predictor.metrics[model_name]
        return jsonify({
            'feature_names': metrics.get('feature_names', []),
            'feature_importance': metrics.get('feature_importance', [])
        })
    return jsonify({'error': 'Model not found'}), 404

@app.route('/train_models')
def train_models():
    """Trigger model training (for demo purposes)"""
    try:
        # Import and run training
        from train_models import BaggageMLPipeline
        pipeline = BaggageMLPipeline()
        metrics = pipeline.train_all_models()
        
        # Reload predictor
        global predictor
        predictor = ModelPredictor()
        
        flash('Models trained successfully!', 'success')
        return redirect(url_for('model_performance'))
    except Exception as e:
        logger.error(f"Error training models: {e}")
        flash(f'Error training models: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/baggage_delay_prediction', methods=['GET', 'POST'])
def baggage_delay_prediction_v1():
    """Baggage Delay Prediction V1 (from V1 backend)"""
    import pickle
    import numpy as np
    import pandas as pd
    import os
    from datetime import datetime
    # Load models (cache for performance)
    if not hasattr(baggage_delay_prediction_v1, 'model'):
        with open(os.path.join('baggage_delay_predictionV1', 'model.pkl'), 'rb') as f:
            baggage_delay_prediction_v1.model = pickle.load(f)
        with open(os.path.join('baggage_delay_predictionV1', 'model_reg.pkl'), 'rb') as f:
            baggage_delay_prediction_v1.model_reg = pickle.load(f)
    model = baggage_delay_prediction_v1.model
    model_reg = baggage_delay_prediction_v1.model_reg
    result = None
    reg_result = None
    form_data = None
    error = None
    if request.method == 'POST':
        try:
            # Collect form data
            form_data = request.form.to_dict()
            # Convert and validate fields
            data = {}
            fields = [
                'origin_airport', 'destination_airport',
                'departure_time', 'actual_departure_time',
                'arrival_time', 'check_in_time',
                'weather_condition', 'number_of_bags', 'bag_weight_kg',
                'day_of_week', 'is_international', 'extra_baggage', 'travel_duration_min'
            ]
            for field in fields:
                data[field] = form_data.get(field)
            # Type conversions
            data['number_of_bags'] = int(data['number_of_bags']) if data['number_of_bags'] else 0
            data['bag_weight_kg'] = float(data['bag_weight_kg']) if data['bag_weight_kg'] else 0.0
            data['travel_duration_min'] = int(data['travel_duration_min']) if data['travel_duration_min'] else 0
            # Ensure correct types for categorical fields
            data['is_international'] = int(data['is_international']) if data['is_international'] in ['0', '1', 0, 1] else 0
            data['extra_baggage'] = 'Yes' if data['extra_baggage'] == 'Yes' else 'No'
            # DataFrame for model
            df = pd.DataFrame([data])
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
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0][1]
            log_delay_pred = model_reg.predict(X)[0]
            delay_minutes = float(np.expm1(log_delay_pred))
            result = {
                'prediction': int(pred),
                'probability': float(proba),
                'predicted_delay_minutes': delay_minutes
            }
        except Exception as e:
            error = str(e)
    return render_template('baggage_delay_prediction.html', result=result, form_data=form_data, error=error)
