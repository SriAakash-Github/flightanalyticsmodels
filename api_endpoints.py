from flask import jsonify, request
import json
from app import app, db, PredictionHistory

@app.route('/api/prediction_details/<int:record_id>')
def get_prediction_details(record_id):
    """Get detailed prediction information"""
    try:
        record = PredictionHistory.query.get_or_404(record_id)
        
        return jsonify({
            'id': record.id,
            'model_type': record.model_type,
            'flight_number': record.flight_number,
            'input_data': json.loads(record.input_data),
            'prediction_result': json.loads(record.prediction_result),
            'timestamp': record.timestamp.isoformat(),
            'batch_id': record.batch_id
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_stats')
def get_model_stats():
    """Get model statistics"""
    try:
        stats = {}
        
        # Get prediction counts by model
        model_counts = db.session.query(
            PredictionHistory.model_type,
            db.func.count(PredictionHistory.id)
        ).group_by(PredictionHistory.model_type).all()
        
        for model_type, count in model_counts:
            stats[model_type] = {
                'total_predictions': count,
                'recent_predictions': PredictionHistory.query.filter_by(
                    model_type=model_type
                ).order_by(PredictionHistory.timestamp.desc()).limit(5).count()
            }
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch_status/<batch_id>')
def get_batch_status(batch_id):
    """Get batch prediction status"""
    try:
        batch_predictions = PredictionHistory.query.filter_by(batch_id=batch_id).all()
        
        if not batch_predictions:
            return jsonify({'error': 'Batch not found'}), 404
        
        return jsonify({
            'batch_id': batch_id,
            'total_predictions': len(batch_predictions),
            'model_type': batch_predictions[0].model_type,
            'created_at': batch_predictions[0].timestamp.isoformat(),
            'predictions': [
                {
                    'id': p.id,
                    'flight_number': p.flight_number,
                    'result': json.loads(p.prediction_result)
                } for p in batch_predictions
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export_predictions')
def export_predictions():
    """Export predictions in various formats"""
    format_type = request.args.get('format', 'json')
    model_type = request.args.get('model_type')
    limit = int(request.args.get('limit', 100))
    
    try:
        query = PredictionHistory.query
        if model_type:
            query = query.filter_by(model_type=model_type)
        
        predictions = query.order_by(PredictionHistory.timestamp.desc()).limit(limit).all()
        
        data = []
        for p in predictions:
            data.append({
                'id': p.id,
                'model_type': p.model_type,
                'flight_number': p.flight_number,
                'timestamp': p.timestamp.isoformat(),
                'input_data': json.loads(p.input_data),
                'prediction_result': json.loads(p.prediction_result),
                'batch_id': p.batch_id
            })
        
        if format_type == 'json':
            return jsonify(data)
        else:
            return jsonify({'error': 'Unsupported format'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500