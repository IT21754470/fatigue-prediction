from flask import Flask, request, jsonify, render_template_string
import traceback
import json
from datetime import datetime
from fatigue_service import predict_fatigue_levels
from improvement_services import predict_improvement
from improvement_utils import make_json_serializable
from recommendation_service import recommendation_service
from recommendation_utils import format_prediction_response


app = Flask(__name__)



@app.route('/dataset/info', methods=['GET'])
def get_dataset_info():
    """Get information about the loaded dataset"""
    try:
        from recommendation_service import get_dataset_information
        dataset_info = get_dataset_information()
        return jsonify(make_json_serializable(dataset_info))
    except Exception as e:
        return jsonify({'error': f'Error retrieving dataset info: {str(e)}'}), 500
    
@app.route('/fatigue/predict', methods=['POST'])
def predict_fatigue():
    try:
        # Get JSON data from request
        data = request.json
        
        # Check if historical data is provided
        if 'history' not in data:
            return jsonify({'error': 'No historical training data provided'}), 400
        
        # Get history and days to predict
        history = data['history']
        days_to_predict = data.get('days', 7)
        
        # Call the prediction service
        response_data = predict_fatigue_levels(history, days_to_predict)
        
        # Return results
        return jsonify(make_json_serializable(response_data))
    
    except Exception as e:
        traceback_str = traceback.format_exc()
        return jsonify({'error': f'Server error: {str(e)}', 'traceback': traceback_str}), 500

@app.route('/improvement/predict', methods=['POST'])
def predict_improvement_endpoint():
    try:
        # Get JSON data from request
        data = request.json
        
        # Check if historical data is provided
        if 'history' not in data:
            return jsonify({'error': 'No historical training data provided'}), 400
        
        # Get history and days to predict
        history = data['history']
        days_to_predict = data.get('days', 7)
        
        # Call the improvement prediction service
        response_data = predict_improvement(history, days_to_predict)
        
        # Return results
        return jsonify(make_json_serializable(response_data))
    
    except Exception as e:
        traceback_str = traceback.format_exc()
        return jsonify({'error': f'Server error: {str(e)}', 'traceback': traceback_str}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for swimming recommendations"""
    try:
        # Get data from request
        data = request.json
        
        # Extract swimmer data
        swimmer_id = data.get('swimmer_id', 'Unknown')
        stroke_type = data.get('stroke_type', 'free')
        improvement = float(data.get('predicted_improvement', 0.0))
        fatigue_level = data.get('fatigue_level', 'Moderate')
        
        # Make prediction
        grshort_code, confidence, top_predictions = recommendation_service.predict_grshort(
            improvement, fatigue_level, stroke_type
        )
        
        # Prepare response
        response = format_prediction_response(
            swimmer_id, stroke_type, improvement, fatigue_level,
            grshort_code, confidence, top_predictions
        )
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "model_loaded": recommendation_service.is_model_loaded()
    })

@app.route('/codes', methods=['GET'])
def get_codes():
    """Get all available grshort codes"""
    codes = recommendation_service.get_available_codes()
    return jsonify({"available_codes": codes})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    print("🏊‍♀ Swimming Recommendation API Starting...")
    print("📝 Available endpoints:")
    print("  POST /predict - Get swimming recommendations")
    print("  GET /health - Check API health")
    print("  GET /codes - Get available grshort codes")
    print("  POST /fatigue/predict - Predict fatigue levels")
    print("  POST /improvement/predict - Predict improvement")
    print("  GET /dataset/info - Get dataset information")
    app.run(debug=False, host='0.0.0.0', port=port)