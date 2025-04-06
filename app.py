from flask import Flask, request, jsonify
import traceback
from fatigue_service import predict_fatigue_levels
from improvement_services import predict_improvement
from improvement_utils import make_json_serializable

app = Flask(__name__)

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

if __name__ == '__main__':
    app.run(debug=True, port=5000)