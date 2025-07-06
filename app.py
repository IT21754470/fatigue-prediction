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

# HTML template for testing interface
TEST_INTERFACE_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Swimming Recommendation API - Test Interface</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .section { margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
        .endpoint { background-color: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 3px; }
        .method-get { color: #28a745; }
        .method-post { color: #007bff; }
        button { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 3px; cursor: pointer; margin: 5px; }
        button:hover { background-color: #0056b3; }
        textarea { width: 100%; height: 200px; margin: 10px 0; }
        .result { background-color: #f8f9fa; padding: 15px; border-radius: 3px; margin-top: 10px; }
        .error { background-color: #f8d7da; color: #721c24; }
        .success { background-color: #d4edda; color: #155724; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Swimming Recommendation API - Test Interface</h1>
        
        <div class="section">
            <h2>Available Endpoints</h2>
            <div class="endpoint">
                <span class="method-get"><strong>GET</strong></span> /health - Health check
            </div>
            <div class="endpoint">
                <span class="method-post"><strong>POST</strong></span> /recommendation/generate - Single recommendation
            </div>
            <div class="endpoint">
                <span class="method-post"><strong>POST</strong></span> /recommendation/generate/batch - Batch recommendations
            </div>
        </div>

        <div class="section">
            <h2>Quick Tests</h2>
            <button onclick="testHealth()">Test Health Check</button>
            <button onclick="testSingle()">Test Single Recommendation</button>
            <button onclick="testBatch()">Test Batch Recommendations</button>
        </div>

        <div class="section">
            <h2>Results</h2>
            <div id="results"></div>
        </div>
    </div>

    <script>
        function displayResult(title, data, isError = false) {
            const resultsDiv = document.getElementById('results');
            const resultClass = isError ? 'result error' : 'result success';
            const html = `
                <div class="${resultClass}">
                    <h3>${title}</h3>
                    <pre>${JSON.stringify(data, null, 2)}</pre>
                </div>
            `;
            resultsDiv.innerHTML = html + resultsDiv.innerHTML;
        }

        async function testHealth() {
            try {
                const response = await fetch('/health');
                const data = await response.json();
                displayResult('Health Check', data, !response.ok);
            } catch (error) {
                displayResult('Health Check Error', {error: error.message}, true);
            }
        }

        async function testSingle() {
            try {
                const response = await fetch('/test/single');
                const data = await response.json();
                displayResult('Single Recommendation Test', data, !response.ok);
            } catch (error) {
                displayResult('Single Recommendation Error', {error: error.message}, true);
            }
        }

        async function testBatch() {
            try {
                const response = await fetch('/test/batch');
                const data = await response.json();
                displayResult('Batch Recommendations Test', data, !response.ok);
            } catch (error) {
                displayResult('Batch Recommendations Error', {error: error.message}, true);
            }
        }
    </script>
</body>
</html>
'''
# Add this new endpoint to your existing app.py

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
    print("🏊‍♀ Swimming Recommendation API Starting...")
    print("📝 Available endpoints:")
    print("  POST /predict - Get swimming recommendations")
    print("  GET /health - Check API health")
    print("  GET /codes - Get available grshort codes")
    app.run(debug=True, host='0.0.0.0', port=5000)