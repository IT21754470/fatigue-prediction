# Save as app.py
from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Helper function to convert NumPy types to Python standard types for JSON serialization
def make_json_serializable(obj):
    """Convert NumPy types to standard Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return make_json_serializable(obj.tolist())
    else:
        return obj

# Function to create enhanced features with longer-term patterns
def create_enhanced_features(swimmer_data, target='Fatigue_Numeric'):
    # Sort by date
    swimmer_data = swimmer_data.sort_values('Date')
    
    # Create lag features (previous fatigue levels)
    for i in range(1, 8):  # Use 7 days of history
        swimmer_data[f'fatigue_lag_{i}'] = swimmer_data[target].shift(i)
    
    # Create rolling window features for training load
    # Short-term (acute) load - 7 days
    swimmer_data['acute_distance'] = swimmer_data['Training Distance'].rolling(window=7).sum().fillna(swimmer_data['Training Distance'])
    swimmer_data['acute_intensity'] = swimmer_data['Intensity'].rolling(window=7).mean().fillna(swimmer_data['Intensity'])
    swimmer_data['acute_duration'] = swimmer_data['Session Duration'].rolling(window=7).sum().fillna(swimmer_data['Session Duration'])
    
    # Calculate fatigue trends
    swimmer_data['fatigue_7day_avg'] = swimmer_data[target].rolling(window=7).mean().fillna(swimmer_data[target])
    
    # Calculate rest metrics
    swimmer_data['rest_ratio_7day'] = swimmer_data['Rest hours'].rolling(window=7).sum().fillna(swimmer_data['Rest hours']) / (24 * 7)
    
    # Calculate training monotony (variation in training)
    swimmer_data['distance_std_7day'] = swimmer_data['Training Distance'].rolling(window=7).std().fillna(0)
    
    # Calculate recent rest days
    swimmer_data['rest_days_7day'] = swimmer_data['Recovery Days'].rolling(window=7).sum().fillna(swimmer_data['Recovery Days'])
    
    return swimmer_data

# Load the saved model from your existing PKL file
print("Loading model...")
try:
    with open('swimmer_fatigue_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
        
    model = model_data['model']
    model_features = model_data['model_features']
    numerical_features = model_data['numerical_features']
    categorical_features = model_data['categorical_features']
    print("Model loaded successfully!")
    print(f"Model features: {model_features}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure 'swimmer_fatigue_model.pkl' exists in the current directory.")

@app.route('/predict', methods=['POST'])
def predict_fatigue():
    try:
        # Get JSON data from request
        data = request.json
        
        # Check if historical data is provided
        if 'history' not in data:
            return jsonify({'error': 'No historical training data provided'}), 400
        
        # Create DataFrame from historical data
        history = data['history']
        history_df = pd.DataFrame(history)
        
        # Ensure Date is datetime
        history_df['Date'] = pd.to_datetime(history_df['Date'])
        
        # Get swimmer ID and latest date
        swimmer_id = history_df['Swimmer ID'].iloc[0]
        latest_date = history_df['Date'].max()
        
        # Standardize column names - handle "Training Distance" vs "Training Distance "
        if 'Training Distance' in history_df.columns and 'Training Distance ' not in history_df.columns:
            history_df['Training Distance '] = history_df['Training Distance']
        
        if 'Session Duration' in history_df.columns and 'Session Duration (hrs)' not in history_df.columns:
            history_df['Session Duration (hrs)'] = history_df['Session Duration']
        
        # Initialize Fatigue_Numeric with default values (moderate fatigue)
        history_df['Fatigue_Numeric'] = 2
        
        # Determine swimmer's primary stroke
        stroke_counts = history_df[history_df['Training Distance'] > 0]['Stroke Type'].value_counts()
        primary_stroke = stroke_counts.index[0] if len(stroke_counts) > 0 else 'Freestyle'
        
        # Get typical training values
        training_data = history_df[history_df['Training Distance'] > 0]
        if len(training_data) > 0:
            typical_values = {
                'Training Distance': training_data['Training Distance'].median(),
                'Training Distance ': training_data['Training Distance'].median(),
                'Session Duration': training_data['Session Duration'].median(),
                'Session Duration (hrs)': training_data['Session Duration'].median(),
                'Intensity': training_data['Intensity'].median(),
                'Rest hours': training_data['Rest hours'].median(),
                'Recovery Days': 0,
                'Energy': 7,  # Default values for possibly missing fields
                'avg heart rate': 150,
                'RPE(1-10)': 6
            }
        else:
            typical_values = {
                'Training Distance': 3000,
                'Training Distance ': 3000,
                'Session Duration': 1.5,
                'Session Duration (hrs)': 1.5,
                'Intensity': 7,
                'Rest hours': 8,
                'Recovery Days': 0,
                'Energy': 7,
                'avg heart rate': 150,
                'RPE(1-10)': 6
            }
        
        # Days to predict
        days_to_predict = data.get('days', 7)  # Default to 7 days
        
        # Identify rest day pattern (default to every 7th day)
        weekday_counts = history_df.groupby([history_df['Date'].dt.dayofweek, 
                                         history_df['Training Distance'] == 0]).size().unstack()
        if not weekday_counts.empty and 1 in weekday_counts.columns:
            typical_rest_days = weekday_counts.index[weekday_counts[1] > weekday_counts.get(0, 0)].tolist()
        else:
            typical_rest_days = [6]  # Default to Sunday
        
        # Prepare for predictions
        future_dates = [latest_date + timedelta(days=i+1) for i in range(days_to_predict)]
        predictions = []
        
        # Create a working copy of history for adding predictions
        current_data = history_df.copy()
        
        # Predict for each future day
        for future_date in future_dates:
            # Check if it's a rest day based on day of week pattern
            is_rest_day = future_date.dayofweek in typical_rest_days
            
            # Create a prediction row based on typical values
            pred_row = {
                'Swimmer ID': swimmer_id,
                'Date': future_date,
                'Stroke Type': 'Rest' if is_rest_day else primary_stroke,
                'Training Distance': 0 if is_rest_day else typical_values['Training Distance'],
                'Training Distance ': 0 if is_rest_day else typical_values['Training Distance '],
                'Session Duration': 0 if is_rest_day else typical_values['Session Duration'],
                'Session Duration (hrs)': 0 if is_rest_day else typical_values['Session Duration (hrs)'],
                'Intensity': 1 if is_rest_day else typical_values['Intensity'],
                'Rest hours': 24 if is_rest_day else typical_values['Rest hours'],
                'Recovery Days': 1 if is_rest_day else 0,
                'Energy': 8 if is_rest_day else typical_values['Energy'],
                'avg heart rate': 70 if is_rest_day else typical_values['avg heart rate'],
                'RPE(1-10)': 1 if is_rest_day else typical_values['RPE(1-10)'],
                'Fatigue_Numeric': 1  # Will be updated with prediction
            }
            
            # Add to current data temporarily
            temp_df = pd.concat([current_data, pd.DataFrame([pred_row])], ignore_index=True)
            
            # Create enhanced features needed for prediction
            enhanced_df = create_enhanced_features(temp_df)
            
            # Get the last row with enhanced features for prediction
            X_pred = enhanced_df.iloc[-1:].copy()
            
            # Ensure all model features exist in X_pred
            for feature in model_features:
                if feature not in X_pred.columns:
                    # Try alternative column names
                    if feature == 'Training Distance ' and 'Training Distance' in X_pred.columns:
                        X_pred[feature] = X_pred['Training Distance']
                    elif feature == 'Session Duration (hrs)' and 'Session Duration' in X_pred.columns:
                        X_pred[feature] = X_pred['Session Duration']
                    else:
                        X_pred[feature] = 0  # Default value
            
            # Select only the features needed for the model
            X_pred = X_pred[model_features]
            
            # Make prediction
            try:
                # Predict fatigue level
                fatigue_numeric = model.predict(X_pred)[0]
                fatigue_categories = {1: 'Low', 2: 'Moderate', 3: 'High'}
                fatigue_category = fatigue_categories.get(int(fatigue_numeric), 'Moderate')
                
                # Store prediction
                predictions.append({
                    'date': future_date.strftime('%Y-%m-%d'),
                    'fatigue_level': fatigue_category
                })
                
                # Update the temp row with predicted fatigue for next iteration
                pred_row['Fatigue_Numeric'] = fatigue_numeric
                
                # Add to current data for next iteration
                current_data = pd.concat([current_data, pd.DataFrame([pred_row])], ignore_index=True)
                
            except Exception as e:
                import traceback
                traceback_str = traceback.format_exc()
                print(f"Prediction error for {future_date}: {str(e)}")
                print(traceback_str)
                
                # Fallback to moderate fatigue
                predictions.append({
                    'date': future_date.strftime('%Y-%m-%d'),
                    'fatigue_level': 'Moderate'
                })
                
                # Update for next iteration with fallback
                pred_row['Fatigue_Numeric'] = 2
                current_data = pd.concat([current_data, pd.DataFrame([pred_row])], ignore_index=True)
        
        # Convert all data to JSON serializable types
        response_data = {
            'swimmer_id': int(swimmer_id),
            'predictions': predictions
        }
        return jsonify(make_json_serializable(response_data))
        
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        return jsonify({'error': f'Server error: {str(e)}', 'traceback': traceback_str}), 500

# Add a simple home route for testing
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'online',
        'message': 'Swimmer fatigue prediction API is running! Send POST requests to /predict',
        'example': {
            'request': '/predict',
            'body': {
                'history': [
                    {
                        'Swimmer ID': 999,
                        'Date': '2023-01-01',
                        'Stroke Type': 'Freestyle',
                        'Training Distance': 3000,
                        'Session Duration': 1.5,
                        'Intensity': 7,
                        'Rest hours': 8,
                        'Recovery Days': 0
                    }
                ],
                'days': 7
            }
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)