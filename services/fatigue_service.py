import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from utils.fatigue_utils import preprocess_data, create_enhanced_features, make_json_serializable, calculate_intensity

# Load the model
print("Loading model...")
try:
    with open('models/random_forest_fatigue_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    # Extract model and features
    if isinstance(model_data, dict) and 'model' in model_data and 'model_features' in model_data:
        model = model_data['model']
        model_features = model_data['model_features']
    else:
        # If model data is just the model itself
        model = model_data
        # Define default features
        model_features = ['Training Distance ', 'Session Duration (hrs)', 'Intensity', 
                         'Rest hours', 'Recovery Days', 'avg heart rate', 'RPE(1-10)', 
                         'day_of_week', 'fatigue_lag_1', 'fatigue_lag_2', 'fatigue_lag_3',
                         'distance_7day', 'intensity_7day', 'duration_7day', 
                         'rest_ratio_7day', 'rest_days_7day', 'Stroke Type']
    
    print(f"Model loaded successfully! Model type: {type(model)}")
    print(f"Model features: {model_features}")
except Exception as e:
    print(f"Error loading model: {e}")
    from sklearn.ensemble import RandomForestClassifier
    # Create a deterministic fallback model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model_features = ['Training Distance ', 'Session Duration (hrs)', 'Intensity', 
                     'Rest hours', 'Recovery Days', 'avg heart rate', 'RPE(1-10)', 
                     'day_of_week', 'fatigue_lag_1', 'distance_7day', 'intensity_7day', 
                     'duration_7day', 'rest_ratio_7day', 'rest_days_7day', 'Stroke Type']
    print("Created fallback model due to missing model file")

def process_training_history(history):
    """Process the training history data"""
    # Create DataFrame from historical data
    history_df = pd.DataFrame(history)
    
    # Sort data by date to ensure deterministic processing
    if 'Date' in history_df.columns:
        history_df['Date'] = pd.to_datetime(history_df['Date'])
        history_df = history_df.sort_values('Date')
    
    # Handle column name standardization
    column_mappings = {
        'Training Distance': 'Training Distance ',
        'Session Duration': 'Session Duration (hrs)'
    }
    
    for orig, target in column_mappings.items():
        if orig in history_df.columns and target not in history_df.columns:
            history_df[target] = history_df[orig]
    
    # Ensure all required columns exist
    required_columns = ['Swimmer ID', 'Date', 'Stroke Type', 'Training Distance ', 
                       'Session Duration (hrs)', 'Rest hours', 
                       'Recovery Days', 'avg heart rate', 'RPE(1-10)']
    
    for col in required_columns:
        if col not in history_df.columns:
            # Use appropriate default values
            if col == 'avg heart rate':
                history_df[col] = 150
            elif col == 'RPE(1-10)':
                history_df[col] = 6
            elif col == 'Rest hours':
                history_df[col] = 8
            elif col == 'Recovery Days':
                history_df[col] = 0
            else:
                history_df[col] = 0
    
    # Calculate intensity if not provided
    if 'Intensity' not in history_df.columns:
        history_df['Intensity'] = history_df.apply(calculate_intensity, axis=1)
    
    # Add Fatigue_Numeric if missing with deterministic calculation
    if 'Fatigue_Numeric' not in history_df.columns:
        # Use RPE and calculated intensity to estimate fatigue level
        def estimate_fatigue(row):
            rpe = row.get('RPE(1-10)', 0)
            intensity = row.get('Intensity', 0)
            if rpe >= 8 or intensity >= 9:
                return 3  # High
            elif rpe <= 4 or intensity <= 4:
                return 1  # Low
            else:
                return 2  # Moderate
        
        history_df['Fatigue_Numeric'] = history_df.apply(estimate_fatigue, axis=1)
    
    return history_df

def predict_fatigue_levels(history, days_to_predict=7):
    """Predict fatigue levels for the next few days"""
    
    # Process history data
    history_df = process_training_history(history)
    
    # Get swimmer ID
    swimmer_id = history_df['Swimmer ID'].iloc[0]
    
    # Preprocess data
    processed_df = preprocess_data(history_df)
    
    # Create enhanced features
    enhanced_df = create_enhanced_features(processed_df)
    
    # Get latest date
    latest_date = enhanced_df['Date'].max()
    
    # Determine typical values deterministically
    training_data = enhanced_df[enhanced_df['Training Distance '] > 0]
    if len(training_data) > 0:
        typical_values = {
            'Training Distance ': training_data['Training Distance '].median(),
            'Session Duration (hrs)': training_data['Session Duration (hrs)'].median(),
            'Rest hours': training_data['Rest hours'].median(),
            'Recovery Days': 0,
            'avg heart rate': training_data['avg heart rate'].median(),
            'RPE(1-10)': training_data['RPE(1-10)'].median()
        }
    else:
        typical_values = {
            'Training Distance ': 3000,
            'Session Duration (hrs)': 1.5,
            'Rest hours': 8,
            'Recovery Days': 0,
            'avg heart rate': 150,
            'RPE(1-10)': 6
        }
    
    # Identify rest days pattern deterministically
    rest_day_data = enhanced_df.copy()
    rest_day_data['is_rest'] = (rest_day_data['Training Distance '] == 0)
    
    # Create a deterministic way to find rest days
    rest_days_count = {}
    for day in range(7):
        day_data = rest_day_data[rest_day_data['day_of_week'] == day]
        if len(day_data) > 0:
            rest_count = day_data['is_rest'].sum()
            train_count = len(day_data) - rest_count
            rest_days_count[day] = (rest_count > train_count)
        else:
            rest_days_count[day] = False
    
    # Get typical rest days
    typical_rest_days = [day for day, is_rest in rest_days_count.items() if is_rest]
    
    # Default to Sunday if no rest days found
    if not typical_rest_days:
        typical_rest_days = [6]
    
    # Get primary stroke deterministically
    if 'Stroke Type' in enhanced_df.columns:
        stroke_counts = enhanced_df[enhanced_df['Training Distance '] > 0]['Stroke Type'].value_counts()
        primary_stroke = stroke_counts.index[0] if len(stroke_counts) > 0 else 0
    else:
        primary_stroke = 0  # Default to Freestyle
    
    # Prepare for predictions
    future_dates = [latest_date + timedelta(days=i+1) for i in range(days_to_predict)]
    predictions = []
    
    # Create a working copy for adding predictions
    current_data = enhanced_df.copy()
    
    # Predict for each future day
    for future_date in future_dates:
        # Check if it's a rest day
        is_rest_day = future_date.dayofweek in typical_rest_days
        
        # Every third day should be more intense (deterministic pattern)
        day_index = (future_date - latest_date).days
        is_intense_day = day_index % 3 == 0 and not is_rest_day
        
        # Create prediction row with fixed values
        pred_row = {
            'Swimmer ID': swimmer_id,
            'Date': future_date,
            'Stroke Type': 5 if is_rest_day else primary_stroke,  # 5 for Rest
            'Training Distance ': 0 if is_rest_day else (
                typical_values['Training Distance '] * 1.2 if is_intense_day else typical_values['Training Distance ']
            ),
            'Session Duration (hrs)': 0 if is_rest_day else (
                typical_values['Session Duration (hrs)'] * 1.2 if is_intense_day else typical_values['Session Duration (hrs)']
            ),
            'Rest hours': 24 if is_rest_day else (6 if is_intense_day else typical_values['Rest hours']),
            'Recovery Days': 1 if is_rest_day else 0,
            'avg heart rate': 70 if is_rest_day else (
                typical_values['avg heart rate'] + 20 if is_intense_day else typical_values['avg heart rate']
            ),
            'RPE(1-10)': 1 if is_rest_day else (8 if is_intense_day else typical_values['RPE(1-10)']),
            'day_of_week': future_date.dayofweek,
            'month': future_date.month,
        }
        
        # Calculate intensity rather than setting it directly
        if is_rest_day:
            pred_row['Intensity'] = 1
        else:
            pred_row['Intensity'] = calculate_intensity(pred_row)
            
        # Set initial fatigue based on calculated intensity
        pred_row['Fatigue_Numeric'] = 1 if is_rest_day else (3 if pred_row['Intensity'] >= 8 else 2)
        
        # Add to current data
        temp_df = pd.concat([current_data, pd.DataFrame([pred_row])], ignore_index=True)
        
        # Create enhanced features deterministically
        temp_enhanced = create_enhanced_features(temp_df)
        
        # Get the last row for prediction
        if len(temp_enhanced) > 0:
            X_pred = temp_enhanced.iloc[-1:].copy()
            
            # Ensure all model features exist
            for feature in model_features:
                if feature not in X_pred.columns:
                    X_pred[feature] = 0
            
            # Select only the features needed for the model
            X_pred_model = X_pred[model_features]
            
            # Make prediction with error handling
            try:
                # Predict fatigue level
                fatigue_numeric = model.predict(X_pred_model)[0]
                
                # Map to fatigue category
                fatigue_categories = {1: 'Low', 2: 'Moderate', 3: 'High'}
                fatigue_category = fatigue_categories.get(int(fatigue_numeric), 'Moderate')
                
                # Store prediction
                predictions.append({
                    'date': future_date.strftime('%Y-%m-%d'),
                    'fatigue_level': fatigue_category,
                    'fatigue_numeric': int(fatigue_numeric),
                    'is_rest_day': bool(is_rest_day),
                    'training_distance': round(float(pred_row['Training Distance '])) if not is_rest_day else 0,
                    'calculated_intensity': round(float(pred_row['Intensity']))
                })
                
                # Update for next iteration
                pred_row['Fatigue_Numeric'] = fatigue_numeric
                current_data = pd.concat([current_data, pd.DataFrame([pred_row])], ignore_index=True)
            except Exception as e:
                print(f"Prediction error for {future_date}: {str(e)}")
                
                # Use deterministic fallback
                if is_rest_day:
                    fatigue_numeric = 1  # Low for rest days
                elif pred_row['Intensity'] >= 8:
                    fatigue_numeric = 3  # High for intense days
                else:
                    fatigue_numeric = 2  # Moderate for regular days
                
                fatigue_categories = {1: 'Low', 2: 'Moderate', 3: 'High'}
                fatigue_category = fatigue_categories.get(int(fatigue_numeric), 'Moderate')
                
                predictions.append({
                    'date': future_date.strftime('%Y-%m-%d'),
                    'fatigue_level': fatigue_category,
                    'fatigue_numeric': int(fatigue_numeric),
                    'is_rest_day': bool(is_rest_day),
                    'training_distance': round(float(pred_row['Training Distance '])) if not is_rest_day else 0,
                    'calculated_intensity': round(float(pred_row['Intensity']))
                })
                
                # Update for next iteration
                pred_row['Fatigue_Numeric'] = fatigue_numeric
                current_data = pd.concat([current_data, pd.DataFrame([pred_row])], ignore_index=True)
        else:
            # Handle empty dataframe case
            if is_rest_day:
                fatigue_numeric = 1  # Low for rest days
            elif pred_row['Intensity'] >= 8:
                fatigue_numeric = 3  # High for intense days
            else:
                fatigue_numeric = 2  # Moderate for regular days
            
            fatigue_categories = {1: 'Low', 2: 'Moderate', 3: 'High'}
            fatigue_category = fatigue_categories.get(int(fatigue_numeric), 'Moderate')
            
            predictions.append({
                'date': future_date.strftime('%Y-%m-%d'),
                'fatigue_level': fatigue_category,
                'fatigue_numeric': int(fatigue_numeric),
                'is_rest_day': bool(is_rest_day),
                'training_distance': round(float(pred_row['Training Distance '])) if not is_rest_day else 0,
                'calculated_intensity': round(float(pred_row['Intensity']))
            })
            
            # Update for next iteration
            pred_row['Fatigue_Numeric'] = fatigue_numeric
            current_data = pd.concat([current_data, pd.DataFrame([pred_row])], ignore_index=True)
    
    # Prepare response data
    response_data = {
        'swimmer_id': int(swimmer_id),
        'predictions': predictions,
        'typical_rest_days': [int(d) for d in typical_rest_days]
    }
    
    return response_data