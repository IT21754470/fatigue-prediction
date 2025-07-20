from pathlib import Path
import pickle

def load_model():
    try:
        # model file is inside the same directory as this script
        model_path = Path(__file__).resolve().parent / 'swimming_improvement_models.pkl'
        print("Trying to load model from:", model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        with open(model_path, 'rb') as f:
            loaded_models = pickle.load(f)

        return loaded_models
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise



def get_prediction_description(value):
    """
    Convert numerical prediction to description
    Negative values are improvements in the model, but we want to display them
    as positive values to the user
    """
    # Use the negative of value for thresholds since improvements are negative in the model
    if value < -0.2:
        return "significant improvement"
    elif value < -0.05:
        return "slight improvement"
    elif value < 0.05:
        return "no significant change"
    elif value < 0.2:
        return "slight decline"
    else:
        return "significant decline"

def predict_with_models(df, model_package):
    """
    Make predictions using the combined model package
    """
    predictions = {}
    models = model_package['models']
    
    # Make stroke-specific predictions
    if 'Stroke Type' in df.columns:
        for stroke, stroke_data in df.groupby('Stroke Type'):
            stroke_lower = stroke.lower() if isinstance(stroke, str) else stroke
            
            # Try to find the matching model (case-insensitive)
            matching_model = None
            for model_key in models.keys():
                if isinstance(model_key, str) and model_key.lower() == stroke_lower:
                    matching_model = model_key
                    break
            
            if matching_model:
                # Get the model and features
                model_info = models[matching_model]
                model = model_info['model']
                model_features = model_info['features']
                
                # Process the data
                processed_data = create_enhanced_features(stroke_data)
                
                # Ensure all required features are present
                for feature in model_features:
                    if feature not in processed_data.columns:
                        processed_data[feature] = 0
                
                # Create feature matrix
                X = processed_data[model_features].fillna(0)
                
                # Make predictions
                try:
                    pred = model.predict(X)
                    pred_array = pred if isinstance(pred, list) else pred.tolist() if hasattr(pred, 'tolist') else [float(pred)]
                    
                    # Create prediction descriptions using original values
                    descriptions = [get_prediction_description(p) for p in pred_array]
                    
                    # Invert values for display (make improvements positive)
                    inverted_pred_array = [-p for p in pred_array]
                    
                    predictions[stroke] = {
                        'predicted_improvement': inverted_pred_array,  # Inverted values
                        'descriptions': descriptions,  # Descriptions based on original values
                        'dates': processed_data['Date'].dt.strftime('%Y-%m-%d').tolist(),
                        'swimmer_ids': processed_data['Swimmer ID'].tolist(),
                        'accuracy': model_info['accuracy'],
                        'stroke_types': [stroke] * len(pred_array)
                    }
                    print(f"Made {len(pred_array)} predictions for {stroke}")
                except Exception as e:
                    print(f"Error predicting for {stroke}: {str(e)}")
    
    # Make aggregated prediction if no stroke-specific predictions were made
    if not predictions and 'aggregated' in models:
        try:
            # Aggregate the sessions
            agg_data = aggregate_sessions(df)
            
            # Process the aggregated data
            processed_agg_data = create_enhanced_features(agg_data)
            
            # Get model info
            model_info = models['aggregated']
            model = model_info['model']
            model_features = model_info['features']
            
            # Ensure all required features are present
            for feature in model_features:
                if feature not in processed_agg_data.columns:
                    processed_agg_data[feature] = 0
            
            # Create feature matrix
            X = processed_agg_data[model_features].fillna(0)
            
            # Make predictions
            pred = model.predict(X)
            pred_array = pred if isinstance(pred, list) else pred.tolist() if hasattr(pred, 'tolist') else [float(pred)]
            
            # Create prediction descriptions using original values
            descriptions = [get_prediction_description(p) for p in pred_array]
            
            # Invert values for display (make improvements positive)
            inverted_pred_array = [-p for p in pred_array]
            
            # Get the actual stroke types from the original data
            stroke_types = []
            for i, (swimmer_id, date) in enumerate(zip(processed_agg_data['Swimmer ID'], processed_agg_data['Date'])):
                # Find the original stroke type for this session
                matching_rows = df[(df['Swimmer ID'] == swimmer_id) & (df['Date'] == date)]
                if len(matching_rows) > 0:
                    # Use the stroke type from the matching row
                    stroke_type = matching_rows['Stroke Type'].iloc[0]
                elif 'Primary_Stroke' in processed_agg_data.columns:
                    # Use the primary stroke if available
                    stroke_type = processed_agg_data['Primary_Stroke'].iloc[i]
                else:
                    # Default to unknown
                    stroke_type = "Unknown"
                stroke_types.append(stroke_type)
            
            predictions['aggregated'] = {
                'predicted_improvement': inverted_pred_array,  # Inverted values
                'descriptions': descriptions,  # Descriptions based on original values
                'dates': processed_agg_data['Date'].dt.strftime('%Y-%m-%d').tolist(),
                'swimmer_ids': processed_agg_data['Swimmer ID'].tolist(),
                'accuracy': model_info['accuracy'],
                'stroke_types': stroke_types
            }
            print(f"Made {len(pred_array)} aggregated predictions")
        except Exception as e:
            print(f"Error making aggregated prediction: {str(e)}")
    
    return predictions

def get_column_case_insensitive(df, column_name):
    """Get a column by name, ignoring case and trailing spaces"""
    for col in df.columns:
        # Compare lowercased column names without whitespace
        if col.lower().strip() == column_name.lower().strip():
            return col
    return None

def generate_future_predictions(history_df, days_to_predict):
    """
    Generate predictions for future days based on historical data
    """
    # Get unique swimmers
    swimmers = history_df['Swimmer ID'].unique()
    
    # Create empty dataframe for future predictions
    future_df = pd.DataFrame()
    
    # Get column names with proper casing
    training_distance_col = get_column_case_insensitive(history_df, 'Training Distance')
    session_duration_col = get_column_case_insensitive(history_df, 'Session Duration')
    pool_length_col = get_column_case_insensitive(history_df, 'pool length')
    pace_col = get_column_case_insensitive(history_df, 'pace per 100m')
    laps_col = get_column_case_insensitive(history_df, 'laps')
    heart_rate_col = get_column_case_insensitive(history_df, 'avg heart rate')
    
    # Use default column names if not found
    training_distance_col = training_distance_col or 'Training Distance '
    session_duration_col = session_duration_col or 'Session Duration (hrs)'
    pool_length_col = pool_length_col or 'pool length'
    pace_col = pace_col or 'pace per 100m'
    laps_col = laps_col or 'laps'
    heart_rate_col = heart_rate_col or 'avg heart rate'
    
    for swimmer_id in swimmers:
        swimmer_data = history_df[history_df['Swimmer ID'] == swimmer_id]
        
        # Get the latest date for this swimmer
        latest_date = swimmer_data['Date'].max()
        
        # Get the most common stroke types for this swimmer
        common_strokes = swimmer_data['Stroke Type'].value_counts().index.tolist()
        
        # For each day to predict
        for i in range(1, days_to_predict + 1):
            future_date = latest_date + timedelta(days=i)
            
            # For each common stroke, create a future training entry
            for stroke in common_strokes[:min(2, len(common_strokes))]:  # Use top 2 most common strokes
                # Get average values for this swimmer and stroke
                stroke_data = swimmer_data[swimmer_data['Stroke Type'] == stroke]
                
                if len(stroke_data) > 0:
                    # Create a future training session based on averages
                    future_row = {
                        'Swimmer ID': swimmer_id,
                        'Date': future_date,
                        'Stroke Type': stroke,
                        pool_length_col: stroke_data[pool_length_col].mean() if pool_length_col in stroke_data.columns else 50,
                        training_distance_col: stroke_data[training_distance_col].mean() if training_distance_col in stroke_data.columns else 3000,
                        session_duration_col: stroke_data[session_duration_col].mean() if session_duration_col in stroke_data.columns else 1.5,
                        pace_col: stroke_data[pace_col].mean() if pace_col in stroke_data.columns else 70,
                        laps_col: stroke_data[laps_col].mean() if laps_col in stroke_data.columns else 60,
                        heart_rate_col: stroke_data[heart_rate_col].mean() if heart_rate_col in stroke_data.columns else 150,
                        'predicted improvement (s)': 0  # This will be predicted
                    }
                    
                    future_df = pd.concat([future_df, pd.DataFrame([future_row])], ignore_index=True)
    
    return future_df

def predict_improvement(history_json, days_to_predict=7):
    """
    Main function to predict swimming improvement
    
    Args:
        history_json: List of training session dictionaries
        days_to_predict: Number of days to predict into the future
        
    Returns:
        Dictionary with predictions organized by date
    """
    try:
        # Load model
        model_package = load_model()
        
        # Convert JSON to DataFrame
        history_df = convert_json_to_dataframe(history_json)
        
        # Generate predictions for historical data
        historical_predictions = predict_with_models(history_df, model_package)
        
        # Generate future training data
        future_df = generate_future_predictions(history_df, days_to_predict)
        
        # Generate predictions for future data
        future_predictions = predict_with_models(future_df, model_package)
        
        # Reorganize predictions by date
        historical_by_date = {}
        
        # Process historical predictions
        for stroke, pred_data in historical_predictions.items():
            for i in range(len(pred_data['dates'])):
                date = pred_data['dates'][i]
                improvement = pred_data['predicted_improvement'][i]  # Already inverted
                description = pred_data['descriptions'][i]
                swimmer_id = pred_data['swimmer_ids'][i]
                stroke_type = pred_data['stroke_types'][i] if 'stroke_types' in pred_data else stroke
                
                if date not in historical_by_date:
                    historical_by_date[date] = []
                
                historical_by_date[date].append({
                    'swimmer_id': swimmer_id,
                    'stroke': stroke_type,
                    'improvement': improvement,  # Positive value
                    'description': description
                })
        
        # Reorganize future predictions by date
        future_by_date = {}
        
        # Process future predictions
        for stroke, pred_data in future_predictions.items():
            for i in range(len(pred_data['dates'])):
                date = pred_data['dates'][i]
                improvement = pred_data['predicted_improvement'][i]  # Already inverted
                description = pred_data['descriptions'][i]
                swimmer_id = pred_data['swimmer_ids'][i]
                stroke_type = pred_data['stroke_types'][i] if 'stroke_types' in pred_data else stroke
                
                if date not in future_by_date:
                    future_by_date[date] = []
                
                future_by_date[date].append({
                    'swimmer_id': swimmer_id,
                    'stroke': stroke_type,
                    'improvement': improvement,  # Positive value
                    'description': description
                })
        
        # Prepare response
        response = {
            'historical_predictions': {
                'by_date': historical_by_date,
                'model_accuracy': next(iter(historical_predictions.values()))['accuracy'] if historical_predictions else None
            },
            'future_predictions': {
                'by_date': future_by_date,
                'model_accuracy': next(iter(future_predictions.values()))['accuracy'] if future_predictions else None
            },
            'model_info': {
                'version': model_package.get('version', 'unknown'),
                'created_date': model_package.get('created_date', 'unknown'),
                'available_models': model_package.get('stroke_types', [])
            }
        }
        
        # Add swimmer-specific summaries
        swimmer_summaries = {}
        for swimmer_id in history_df['Swimmer ID'].unique():
            # Get all predictions for this swimmer
            all_predictions = []
            
            # Check historical predictions organized by date
            for date, predictions in historical_by_date.items():
                for pred in predictions:
                    if pred['swimmer_id'] == swimmer_id:
                        all_predictions.append(pred['improvement'])
            
            # Check future predictions organized by date
            for date, predictions in future_by_date.items():
                for pred in predictions:
                    if pred['swimmer_id'] == swimmer_id:
                        all_predictions.append(pred['improvement'])
            
            # Calculate average improvement
            if all_predictions:
                avg_improvement = sum(all_predictions) / len(all_predictions)
                
                # Generate summary - now using positive values for improvements
                if avg_improvement > 0.1:
                    trend = "consistently improving"
                elif avg_improvement < -0.1:
                    trend = "showing performance decline"
                else:
                    trend = "maintaining stable performance"
                
                swimmer_summaries[str(swimmer_id)] = {
                    'average_improvement': avg_improvement,
                    'trend': trend,
                    'prediction_count': len(all_predictions)
                }
        
        response['swimmer_summaries'] = swimmer_summaries
        
        return response
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return {
            'error': str(e),
            'details': error_details
        }