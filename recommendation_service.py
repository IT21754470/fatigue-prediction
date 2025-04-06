import pickle
import pandas as pd
import os
import numpy as np
from datetime import datetime

# Define path to model files
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

def load_model_components():
    """Load all necessary model components"""
    try:
        with open(os.path.join(MODEL_PATH, 'recommendation_model.pkl'), 'rb') as f:
            model = pickle.load(f)
        with open(os.path.join(MODEL_PATH, 'encoders.pkl'), 'rb') as f:
            encoders = pickle.load(f)
        with open(os.path.join(MODEL_PATH, 'imputer.pkl'), 'rb') as f:
            imputer = pickle.load(f)
        with open(os.path.join(MODEL_PATH, 'target_column.pkl'), 'rb') as f:
            target_column = pickle.load(f)
        
        return model, encoders, imputer, target_column
    except Exception as e:
        raise Exception(f"Failed to load model components: {str(e)}")

def predict_recommendation(model, encoders, imputer, improvement, fatigue_level, stroke_type):
    """Generate a recommendation based on input parameters"""
    # Create feature vector
    features = pd.DataFrame({
        'predicted improvement (s)': [improvement],
        'Predicted Fatigue Level': [fatigue_level],
        'Stroke Type': [stroke_type],
        'is_positive_improvement': [improvement > 0],
        'improvement_magnitude': [abs(improvement)]
    })
    
    # Handle missing values
    features[['predicted improvement (s)', 'improvement_magnitude']] = imputer.transform(
        features[['predicted improvement (s)', 'improvement_magnitude']])
    
    # Encode categorical features
    for feature, encoder in encoders.items():
        if feature in features.columns:
            try:
                features[feature] = encoder.transform(features[feature])
            except:
                print(f"Warning: Value '{features[feature][0]}' for {feature} not seen during training.")
                # Use a default value (most common category)
                if feature == 'Stroke Type':
                    features[feature] = 5  # 'free' based on your encoding
                elif feature == 'Predicted Fatigue Level':
                    features[feature] = 2  # 'Moderate' based on your encoding
                else:
                    features[feature] = 0
    
    # Predict category (the target is 'rec_theme' based on your output)
    category = model.predict(features)[0]
    
    # Get model confidence (probability)
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features)[0]
        max_prob_index = probabilities.argmax()
        confidence = probabilities[max_prob_index]
    else:
        confidence = None
    
    # Define some example recommendations for each category based on your output
    recommendation_examples = {
        'technique_focus': "Focus on technique rather than volume. Add specific drills targeting your weakest phase of the stroke.",
        'intensity_focus': "Incorporate 4×200m technique-focused sets with paddles. Emphasize distance per stroke rather than speed.",
        'recovery_focus': "Ensure 48-hour recovery periods between intensive drill sessions to allow for proper motor pattern development.",
        'maintain_current': "Maintain current training approach with balanced workload across all strokes.",
        'other': "Balance drill work between technique refinement and endurance development with equal focus."
    }
    
    # Get recommendation based on category
    recommendation = recommendation_examples.get(category, f"Focus on {category.replace('_', ' ')}.")
    
    return recommendation, category, confidence

def generate_recommendations(swimmer_id, improvement, fatigue_level, stroke_type="free", date=None):
    """Generate recommendation for a single swimmer"""
    try:
        # Load model components
        model, encoders, imputer, target_column = load_model_components()
        
        # Generate recommendation
        recommendation, category, confidence = predict_recommendation(
            model, encoders, imputer, improvement, fatigue_level, stroke_type
        )
        
        # Create response
        response = {
            'swimmer_id': swimmer_id,
            'date': date or datetime.now().strftime("%Y-%m-%d"),
            'stroke_type': stroke_type,
            'improvement': improvement,
            'fatigue_level': fatigue_level,
            'category': category,
            'recommendation': recommendation
        }
        
        if confidence is not None:
            response['confidence'] = float(confidence)
            
        return response
        
    except Exception as e:
        print(f"Error generating recommendation: {str(e)}")
        raise

def generate_batch_recommendations(swimmers_data):
    """Generate recommendations for multiple swimmers"""
    results = []
    
    try:
        # Load model components once to improve efficiency
        model, encoders, imputer, target_column = load_model_components()
        
        for swimmer in swimmers_data:
            try:
                swimmer_id = swimmer.get('swimmer_id')
                improvement = swimmer.get('improvement')
                fatigue_level = swimmer.get('fatigue_level')
                stroke_type = swimmer.get('stroke_type', 'free')
                date = swimmer.get('date')
                
                recommendation, category, confidence = predict_recommendation(
                    model, encoders, imputer, improvement, fatigue_level, stroke_type
                )
                
                result = {
                    'swimmer_id': swimmer_id,
                    'date': date or datetime.now().strftime("%Y-%m-%d"),
                    'stroke_type': stroke_type,
                    'improvement': improvement,
                    'fatigue_level': fatigue_level,
                    'category': category,
                    'recommendation': recommendation
                }
                
                if confidence is not None:
                    result['confidence'] = float(confidence)
                    
                results.append(result)
                
            except Exception as e:
                # Log error but continue processing other swimmers
                print(f"Error processing swimmer {swimmer_id}: {str(e)}")
                results.append({
                    'swimmer_id': swimmer_id,
                    'error': str(e)
                })
    except Exception as e:
        raise Exception(f"Failed to process batch recommendations: {str(e)}")
    
    return results