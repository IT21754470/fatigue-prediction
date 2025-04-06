import numpy as np
import pandas as pd
import json
from datetime import datetime, date

def make_json_serializable(obj):
    """Convert objects to JSON serializable format"""
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    else:
        return obj

def validate_recommendation_input(data):
    """Validate input data for recommendation generation"""
    required_fields = ['improvement', 'fatigue_level']
    errors = []
    
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    # Validate improvement is numeric
    if 'improvement' in data:
        try:
            float(data['improvement'])
        except:
            errors.append("Improvement must be a number")
    
    # Validate fatigue level is one of expected values
    if 'fatigue_level' in data:
        valid_fatigue_levels = ['Low', 'Moderate', 'High', 'Unknown']
        if data['fatigue_level'] not in valid_fatigue_levels:
            errors.append(f"Fatigue level must be one of: {', '.join(valid_fatigue_levels)}")
    
    # Validate stroke type if provided
    if 'stroke_type' in data:
        valid_stroke_types = ['free', 'back', 'breast', 'fly', 'im', 'Unknown']
        if data['stroke_type'] not in valid_stroke_types:
            errors.append(f"Stroke type must be one of: {', '.join(valid_stroke_types)}")
    
    return errors

def format_recommendation(recommendation, category, additional_info=None):
    """Format recommendation for output"""
    result = {
        'recommendation': recommendation,
        'category': category
    }
    
    if additional_info:
        result.update(additional_info)
        
    return result