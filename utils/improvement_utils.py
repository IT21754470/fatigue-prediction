import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

def make_json_serializable(obj):
    """
    Convert objects to JSON serializable formats
    """
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, datetime):
        return obj.strftime('%Y-%m-%d')
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    return obj

def convert_json_to_dataframe(json_data):
    """
    Convert JSON training data to pandas DataFrame
    """
    df = pd.DataFrame(json_data)
    
    # Convert date strings to datetime objects
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Ensure numeric columns are numeric
    numeric_cols = ['Swimmer ID', 'pool length', 'Training Distance ', 
                   'Session Duration (hrs)', 'pace per 100m', 'laps', 
                   'avg heart rate', 'predicted improvement (s)']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def aggregate_sessions(df):
    """
    Aggregate training sessions by date and swimmer
    """
    # Create session ID
    df['Session_ID'] = df['Swimmer ID'].astype(str) + '_' + df['Date'].dt.strftime('%Y-%m-%d')

    # Group by session ID
    session_data = df.groupby('Session_ID').agg({
        'Swimmer ID': 'first',
        'Date': 'first',
        'pool length': 'first',
        'Training Distance ': 'sum',
        'Session Duration (hrs)': 'mean',
        'pace per 100m': 'mean',
        'laps': 'sum',
        'avg heart rate': 'mean',
        'predicted improvement (s)': 'mean',
    }).reset_index()

    # Find primary stroke
    stroke_distances = df.groupby(['Session_ID', 'Stroke Type'])['Training Distance '].sum().reset_index()
    primary_strokes = stroke_distances.loc[stroke_distances.groupby('Session_ID')['Training Distance '].idxmax()]
    primary_strokes = primary_strokes[['Session_ID', 'Stroke Type']].rename(columns={'Stroke Type': 'Primary_Stroke'})
    session_data = session_data.merge(primary_strokes, on='Session_ID')

    # Add stroke diversity features
    stroke_counts = df.groupby('Session_ID')['Stroke Type'].nunique().reset_index()
    stroke_counts = stroke_counts.rename(columns={'Stroke Type': 'Stroke_Count'})
    session_data = session_data.merge(stroke_counts, on='Session_ID')

    # Add individual stroke distances as features
    stroke_types = df['Stroke Type'].unique()
    for stroke in stroke_types:
        stroke_data = df[df['Stroke Type'] == stroke].groupby('Session_ID')['Training Distance '].sum().reset_index()
        stroke_data = stroke_data.rename(columns={'Training Distance ': f'{stroke}_Distance'})
        session_data = session_data.merge(stroke_data, on='Session_ID', how='left')
        session_data[f'{stroke}_Distance'] = session_data[f'{stroke}_Distance'].fillna(0)
        # Add percentage features
        if session_data['Training Distance '].sum() > 0:
            session_data[f'{stroke}_Percentage'] = session_data[f'{stroke}_Distance'] / session_data['Training Distance '] * 100
        else:
            session_data[f'{stroke}_Percentage'] = 0

    # Add intensity feature
    if 'Intensity' not in session_data.columns:
        session_data['Intensity'] = session_data['avg heart rate'] / session_data['pace per 100m']
        session_data['Intensity'] = session_data['Intensity'].replace([np.inf, -np.inf], np.nan).fillna(0)

    return session_data

def create_enhanced_features(swimmer_data, target='predicted improvement (s)'):
    """
    Create enhanced features for prediction
    """
    # Sort by date
    swimmer_data = swimmer_data.sort_values('Date')
    
    # Create copy to avoid modifying original data
    enhanced_data = swimmer_data.copy()
    
    # Create lag features
    if target in enhanced_data.columns:
        for i in range(1, 4):
            if len(enhanced_data) > i:
                enhanced_data[f'improvement_lag_{i}'] = enhanced_data[target].shift(i)

    # Create rolling window features
    if 'Training Distance ' in enhanced_data.columns:
        enhanced_data['acute_distance'] = enhanced_data['Training Distance '].rolling(window=3, min_periods=1).sum()
    
    if 'Intensity' in enhanced_data.columns:
        enhanced_data['acute_intensity'] = enhanced_data['Intensity'].rolling(window=3, min_periods=1).mean()
    
    if 'Session Duration (hrs)' in enhanced_data.columns:
        enhanced_data['acute_duration'] = enhanced_data['Session Duration (hrs)'].rolling(window=3, min_periods=1).sum()

    # Calculate training volume trends
    if 'Training Distance ' in enhanced_data.columns:
        enhanced_data['distance_3day_avg'] = enhanced_data['Training Distance '].rolling(window=3, min_periods=1).mean()

    # Calculate training intensity trends
    if 'pace per 100m' in enhanced_data.columns:
        enhanced_data['pace_3day_avg'] = enhanced_data['pace per 100m'].rolling(window=3, min_periods=1).mean()
    
    if 'avg heart rate' in enhanced_data.columns:
        enhanced_data['hr_3day_avg'] = enhanced_data['avg heart rate'].rolling(window=3, min_periods=1).mean()

    # Add stroke diversity trends
    if 'Stroke_Count' in enhanced_data.columns:
        enhanced_data['stroke_diversity_3day'] = enhanced_data['Stroke_Count'].rolling(window=3, min_periods=1).mean()

    # Add improvement trend features
    if target in enhanced_data.columns:
        enhanced_data['improvement_3day_avg'] = enhanced_data[target].rolling(window=3, min_periods=1).mean()

    # Calculate training load ratios
    if 'avg heart rate' in enhanced_data.columns and 'pace per 100m' in enhanced_data.columns:
        enhanced_data['hr_to_pace_ratio'] = enhanced_data['avg heart rate'] / enhanced_data['pace per 100m']
        enhanced_data['hr_to_pace_ratio'] = enhanced_data['hr_to_pace_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Fill missing values
    numeric_cols = enhanced_data.select_dtypes(include=['int64', 'float64']).columns
    enhanced_data[numeric_cols] = enhanced_data[numeric_cols].fillna(0)
    
    return enhanced_data

def generate_explanations_for_prediction(sample_data, prediction_value, feature_names):
    """
    Generate human-readable explanations for a single prediction
    Works with any model type - no SHAP needed
    """
    reasons = []
    top_factors = []
    
    # Training intensity analysis
    if 'Intensity' in sample_data and sample_data['Intensity'] > 0:
        intensity = sample_data['Intensity']
        if intensity > 7.5:
            reasons.append(f"Very high training intensity ({intensity:.1f}/10)")
            top_factors.append('Intensity')
        elif intensity > 5.5:
            reasons.append(f"High training intensity ({intensity:.1f}/10)")
            top_factors.append('Intensity')
        elif intensity < 3.0:
            reasons.append(f"Low training intensity ({intensity:.1f}/10)")
            top_factors.append('Intensity')
    
    # Recent improvement trends
    if 'improvement_lag_1' in sample_data and sample_data['improvement_lag_1'] != 0:
        prev_improvement = sample_data['improvement_lag_1']
        if prev_improvement > 0.5:  # Remember: positive = improvement in your inverted system
            reasons.append("Recent positive performance trend")
            top_factors.append('improvement_lag_1')
        elif prev_improvement < -0.5:
            reasons.append("Recent performance decline pattern")
            top_factors.append('improvement_lag_1')
    
    # Training volume
    if 'acute_distance' in sample_data and sample_data['acute_distance'] > 0:
        distance = sample_data['acute_distance']
        if distance > 8000:
            reasons.append(f"High weekly training volume ({distance:.0f}m)")
            top_factors.append('acute_distance')
        elif distance > 5000:
            reasons.append(f"Moderate training volume ({distance:.0f}m)")
            top_factors.append('acute_distance')
        elif distance < 2000:
            reasons.append(f"Low training volume ({distance:.0f}m)")
            top_factors.append('acute_distance')
    
    # Pace trends
    if 'pace_3day_avg' in sample_data and sample_data['pace_3day_avg'] > 0:
        pace = sample_data['pace_3day_avg']
        if pace < 70:
            reasons.append(f"Fast average pace ({pace:.1f}s/100m)")
            top_factors.append('pace_3day_avg')
        elif pace > 120:
            reasons.append(f"Slow average pace ({pace:.1f}s/100m)")
            top_factors.append('pace_3day_avg')
    
    # Heart rate analysis
    if 'hr_3day_avg' in sample_data and sample_data['hr_3day_avg'] > 0:
        hr = sample_data['hr_3day_avg']
        if hr > 165:
            reasons.append(f"Very high heart rate ({hr:.0f} bpm)")
            top_factors.append('hr_3day_avg')
        elif hr > 150:
            reasons.append(f"Elevated heart rate ({hr:.0f} bpm)")
            top_factors.append('hr_3day_avg')
        elif hr < 110:
            reasons.append(f"Low heart rate ({hr:.0f} bpm)")
            top_factors.append('hr_3day_avg')
    
    # Rest interval
    if 'rest interval' in sample_data and sample_data['rest interval'] > 0:
        rest = sample_data['rest interval']
        if rest < 30:
            reasons.append(f"Short rest intervals ({rest:.0f}s)")
            top_factors.append('rest interval')
        elif rest > 120:
            reasons.append(f"Long rest intervals ({rest:.0f}s)")
            top_factors.append('rest interval')
    
    # Session duration
    if 'Session Duration (hrs)' in sample_data and sample_data['Session Duration (hrs)'] > 0:
        duration = sample_data['Session Duration (hrs)']
        if duration > 2.5:
            reasons.append(f"Long training session ({duration:.1f} hours)")
            top_factors.append('Session Duration (hrs)')
        elif duration < 0.5:
            reasons.append(f"Short training session ({duration:.1f} hours)")
            top_factors.append('Session Duration (hrs)')
    
    # If not enough specific reasons, add general ones based on prediction
    if len(reasons) < 2:
        if prediction_value > 0.1:  # Improvement (positive in your inverted system)
            reasons.append("Overall training pattern shows improvement")
        elif prediction_value < -0.1:  # Decline
            reasons.append("Training load may need adjustment")
        else:
            reasons.append("Stable performance expected")
    
    # Return top 3 reasons
    return {
        'reasons': reasons[:3],
        'top_factors': top_factors[:3]
    }