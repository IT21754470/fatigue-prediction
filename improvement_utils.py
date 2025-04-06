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