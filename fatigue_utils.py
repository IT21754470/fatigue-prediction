import numpy as np
import pandas as pd
import random

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Helper function for JSON serialization
def make_json_serializable(obj):
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

# Calculate intensity based on other metrics
def calculate_intensity(row):
    """
    Calculate workout intensity based on other available metrics
    Returns a value between 1-10
    """
    # Base factors that influence intensity
    factors = []
    weights = []
    
    # Use RPE if available (strong indicator of intensity)
    if 'RPE(1-10)' in row and not pd.isna(row['RPE(1-10)']):
        factors.append(row['RPE(1-10)'])
        weights.append(2.0)  # Give RPE a strong weight
    
    # Use heart rate if available
    if 'avg heart rate' in row and not pd.isna(row['avg heart rate']):
        # Convert heart rate to intensity scale (assuming 60-200 range maps to 1-10)
        hr_min, hr_max = 60, 200
        hr = max(min(row['avg heart rate'], hr_max), hr_min)  # Clamp to range
        hr_intensity = 1 + 9 * (hr - hr_min) / (hr_max - hr_min)
        factors.append(hr_intensity)
        weights.append(1.5)
    
    # Consider training distance relative to typical distances
    if 'Training Distance ' in row and not pd.isna(row['Training Distance ']):
        # Assuming typical distances range from 0 to 8000m
        dist_max = 8000
        dist_intensity = min(10 * row['Training Distance '] / dist_max, 10)
        factors.append(dist_intensity)
        weights.append(1.0)
    
    # Consider session duration
    if 'Session Duration (hrs)' in row and not pd.isna(row['Session Duration (hrs)']):
        # Longer sessions might indicate lower intensity, shorter might be higher
        # Inverse relationship: 0.5hr -> intensity factor of 8, 2hr -> intensity factor of 2
        duration = row['Session Duration (hrs)']
        if duration > 0:
            duration_intensity = max(10 - 4 * duration, 1)
            factors.append(duration_intensity)
            weights.append(0.8)
    
    # Consider rest hours (less rest might indicate higher intensity training)
    if 'Rest hours' in row and not pd.isna(row['Rest hours']):
        rest_intensity = max(10 - row['Rest hours'] / 2.4, 1)  # 24hrs rest -> intensity 1
        factors.append(rest_intensity)
        weights.append(0.5)
    
    # If we have no factors, return a default intensity
    if not factors:
        return 5  # Middle intensity as default
    
    # Calculate weighted average
    weighted_sum = sum(f * w for f, w in zip(factors, weights))
    total_weight = sum(weights)
    
    # Return rounded intensity between 1-10
    intensity = round(weighted_sum / total_weight)
    return max(1, min(intensity, 10))  # Ensure it's between 1-10

# Preprocess data
def preprocess_data(df):
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Convert fatigue level to numeric
    if 'Predicted Fatigue Level' in df.columns:
        fatigue_mapping = {'Low': 1, 'Moderate': 2, 'High': 3}
        df['Fatigue_Numeric'] = df['Predicted Fatigue Level'].map(fatigue_mapping)
    
    # Handle dates
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['month'] = df['Date'].dt.month
    
    # Handle categorical features with consistent encoding
    if 'Stroke Type' in df.columns:
        stroke_mapping = {
            'Freestyle': 0,
            'Backstroke': 1,
            'Breaststroke': 2,
            'Butterfly': 3,
            'Medley': 4,
            'Rest': 5
        }
        df['Stroke Type'] = df['Stroke Type'].map(lambda x: stroke_mapping.get(x, 0))
    
    # Calculate intensity if not provided
    if 'Intensity' not in df.columns:
        df['Intensity'] = df.apply(calculate_intensity, axis=1)
    
    # Handle other categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        if col != 'Predicted Fatigue Level' and col != 'Date':
            df[col] = df[col].astype('category').cat.codes
    
    return df

# Enhance features with deterministic processing
def create_enhanced_features(df):
    # Create swimmer-specific features
    enhanced_dfs = []
    
    for swimmer_id, swimmer_data in df.groupby('Swimmer ID'):
        # Create a copy to avoid warnings
        swimmer_data = swimmer_data.copy()
        
        # Sort by date deterministically
        swimmer_data = swimmer_data.sort_values('Date')
        
        # Create lag features (previous fatigue levels)
        for i in range(1, 4):
            swimmer_data[f'fatigue_lag_{i}'] = swimmer_data['Fatigue_Numeric'].shift(i)
        
        # Create rolling window features
        if 'Training Distance ' in swimmer_data.columns:
            swimmer_data['distance_7day'] = swimmer_data['Training Distance '].rolling(window=min(7, len(swimmer_data)), min_periods=1).sum()
            swimmer_data['distance_std_7day'] = swimmer_data['Training Distance '].rolling(window=min(7, len(swimmer_data)), min_periods=1).std()
        
        swimmer_data['intensity_7day'] = swimmer_data['Intensity'].rolling(window=min(7, len(swimmer_data)), min_periods=1).mean()
        
        if 'Session Duration (hrs)' in swimmer_data.columns:
            swimmer_data['duration_7day'] = swimmer_data['Session Duration (hrs)'].rolling(window=min(7, len(swimmer_data)), min_periods=1).sum()
        
        # Calculate rest metrics
        if 'Rest hours' in swimmer_data.columns:
            swimmer_data['rest_ratio_7day'] = swimmer_data['Rest hours'].rolling(window=min(7, len(swimmer_data)), min_periods=1).sum() / (24 * min(7, len(swimmer_data)))
        
        if 'Recovery Days' in swimmer_data.columns:
            swimmer_data['rest_days_7day'] = swimmer_data['Recovery Days'].rolling(window=min(7, len(swimmer_data)), min_periods=1).sum()
        
        enhanced_dfs.append(swimmer_data)
    
    if not enhanced_dfs:
        return pd.DataFrame()
    
    enhanced_df = pd.concat(enhanced_dfs, ignore_index=True)
    
    # Fill NaN values with deterministic values
    numeric_cols = enhanced_df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if enhanced_df[col].isna().any():
            median_val = enhanced_df[col].median()
            if pd.isna(median_val):
                enhanced_df[col] = enhanced_df[col].fillna(0)
            else:
                enhanced_df[col] = enhanced_df[col].fillna(median_val)
    
    return enhanced_df