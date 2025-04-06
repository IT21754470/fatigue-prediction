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