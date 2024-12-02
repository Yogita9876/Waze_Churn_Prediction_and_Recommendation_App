import pytest
import pandas as pd
import numpy as np
from preprocessing import FeatureDropper, OutlierHandler, FeatureCreator

def test_feature_dropper():
    # Create sample data
    data = pd.DataFrame({
        'ID': [1, 2, 3],
        'feature1': [4, 5, 6],
        'feature2': [7, 8, 9]
    })
    
    # Test dropping ID column
    dropper = FeatureDropper(columns_to_drop=['ID'])
    result = dropper.fit_transform(data)
    
    assert 'ID' not in result.columns
    assert len(result.columns) == 2

def test_outlier_handler():
    # Create sample data with outliers
    data = pd.DataFrame({
        'values': [1, 2, 3, 100, 2, 3]  # 100 is an outlier
    })
    
    # Test outlier handling
    handler = OutlierHandler(method='clip')
    result = handler.fit_transform(data)
    
    assert result['values'].max() < 100  # Outlier should be clipped

def test_feature_creator():
    # Create sample data
    data = pd.DataFrame({
        'sessions': [10, 20],
        'drives': [5, 10],
        'duration_minutes_drives': [50, 100],
        'driven_km_drives': [100, 200],
        'activity_days': [15, 25],
        'driving_days': [10, 20],
        'total_navigations_fav1': [2, 4],
        'total_navigations_fav2': [3, 6],
        'total_sessions': [20, 40]
    })
    
    creator = FeatureCreator()
    result = creator.fit_transform(data)
    
    # Test if new features are created
    expected_features = [
        'Sessions_per_day_last_month',
        'Drives_per_day_last_month',
        'Average_drive_duration'
    ]
    
    for feature in expected_features:
        assert feature in result.columns