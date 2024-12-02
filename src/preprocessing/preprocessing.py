import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from joblib import dump, load
from imblearn.over_sampling import SMOTE



class FeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        if self.columns_to_drop:
            return X.drop(columns=[col for col in self.columns_to_drop if col in X.columns])
        return X


class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, columns_config=None, method='clip'):
        self.columns_config = columns_config
        self.method = method
        self.bounds_ = {}
        
    def fit(self, X, y=None):
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        
        for col in numeric_cols:
            if self.columns_config and col in self.columns_config:
                if self.columns_config[col] == 'iqr':
                    Q1 = X[col].quantile(0.25)
                    Q3 = X[col].quantile(0.75)
                    IQR = Q3 - Q1
                    self.bounds_[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
                else:
                    self.bounds_[col] = self.columns_config[col]
            else:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                self.bounds_[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        return self
    
    
    def transform(self, X):
        X = X.copy()
        for col, bounds in self.bounds_.items():
            if col in X.columns:
                lower_bound, upper_bound = bounds
                if self.method == 'clip':
                    X[col] = X[col].clip(lower_bound, upper_bound)
                elif self.method == 'remove':
                    X = X[(X[col] >= lower_bound) & (X[col] <= upper_bound)]
        return X
    
    
    
class FeatureCreator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
            
    def transform(self, X):
        X = X.copy()
        
        
        # User engagement features
        X['Sessions_per_day_last_month'] = X['sessions'] / 30
        X['Drives_per_day_last_month'] = X['drives'] / 30
        
        # Safe division using stored parameters for inference
        X['Average_drive_duration'] = np.where(X['drives'] != 0, 
                                            X['duration_minutes_drives'] / X['drives'], 0)
        X['Average_drive_distance'] = np.where(X['drives'] != 0, 
                                            X['driven_km_drives'] / X['drives'], 0)
        
        # Activity patterns
        X['activity_ratio_last_month'] = X['activity_days'] / 30
        X['driving_days_ratio'] = np.where(X['activity_days'] != 0,
                                        X['driving_days'] / X['activity_days'], 0)
        
        # Navigation behavior
        X['total_navigations'] = X['total_navigations_fav1'] + X['total_navigations_fav2']
        X['fav1_and_2_navigation_ratio'] = np.where(X['total_sessions'] != 0,
                                                    X['total_navigations'] / X['total_sessions'], 
                                                    0)
        
        # Intensity metrics using stored parameters for safe division
        X['Average_distance_per_driving_days'] = np.where(X['driving_days'] != 0,
                                                        X['driven_km_drives'] / X['driving_days'], 
                                                        0)
        
        X['Average_drives_per_driving_day'] = np.where(X['driving_days'] != 0,
                                                    X['drives'] / X['driving_days'], 
                                                    0)
        
        X['Average_km_per_drive'] = np.where(X['drives'] != 0,
                                            X['driven_km_drives'] / X['drives'], 0)
        
        X['Average_min_per_drive'] = np.where(X['drives'] != 0,
                                            X['duration_minutes_drives'] / X['drives'], 0)
        
        X['Percentage_of_sessions_in_last_month'] = np.where(X['total_sessions'] != 0,
                                                            X['sessions'] / X['total_sessions'],0)
        
        # # Drop original features used in calculations
        # features_to_drop = ['sessions', 'drives', 'duration_minutes_drives', 'driven_km_drives',
        #                   'activity_days', 'driving_days', 'total_navigations_fav1',
        #                   'total_navigations_fav2', 'total_sessions']
        
        
        #  # Drop ID column as it's not needed for modeling
        # features_to_drop.append('ID')
        
        # X = X.drop(columns=features_to_drop)
        
        return X
    
class MulticollinearityHandler(BaseEstimator, TransformerMixin):
    def __init__(self, correlation_threshold=0.85):
        self.correlation_threshold = correlation_threshold
        self.features_to_keep_ = None
        
    def fit(self, X, y=None):
        numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
        corr_matrix = X[numeric_columns].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = set()
        for col in upper.columns:
            if col not in to_drop:
                correlated = upper[col][upper[col] > self.correlation_threshold].index
                if len(correlated) > 0:
                    to_drop.update(correlated)
                    
        self.features_to_keep_ = [col for col in X.columns if col not in to_drop]
        return self
        
    def transform(self, X):
        X_transformed = X[self.features_to_keep_]
        # Ensure the transformed DataFrame maintains column names
        return pd.DataFrame(X_transformed, columns=self.features_to_keep_)
  
    
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        X['device'] = (X['device'] == 'iPhone').astype(int)
        return X
    
    


def create_pipeline():
    return Pipeline([
       ('feature_dropper', FeatureDropper(columns_to_drop=['ID'])),
       ('outlier_handler', OutlierHandler(method='clip')),
       ('feature_creator', FeatureCreator()),
       ('multicollinearity', MulticollinearityHandler(correlation_threshold=0.85)),
       ('categorical_encoder', CategoricalEncoder()),
       ('scaler', StandardScaler())  # Using sklearn's StandardScaler
    ])
    
    
    
def main():
    # Load data
    df = pd.read_csv('data/raw/waze_data.csv')
    
    
    #Drop rows with missing target values
    #It's important to do it before splitting and label encoding the target class 
    df = df.dropna(subset=['label'])
    
    
    # Split data and encode target column label to 1 if churned else 0 
    X = df.drop('label', axis=1)
    y = (df['label'] == 'churned').astype(int)
    
    
    # Save raw feature
    pd.DataFrame(X.columns).to_csv('data/processed/raw_feature_names.csv', index=False)
    
    
    #spiit the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and fit pipeline
    pipeline = create_pipeline()
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)
    
    
    # Get the final feature names from MulticollinearityHandler
    final_feature_names = pipeline.named_steps['multicollinearity'].features_to_keep_
    
    # Convert numpy array back to DataFrame with feature names
    X_train_transformed = pd.DataFrame(X_train_transformed, columns=final_feature_names)
    X_test_transformed = pd.DataFrame(X_test_transformed, columns=final_feature_names)
    
    # Create feature mapping DataFrame
    feature_mapping = pd.DataFrame({
        'column_index': range(len(final_feature_names)),
        'feature_name': final_feature_names
    })
    
    
    
    # Save feature mapping
    feature_mapping.to_csv('data/processed/feature_mapping.csv', index=False)
    
    # Apply SMOTE AFTER preprocessing but ONLY to training data
    print("Class distribution before SMOTE:", np.bincount(y_train))
    smote = SMOTE(random_state=42)
    X_train_transformed_smote, y_train_smote = smote.fit_resample(
        X_train_transformed, y_train
    )
    print("Class distribution after SMOTE:", np.bincount(y_train_smote))
    
    
    # Convert SMOTE output back to DataFrame with feature names
    X_train_transformed_smote = pd.DataFrame(X_train_transformed_smote, columns=final_feature_names)
    
    # Save pipeline and processed data
    dump(pipeline, 'models/preprocessor_pipeline.joblib')
    
    # Save processed data
    X_train_transformed_smote.to_csv('data/processed/X_train.csv', index=False)
    X_test_transformed.to_csv('data/processed/X_test.csv', index=False)
    pd.DataFrame(y_train_smote, columns=['label']).to_csv('data/processed/y_train.csv', index=False)
    pd.DataFrame(y_test, columns=['label']).to_csv('data/processed/y_test.csv', index=False)
    
    
    
    
    print(f"X_train shape after SMOTE: {X_train_transformed_smote.shape}")
    print(f"X_test shape: {X_test_transformed.shape}")
    print(f"Feature mapping saved to: data/processed/feature_mapping.csv")
    print(f"Number of features after preprocessing: {len(final_feature_names)}")

if __name__ == "__main__":
    main()


  

    
    
    