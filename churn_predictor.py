import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import shap
from joblib import load
import os.path

class WazeChurnModeler:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.feature_importance = None
        self.model_scores = {}
        

    def load_processed_data(self, processed_data_path='data/processed'):
        """Load the preproced data"""
        print("Loading pre-processed data...")
        X_train = pd.read_csv(os.path.join(processed_data_path, 'X_train.csv'))
        X_test = pd.read_csv(os.path.join(processed_data_path, 'X_test.csv'))
        y_train = pd.read_csv(os.path.join(processed_data_path, 'y_train.csv')).values.ravel()
        y_test = pd.read_csv(os.path.join(processed_data_path, 'y_test.csv')).values.ravel()
        return X_train, X_test, y_train, y_test
    
    
    
    def train_baseline_model(self, X_train, y_train):
        """Train a simple logistic regression as baseline"""
        print("Training baseline Logistic Regression model...")
        lr_model = LogisticRegression(random_state=42)
        lr_model.fit(X_train, y_train)
        self.models['baseline'] = lr_model
        return lr_model
    
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest with hyperparameter tuning"""
        print("Training Random Forest model...")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        print("Best Random Forest parameters:", grid_search.best_params_)
        self.models['random_forest'] = grid_search.best_estimator_
        return grid_search.best_estimator_
    
    def train_lightgbm(self, X_train, y_train, X_test, y_test):
        """Train LightGBM model with early stopping"""
        print("Training LightGBM model...")
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test)
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        }
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=10, show_stdv=False)
            ]
        )
        
        self.models['lightgbm'] = model
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        if isinstance(model, lgb.basic.Booster):
            y_pred = (model.predict(X_test) > 0.5).astype(int)
            y_prob = model.predict(X_test)
        else:
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }
        
        print(f"\nResults for {model_name}:")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self._plot_confusion_matrix(y_test, y_pred, model_name)
        
        self.model_scores[model_name] = results
        return results
    
    def _plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def select_best_model(self, metric='roc_auc'):
        """Select the best performing model based on specified metric"""
        best_score = -1
        best_model_name = None
        
        for name, model in self.models.items():
            if name in self.model_scores and self.model_scores[name][metric] > best_score:
                best_score = self.model_scores[name][metric]
                best_model_name = name
                self.best_model = model
        
        print(f"Best model selected: {best_model_name} with {metric} of {best_score:.4f}")
        return self.best_model
    
    def analyze_feature_importance(self, X):
        """Analyze feature importance for the best model"""
        if not self.best_model:
            print("No best model selected. Please evaluate models first.")
            return None
        
        try:
            if isinstance(self.best_model, lgb.basic.Booster):
                # For LightGBM
                importance = self.best_model.feature_importance(importance_type='gain')
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': importance / importance.sum()  # Normalize
                })
            elif isinstance(self.best_model, RandomForestClassifier):
                # For Random Forest
                importance = self.best_model.feature_importances_
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': importance
                })
            elif isinstance(self.best_model, LogisticRegression):
                # For Logistic Regression - use absolute coefficients
                importance = np.abs(self.best_model.coef_[0])
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': importance / importance.sum()  # Normalize
                })
            else:
                print(f"Model type {type(self.best_model)} not supported for importance analysis")
                return None
                
            return importance_df.sort_values('importance', ascending=False)
                
        except Exception as e:
            print(f"Error in feature importance analysis: {e}")
            return None