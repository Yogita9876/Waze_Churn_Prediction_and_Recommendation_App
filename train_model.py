from churn_predictor import WazeChurnModeler
import joblib
import os
import json
import mlflow
import mlflow.sklearn
from datetime import datetime
import pandas as pd

def main():
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("waze-churn-prediction")
    
    # Initialize modeler
    modeler = WazeChurnModeler()
    
    try:
        # Load and preprocess data
        X_train, X_test, y_train, y_test = modeler.load_processed_data(processed_data_path='data/processed')
        
        print("Data loaded successfully")
        print(f"Training data shape: {X_train.shape}")
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"training_run_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            # Train and evaluate models
            print("\nTraining baseline model...")
            baseline_model = modeler.train_baseline_model(X_train, y_train)
            modeler.evaluate_model(baseline_model, X_test, y_test, 'baseline')
            
            print("\nTraining Random Forest...")
            rf_model = modeler.train_random_forest(X_train, y_train)
            modeler.evaluate_model(rf_model, X_test, y_test, 'random_forest')
            
            print("\nTraining LightGBM...")
            lgb_model = modeler.train_lightgbm(X_train, y_train, X_test, y_test)
            modeler.evaluate_model(lgb_model, X_test, y_test, 'lightgbm')
            
            # Select best model
            best_model = modeler.select_best_model(metric='roc_auc')
            
            if best_model is not None:
                print("\nGenerating feature importance...")
                importance_df = modeler.analyze_feature_importance(X_train)
                
                if importance_df is not None:
                    # Save feature importance
                    importance_path = os.path.join('models', 'feature_importance.csv')
                    importance_df.to_csv(importance_path, index=False)
                    print(f"Feature importance saved to: {importance_path}")
                    
                    # Log with MLflow
                    mlflow.log_artifact(importance_path, "feature_importance")
                else:
                    # Create a basic feature importance if SHAP analysis fails
                    if hasattr(best_model, 'feature_importances_'):
                        importance_df = pd.DataFrame({
                            'feature': X_train.columns,
                            'importance': best_model.feature_importances_
                        })
                    else:
                        # For models without feature_importances_
                        importance_df = pd.DataFrame({
                            'feature': X_train.columns,
                            'importance': [1/len(X_train.columns)] * len(X_train.columns)
                        })
                    importance_df = importance_df.sort_values('importance', ascending=False)
                    importance_df.to_csv('models/feature_importance.csv', index=False)
                    print("Basic feature importance saved as fallback")
                
                # Save the best model
                model_path = os.path.join('models', 'churn_model.joblib')
                joblib.dump(best_model, model_path)
                print(f"Best model saved to: {model_path}")
                
                # Save model metrics
                best_model_name = [name for name, model in modeler.models.items() if model == best_model][0]
                metrics = modeler.model_scores[best_model_name]
                
                metrics_path = os.path.join('models', 'model_metrics.json')
                with open(metrics_path, 'w') as f:
                    json.dump({
                        'model_type': best_model_name,
                        'metrics': metrics
                    }, f, indent=4)
                print(f"Model metrics saved to: {metrics_path}")
                
                # Log best model with MLflow
                mlflow.sklearn.log_model(best_model, "best_model")
                
                print(f"\nTraining completed successfully!")
                print(f"Best model: {best_model_name}")
                print(f"ROC-AUC Score: {metrics['roc_auc']:.4f}")
            else:
                print("Error: No best model selected")
                
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()