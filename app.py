import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

from src.preprocessing.preprocessing import FeatureDropper, OutlierHandler, FeatureCreator, MulticollinearityHandler, CategoricalEncoder

# Load models, preprocessor and feature names
@st.cache_resource
def load_models():
    model = joblib.load('models/churn_model.joblib')
    preprocessor = joblib.load('models/preprocessor_pipeline.joblib')
    feature_importance = pd.read_csv('models/feature_importance.csv')
    raw_features = pd.read_csv('data/processed/raw_feature_names.csv')['0'].tolist()
    return model, preprocessor, feature_importance, raw_features

def get_feature_explanation(feature_name):
    """
    Convert technical feature names to business-friendly explanations
    """
    feature_mapping = {
        'total_sessions': 'Total App Sessions',
        'n_days_after_onboarding': 'Days Since Sign Up',
        'total_navigations_fav2': 'Secondary Location Usage',
        'driven_km_drives': 'Total Distance Driven',
        'duration_minutes_drives': 'Total Drive Time',
        'device': 'Device Type',
        'Drives_per_day_last_month': 'Recent Daily Drive Frequency',
        'activity_ratio_last_month': 'Recent Activity Level',
        'driving_days_ratio': 'Regular Usage Pattern',
        'total_navigations': 'Navigation Frequency',
        'fav1_and_2_navigation_ratio': 'Saved Locations Usage',
        'Average_distance_per_driving_days': 'Average Daily Distance',
        'Average_drives_per_driving_day': 'Drives per Active Day',
        'Average_min_per_drive': 'Average Trip Duration',
        'Percentage_of_sessions_in_last_month': 'Recent Usage Trend'
    }
    return feature_mapping.get(feature_name, feature_name)

def format_churn_probability(prob):
    """
    Format probability as percentage and risk level with color coding
    """
    percentage = f"{prob * 100:.1f}%"
    if prob > 0.7:
        return percentage, "🔴 High Risk", "red"
    elif prob > 0.4:
        return percentage, "🟡 Medium Risk", "orange"
    else:
        return percentage, "🟢 Low Risk", "green"

def get_detailed_recommendations(churn_prob, user_data):
    """
    Generate detailed, actionable recommendations based on user metrics
    """
    recommendations = []
    
    # High Risk Recommendations
    if churn_prob > 0.7:
        recommendations.extend([
            {
                "priority": "IMMEDIATE ACTION REQUIRED",
                "action": "Implement retention campaign",
                "details": "User shows high likelihood of churning. Consider direct outreach within 24-48 hours."
            }
        ])
        
        if user_data['activity_days'].iloc[0] < 15:
            recommendations.append({
                "priority": "High Priority",
                "action": "Boost user engagement",
                "details": "Send personalized push notifications highlighting traffic updates and route suggestions for frequently visited places."
            })
        
        if user_data['drives'].iloc[0] < 5:
            recommendations.append({
                "priority": "High Priority",
                "action": "Increase feature adoption",
                "details": "Provide tutorial on key features like real-time traffic updates and route optimization."
            })
            
    # Medium Risk Recommendations
    elif churn_prob > 0.4:
        recommendations.extend([
            {
                "priority": "MONITOR CLOSELY",
                "action": "Enhance user experience",
                "details": "Focus on increasing app usage through targeted feature recommendations."
            }
        ])
        
        if user_data['activity_days'].iloc[0] < 20:
            recommendations.append({
                "priority": "Medium Priority",
                "action": "Maintain engagement",
                "details": "Send weekly digest of nearby traffic patterns and suggested alternate routes."
            })
            
    # Low Risk Recommendations
    else:
        recommendations.extend([
            {
                "priority": "MAINTAIN ENGAGEMENT",
                "action": "Reward loyal usage",
                "details": "Consider user for beta features and loyalty rewards program."
            }
        ])
        
        if user_data['total_sessions'].iloc[0] > 50:
            recommendations.append({
                "priority": "Low Priority",
                "action": "Leverage user expertise",
                "details": "Invite user to beta testing program and encourage community participation."
            })
    
    return recommendations

def process_prediction(input_data, model, preprocessor, feature_importance, raw_features):
    try:
        user_id = input_data['ID'].iloc[0]
        input_data_ordered = input_data[raw_features]
        
        # Preprocess and predict
        processed_data = preprocessor.transform(input_data_ordered)
        processed_df = pd.DataFrame(processed_data, columns=preprocessor.named_steps['multicollinearity'].features_to_keep_)
        churn_prob = model.predict_proba(processed_df)[0][1]
        
        # Format probability and risk level
        prob_formatted, risk_level, risk_color = format_churn_probability(churn_prob)
        
        # Get feature importance with actual feature names
        importance_df = feature_importance.copy()
        importance_df['feature_explanation'] = importance_df['feature'].apply(get_feature_explanation)
        
        return {
            "user_summary": {
                "User ID": user_id,
                "Churn Risk": prob_formatted,
                "Risk Level": risk_level,
                "Risk Color": risk_color
            },
            "key_factors": importance_df[['feature_explanation', 'importance']].head(5).to_dict('records'),
            "recommendations": get_detailed_recommendations(churn_prob, input_data_ordered)
        }
    except Exception as e:
        print(f"Detailed error in process_prediction: {str(e)}")
        raise

def main():
    st.set_page_config(page_title="Waze User Churn Predictor", layout="wide")
    
    st.title("🎯 Waze User Churn Risk Analysis")
    st.markdown("""
    This tool analyzes user behavior to predict the likelihood of user churn and provides actionable recommendations.
    """)
    
    try:
        model, preprocessor, feature_importance, raw_features = load_models()
        
        # Create two columns for input and example
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("📝 Input User Data")
            json_input = st.text_area("Paste user data in JSON format:", height=300)
        
        with col2:
            st.subheader("📋 Example Format")
            json_example = {field: "value" for field in raw_features}
            json_example.update({
                "ID": "user_123",
                "sessions": 10,
                "drives": 8,
                "device": "iPhone"
            })
            st.code(json.dumps(json_example, indent=2), language="json")
        
        if st.button("📊 Analyze User"):
            try:
                input_dict = json.loads(json_input)
                
                # Validate required fields
                missing_fields = [field for field in raw_features if field not in input_dict]
                if missing_fields:
                    st.error(f"❌ Missing required fields: {', '.join(missing_fields)}")
                    return
                
                input_data = pd.DataFrame([input_dict])
                results = process_prediction(input_data, model, preprocessor, feature_importance, raw_features)
                
                # Display results in a structured format
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🎯 Churn Risk Assessment")
                    risk_color = results["user_summary"]["Risk Color"]
                    st.markdown(f"""
                    ### User ID: {results["user_summary"]["User ID"]}
                    - **Risk Level:** <span style='color: {risk_color}'>{results["user_summary"]["Risk Level"]}</span>
                    - **Churn Probability:** {results["user_summary"]["Churn Risk"]}
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.subheader("🔑 Key Factors")
                    for factor in results["key_factors"]:
                        impact_percentage = factor['importance'] * 100
                        st.markdown(f"""
                        - **{factor['feature_explanation']}** 
                          (Impact: {impact_percentage:.1f}%)
                        """)
                
                st.subheader("📋 Recommended Actions")
                for rec in results["recommendations"]:
                    st.markdown(f"""
                    #### {rec['priority']}
                    - **Action:** {rec['action']}
                    - **Details:** {rec['details']}
                    ---
                    """)
                
            except json.JSONDecodeError:
                st.error("❌ Invalid JSON format. Please check your input.")
            except Exception as e:
                st.error(f"❌ Error processing input: {str(e)}")
                
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    main()