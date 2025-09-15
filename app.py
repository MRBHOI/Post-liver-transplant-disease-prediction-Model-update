from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load the model and metadata
try:
    with open('models/liver_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    print("Model loaded successfully!")
    print(f"Model accuracy: {model_data['accuracy']:.3f}")
    print(f"Number of features: {len(model_data['features'])}")
    print(f"Classes: {model_data['classes']}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

def get_risk_level(probabilities, threshold_high=0.7, threshold_moderate=0.4):
    """Get risk level based on probability distribution with adjustable thresholds"""
    max_prob = np.max(probabilities)
    if max_prob > threshold_high:
        return "High Risk"
    elif max_prob > threshold_moderate:
        return "Moderate Risk"
    else:
        return "Low Risk"

def get_complication_details(probabilities, classes, feature_importance):
    """Get detailed information about predicted complications"""
    complications = []
    for i, prob in enumerate(probabilities):
        if prob > 0.1:  # Only include complications with >10% probability
            complication = {
                'name': classes[i],
                'probability': float(prob),
                'description': f"Risk of {classes[i].lower()} based on patient factors",
                'contributing_factors': []
            }
            
            # Add top contributing factors
            for factor in feature_importance[:3]:  # Top 3 factors
                complication['contributing_factors'].append(factor['feature'])
            
            complications.append(complication)
    
    return sorted(complications, key=lambda x: x['probability'], reverse=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        form_data = request.get_json()
        if not form_data:
            return jsonify({
                'status': 'error',
                'message': 'No form data received'
            }), 400
        
        print("Received form data:", form_data)
        
        # Map form field names to model feature names
        field_mapping = {
            'r_age': 'R_Age',
            'r_bmi': 'R_BMI',
            'r_meld_score': 'R_MELD_Score',
            'r_diabetes': 'R_Diabetes',
            'r_hypertension': 'R_Hypertension',
            'r_alcohol_abuse': 'R_Alcohol_Abuse',
            'r_smoking': 'R_Smoking',
            'r_hepatitis_b': 'R_Hepatitis_B',
            'r_hepatitis_c': 'R_Hepatitis_C',
            'r_albumin_level': 'R_Albumin_level',
            'r_na': 'R_Na',
            'r_mg': 'R_Mg',
            'r_wbc': 'R_WBC',
            'r_lympochyte': 'R_Lympochyte',
            'r_platelets': 'R_Platelets',
            'r_cold_ischemia_time': 'R_Cold_Ischemia_Time',
            'r_warm_ischemia_time': 'R_Warm_Ischemia_Time',
            'r_blood_transfusion': 'R_Blood_Transfusion',
            'r_alcoholic_cirrhosis': 'R_Alcoholic_cirrhosis',
            'r_primary_biliary_cirrhosis': 'R_Primary_biliary_cirrhosis',
            'r_gender': 'R_Gender',
            'r_rejection_episodes': 'R_Rejection_Episodes'
        }
        
        # Create input data dictionary with mapped field names
        input_data = {}
        defaults = {
            'R_Age': 50,
            'R_BMI': 25,
            'R_MELD_Score': 15,
            'R_Albumin_level': 40,
            'R_Na': 140,
            'R_Mg': 2,
            'R_WBC': 7000,
            'R_Lympochyte': 30,
            'R_Platelets': 200000,
            'R_Cold_Ischemia_Time': 360,
            'R_Warm_Ischemia_Time': 30,
            'R_Rejection_Episodes': 0,
            'R_Gender': 0,
            'R_Diabetes': 0,
            'R_Hypertension': 0,
            'R_Alcohol_Abuse': 0,
            'R_Smoking': 0,
            'R_Hepatitis_B': 0,
            'R_Hepatitis_C': 0,
            'R_Alcoholic_cirrhosis': 0,
            'R_Primary_biliary_cirrhosis': 0,
            'R_Blood_Transfusion': 0
        }
        
        for form_field, model_field in field_mapping.items():
            try:
                value = form_data.get(form_field)
                if value is None or value == '':
                    value = defaults[model_field]
                elif model_field in ['R_Gender']:
                    value = 1 if str(value).lower() == 'male' else 0
                elif model_field in ['R_Diabetes', 'R_Hypertension', 'R_Alcohol_Abuse', 'R_Smoking',
                                   'R_Hepatitis_B', 'R_Hepatitis_C', 'R_Alcoholic_cirrhosis',
                                   'R_Primary_biliary_cirrhosis', 'R_Blood_Transfusion']:
                    value = int(float(value))
                else:
                    value = float(value)
                input_data[model_field] = value
            except (ValueError, TypeError) as e:
                print(f"Error converting {model_field}: {str(e)}")
                input_data[model_field] = defaults[model_field]
        
        # Create DataFrame with single row
        input_df = pd.DataFrame([input_data])
        
        # Ensure all model features are present
        for feature in model_data['features']:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Reorder columns to match training data
        input_df = input_df[model_data['features']]
        
        # Scale the features
        input_df_scaled = pd.DataFrame(
            model_data['scaler'].transform(input_df),
            columns=input_df.columns
        )
        
        # Make prediction
        try:
            probabilities = model_data['model'].predict_proba(input_df_scaled)[0]
            risk_level = get_risk_level(probabilities)
            complications = get_complication_details(probabilities, model_data['classes'], model_data['feature_importance'])
            
            # Calculate overall health score (0-100)
            health_score = 100 * (1 - np.max(probabilities))
            
            return jsonify({
                'status': 'success',
                'risk_level': risk_level,
                'health_score': round(health_score, 1),
                'complications': complications
            })
        except Exception as e:
            print(f"Error in prediction step: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': 'Error making prediction. Please check your input values.'
            }), 500
            
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': 'An unexpected error occurred. Please try again.'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
