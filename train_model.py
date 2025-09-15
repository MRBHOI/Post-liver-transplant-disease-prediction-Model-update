import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import xgboost as xgb
from lightgbm import LGBMClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
def load_data(file_path):
    df = pd.read_csv(file_path)
    
    # Select recipient features (R_) and target
    recipient_cols = [col for col in df.columns if col.startswith('R_')]
    feature_cols = recipient_cols
    
    # Prepare features
    X = df[feature_cols].copy()
    y = df['Complications']
    
    # Handle categorical variables
    X['R_Gender'] = (X['R_Gender'] == 'Male').astype(int)
    
    # Fill missing values with appropriate defaults
    defaults = {
        'R_Age': X['R_Age'].median(),
        'R_BMI': X['R_BMI'].median(),
        'R_MELD_Score': X['R_MELD_Score'].median(),
        'R_Albumin_level': X['R_Albumin_level'].median(),
        'R_Na': X['R_Na'].median(),
        'R_Mg': X['R_Mg'].median(),
        'R_WBC': X['R_WBC'].median(),
        'R_Lympochyte': X['R_Lympochyte'].median(),
        'R_Platelets': X['R_Platelets'].median(),
        'R_Cold_Ischemia_Time': X['R_Cold_Ischemia_Time'].median(),
        'R_Warm_Ischemia_Time': X['R_Warm_Ischemia_Time'].median(),
        'R_Rejection_Episodes': 0
    }
    
    # Fill missing values
    for col in X.columns:
        if col in defaults:
            X[col] = X[col].fillna(defaults[col])
        else:
            X[col] = X[col].fillna(0)  # Fill binary/categorical with 0
    
    # Drop any non-numeric columns that we don't want to use
    X = X.drop(['R_Etiology', 'R_Immunosuppressant_Medication'], axis=1, errors='ignore')
    
    return X, y

def preprocess_data(df):
    # Keep all columns except 'Column1' and target
    feature_cols = [col for col in df.columns if col != 'Column1' and col != 'Complications']
    
    # Handle categorical variables
    categorical_cols = ['D_Gender', 'D_Cause_of_Death', 'R_Gender', 'R_Etiology', 'R_Immunosuppressant_Medication']
    
    # Convert binary fields
    binary_fields = ['D_Diabetes', 'D_Hypertension', 'D_Alcohol_Abuse', 'D_Smoking',
                    'D_Hepatitis_B', 'D_Hepatitis_C', 'R_Diabetes', 'R_Hypertension',
                    'R_Alcohol_Abuse', 'R_Smoking', 'R_Hepatitis_B', 'R_Hepatitis_C',
                    'R_Alcoholic_cirrhosis', 'R_Primary_biliary_cirrhosis', 'R_Blood_Transfusion']
    
    numeric_fields = ['D_Age', 'D_BMI', 'D_Lympochyte', 'R_MELD_Score', 'R_Age',
                     'R_BMI', 'R_Lympochyte', 'R_Albumin_level', 'R_Na', 'R_Mg',
                     'R_WBC', 'R_Platelets', 'R_Cold_Ischemia_Time',
                     'R_Warm_Ischemia_Time', 'R_Rejection_Episodes']
    
    # Handle categorical variables
    df_processed = pd.get_dummies(df[categorical_cols], prefix=categorical_cols)
    
    # Process binary fields
    for field in binary_fields:
        df_processed[field] = df[field].fillna(0).astype(int)
    
    # Process numeric fields
    for field in numeric_fields:
        df[field] = df[field].fillna(df[field].median())
        
    # Add numeric fields to processed dataframe
    df_processed = pd.concat([df_processed, df[numeric_fields]], axis=1)
    
    # Create interaction features
    df_processed['MELD_Age_Interaction'] = df['R_MELD_Score'] * df['R_Age']
    df_processed['D_R_Age_Diff'] = df['D_Age'] - df['R_Age']
    df_processed['D_R_BMI_Ratio'] = df['D_BMI'] / df['R_BMI']
    df_processed['Ischemia_Total'] = df['R_Cold_Ischemia_Time'] + df['R_Warm_Ischemia_Time']
    
    # Medical risk scores
    df_processed['Donor_Risk_Score'] = (
        df['D_Age'] / 100 * 0.3 +
        df['D_Diabetes'].fillna(0).astype(int) * 0.2 +
        df['D_Hypertension'].fillna(0).astype(int) * 0.2 +
        (df['D_BMI'] > 30).astype(int) * 0.3
    )
    
    df_processed['Recipient_Risk_Score'] = (
        df['R_Age'] / 100 * 0.2 +
        df['R_MELD_Score'] / 40 * 0.3 +
        df['R_Diabetes'].fillna(0).astype(int) * 0.2 +
        (df['R_BMI'] > 30).astype(int) * 0.3
    )
    
    # Compatibility score
    df_processed['Compatibility_Score'] = (
        (1 - abs(df['D_Age'] - df['R_Age']) / 100) * 0.4 +
        (1 - abs(df['D_BMI'] - df['R_BMI']) / 50) * 0.3 +
        (df['D_Gender'] == df['R_Gender']).astype(int) * 0.3
    )
    
    return df_processed

def train_model(X, y):
    # Convert target to numerical
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize scalers
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to dataframe to keep column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y_train)
    class_weight_dict = dict(zip(np.unique(y), class_weights))
    
    # Create base models with optimized parameters
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight=class_weight_dict,
        random_state=42,
        n_jobs=-1
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=2,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,
        random_state=42,
        n_jobs=-1
    )
    
    # Create voting classifier
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('gb', gb),
            ('xgb', xgb_model)
        ],
        voting='soft'
    )
    
    # Fit the model
    print("Training ensemble model...")
    voting_clf.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = voting_clf.predict(X_test_scaled)
    y_pred_proba = voting_clf.predict_proba(X_test_scaled)
    
    # Convert predictions back to original labels
    y_test_original = le.inverse_transform(y_test)
    y_pred_original = le.inverse_transform(y_pred)
    
    # Print model performance
    print("\nClassification Report:")
    print(classification_report(y_test_original, y_pred_original))
    
    # Calculate and print accuracy
    accuracy = accuracy_score(y_test_original, y_pred_original)
    print(f"\nAccuracy Score: {accuracy:.3f}")
    
    # Calculate and print ROC AUC score
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    print(f"\nROC AUC Score: {roc_auc:.3f}")
    
    # Get feature importance from random forest
    rf_model = voting_clf.named_estimators_['rf']
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 Most Important Features:")
    print(feature_importance.head(20))
    
    # Save the model and preprocessing objects
    print("\nSaving model...")
    model_data = {
        'model': voting_clf,
        'scaler': scaler,
        'label_encoder': le,
        'features': X_train.columns.tolist(),
        'classes': le.classes_.tolist(),
        'feature_importance': feature_importance.to_dict('records'),
        'accuracy': accuracy,
        'roc_auc': roc_auc
    }
    
    with open('models/liver_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Model saved successfully!")
    
    return model_data

def main():
    print("Loading data...")
    X, y = load_data('LiverT_dataset_upd.csv')
    
    print("Training model...")
    model_data = train_model(X, y)
    
    print("\nTraining completed!")
    print(f"Model Accuracy: {model_data['accuracy']:.3f}")
    print(f"ROC AUC Score: {model_data['roc_auc']:.3f}")

if __name__ == "__main__":
    main()
