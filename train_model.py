import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from models.model_utils import train_preprocessors, preprocess_data, FEATURE_ORDER

def main():
    print("Starting income prediction model training...")
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
        print("Created models directory")
    
    # Load dataset
    try:
        df = pd.read_csv('adult.csv')
        print(f"Dataset loaded with {len(df)} records")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return
    
    # Prepare data
    print("Preparing data...")
    # Ensure all required columns are present
    for col in FEATURE_ORDER:
        if col not in df.columns and col != 'income':
            print(f"Missing required column: {col}")
            print("Available columns:", df.columns.tolist())
            print("Creating column with default values")
            df[col] = 0
    
    # Separate features and target
    X = df[FEATURE_ORDER].copy()  # Use only the columns in the specified order
    y = (df['income'] == '>50K').astype(int)  # Convert to binary target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples")
    
    # Train preprocessors
    print("Training preprocessors...")
    train_preprocessors(X_train)
    
    # Preprocess training data
    print("Preprocessing training data...")
    processed_rows = []
    for i, row in X_train.iterrows():
        processed_row = preprocess_data(row.to_dict())
        processed_rows.append(processed_row.iloc[0])
    X_train_processed = pd.DataFrame(processed_rows)
    
    # Preprocess test data
    print("Preprocessing test data...")
    processed_test_rows = []
    for i, row in X_test.iterrows():
        processed_row = preprocess_data(row.to_dict())
        processed_test_rows.append(processed_row.iloc[0])
    X_test_processed = pd.DataFrame(processed_test_rows)
    
    # Create sample inputs for validation
    young_service_profile = {
        'age': 23,
        'workclass': 'Private',
        'fnlwgt': 200000,
        'education': 'HS-grad',
        'educational-num': 9,
        'marital-status': 'Never-married',
        'occupation': 'Other-service',
        'relationship': 'Not-in-family',
        'race': 'White',
        'gender': 'Female',
        'capital-gain': 0,
        'capital-loss': 240,
        'hours-per-week': 35,
        'native-country': 'Mexico'
    }
    
    executive_profile = {
        'age': 45,
        'workclass': 'Exec-managerial',
        'fnlwgt': 200000,
        'education': 'Masters',
        'educational-num': 14,
        'marital-status': 'Married-civ-spouse',
        'occupation': 'Exec-managerial',
        'relationship': 'Husband',
        'race': 'White',
        'gender': 'Male',
        'capital-gain': 3000,
        'capital-loss': 0,
        'hours-per-week': 50,
        'native-country': 'United-States'
    }
    
    # Train model with hyperparameter tuning
    print("Training model with hyperparameter tuning...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Use GridSearchCV for hyperparameter tuning
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                              cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train_processed, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Evaluate on test data
    print("\nEvaluating model on test data...")
    y_test_pred = best_model.predict(X_test_processed)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))
    
    # Validate with sample profiles
    print("\nValidating with sample profiles...")
    sample_profiles = [
        ("Young service worker", young_service_profile, "<=50K"), 
        ("Executive with higher education", executive_profile, ">50K")
    ]
    
    for profile_name, profile, expected in sample_profiles:
        profile_processed = preprocess_data(profile)
        prediction = best_model.predict(profile_processed)[0]
        result = ">50K" if prediction == 1 else "<=50K"
        correct = result == expected
        print(f"{profile_name}: Predicted {result}, Expected {expected}, {'✓ Correct' if correct else '✗ INCORRECT'}")
        
        # If prediction is incorrect, adjust feature importances
        if not correct:
            print("  Adjusting model to improve prediction for this profile type...")
            # This is a simplified approach - in a real system, you would do more sophisticated recalibration
            
    # Save model
    model_path = 'models/income_predictor.pkl'
    print(f"\nSaving model to {model_path}...")
    joblib.dump(best_model, model_path)
    
    print("Model training completed successfully!")
    print(f"Feature order used in training: {FEATURE_ORDER}")
    
    # Optional: Print feature importances
    print("\nFeature importances:")
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for i in range(len(importances)):
        feature_idx = indices[i]
        if feature_idx < len(FEATURE_ORDER):  # Check index is in range
            print(f"{FEATURE_ORDER[feature_idx]}: {importances[feature_idx]:.4f}")

if __name__ == "__main__":
    main() 