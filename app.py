from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
from models.model_utils import preprocess_data

app = Flask(__name__)

# Load the dataset
try:
    df = pd.read_csv('adult.csv')
    print(f"Dataset loaded successfully with {len(df)} records")
except Exception as e:
    print(f"Error loading dataset: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        data = request.get_json()
        print(f"Received data: {data}")
        
        # Check if required fields are present
        required_fields = ['age', 'workclass', 'fnlwgt', 'education', 'occupation',
                          'gender', 'capital-gain', 'capital-loss', 
                          'hours-per-week', 'native-country']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f"Missing required fields: {', '.join(missing_fields)}"
            })
            
        # Add missing fields with default values
        if 'educational-num' not in data:
            education_map = {
                'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4,
                '9th': 5, '10th': 6, '11th': 7, '12th': 8, 'HS-grad': 9,
                'Some-college': 10, 'Assoc-voc': 11, 'Assoc-acdm': 12,
                'Bachelors': 13, 'Masters': 14, 'Prof-school': 15, 'Doctorate': 16
            }
            data['educational-num'] = education_map.get(data['education'], 0)
            
        if 'race' not in data:
            data['race'] = 'White'  # Default value
            
        if 'marital-status' not in data:
            data['marital-status'] = 'Never-married'  # Default value
            
        if 'relationship' not in data:
            data['relationship'] = 'Not-in-family'  # Default value
            
        # Check if model exists
        model_path = 'models/income_predictor.pkl'
        if not os.path.exists(model_path):
            # Use simulated prediction for testing based on rules
            print(f"Model file not found: {model_path}, using rule-based prediction")
            # Logic for determining income based on key factors
            high_income = False
            
            # Simple rules based on the Adult dataset patterns:
            age = int(data['age'])
            education = data['education']
            hours = int(data['hours-per-week'])
            occupation = data['occupation']
            capital_gain = int(data['capital-gain'])
            
            # Check different profile types
            # Young person with basic education and service job: Likely <=50K
            if age < 30 and education in ['HS-grad', 'Some-college', '11th', '12th', '10th', '9th'] and \
               occupation in ['Other-service', 'Adm-clerical', 'Handlers-cleaners', 'Farming-fishing'] and hours < 40:
                high_income = False
                print("Rule applied: Young worker with basic education")
            # Young person regardless of job with no capital gain: Likely <=50K    
            elif age < 25 and capital_gain < 1000:
                high_income = False
                print("Rule applied: Young worker with no capital gain")
            # Executive with higher education working long hours: Likely >50K
            elif age > 35 and education in ['Bachelors', 'Masters', 'Doctorate'] and \
                 occupation in ['Exec-managerial', 'Prof-specialty'] and hours >= 45:
                high_income = True
                print("Rule applied: Executive with higher education")
            # Capital gain is a strong predictor
            elif capital_gain > 5000:  # Significant capital gain often indicates >50K
                high_income = True
                print("Rule applied: High capital gain")
            # Default to demographic averages
            else:
                # Men over 35 with good education: More likely >50K
                if data['gender'] == 'Male' and age >= 35 and \
                   education in ['Bachelors', 'Masters', 'Doctorate', 'Prof-school']:
                    high_income = True
                    print("Rule applied: Older male with higher education")
                else:
                    high_income = False
                    print("Rule applied: Default demographic")
            
            return jsonify({
                'success': True,
                'prediction': '>50K' if high_income else '<=50K',
                'note': 'This is a simulated prediction based on demographic rules since the model file is missing'
            })
        
        # Preprocess the input data
        processed_data = preprocess_data(data)
        
        # Load the model
        model = joblib.load(model_path)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        
        # Override specific edge cases where the model might be wrong
        # Young service worker with basic education rule
        if (int(data['age']) < 30 and 
            data['education'] in ['HS-grad', 'Some-college', '11th', '12th', '10th', '9th'] and
            data['occupation'] in ['Other-service', 'Handlers-cleaners', 'Adm-clerical'] and
            int(data['hours-per-week']) < 40 and
            int(data['capital-gain']) < 1000):
            # This profile should almost always be <=50K
            prediction = 0
            override_note = "Prediction adjusted based on demographic rules (young worker profile)"
        else:
            override_note = None
        
        # For debugging, include input data characteristics in response
        debug_info = {
            'input_age': int(data['age']),
            'input_education': data['education'],
            'input_occupation': data['occupation'],
            'input_hours': int(data['hours-per-week']),
            'input_gender': data['gender'],
        }
        
        response_data = {
            'success': True,
            'prediction': '>50K' if prediction == 1 else '<=50K',
            'debug': debug_info
        }
        
        if override_note:
            response_data['note'] = override_note
            
        return jsonify(response_data)
    except Exception as e:
        import traceback
        print(f"Error in prediction: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/model-status')
def model_status():
    """Check if the model file exists"""
    model_path = 'models/income_predictor.pkl'
    return jsonify({
        'model_exists': os.path.exists(model_path),
        'model_path': model_path
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 