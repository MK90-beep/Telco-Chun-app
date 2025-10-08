import pandas as pd
import joblib
from flask import Flask, request, render_template, jsonify
import os

# =====================================
#  Initialize Flask App
# =====================================
app = Flask(__name__)

# =====================================
#  Load Model
# =====================================
try:
    model = joblib.load('models/best_churn_model.pkl')
    print("✅ Model loaded successfully!")
    print(f"Model type: {type(model)}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# =====================================
#  Define Manual Feature Names
# =====================================
FEATURE_NAMES = [
    'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'OnlineBackup',
    'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 'gender_Male',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineBackup_No internet service', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year',
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check'
]
print(f"Loaded {len(FEATURE_NAMES)} manual feature names.")

# =====================================
#  Preprocessing Function
# =====================================
def preprocess_input(data):
    """Preprocess single input to match model training features."""
    input_df = pd.DataFrame([data])

    # Convert Yes/No columns to 1/0
    yes_no_cols = [
        'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
        'OnlineBackup', 'OnlineSecurity', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    for col in yes_no_cols:
        if col in input_df.columns:
            input_df[col] = input_df[col].map({'Yes': 1, 'No': 0})

    # One-hot encode categorical columns
    categorical_cols = [
        'gender', 'MultipleLines', 'InternetService',
        'Contract', 'PaymentMethod'
    ]
    input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

    # Add any missing columns expected by model
    for col in FEATURE_NAMES:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure same order as model training
    input_df = input_df[FEATURE_NAMES]

    print("✅ Preprocessing complete. Columns aligned with model.")
    return input_df

# =====================================
#  Home Route
# =====================================
@app.route('/')
def home():
    return render_template('index.html', prediction={})

# =====================================
#  Prediction Route
# =====================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return render_template('index.html', prediction={'error': 'Model not loaded. Please check model path.'})

        # Collect and prepare input
        data = request.form.to_dict()
        print("\n=== Incoming Data ===")
        print(data)

        # Convert numeric values safely
        data['SeniorCitizen'] = int(data.get('SeniorCitizen', 0))
        data['MonthlyCharges'] = float(data.get('MonthlyCharges', 0))
        data['TotalCharges'] = float(data.get('TotalCharges', 0))
        data['tenure'] = int(data.get('tenure', 12))

        # Provide defaults for missing values
        defaults = {
            'gender': 'Male',
            'Partner': 'No',
            'Dependents': 'No',
            'PhoneService': 'Yes',
            'OnlineBackup': 'No',
            'PaperlessBilling': 'Yes',
            'MultipleLines': 'No',
            'OnlineSecurity': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'InternetService': 'Fiber optic',
            'Contract': 'Month-to-month',
            'PaymentMethod': 'Electronic check'
        }

        for key, val in defaults.items():
            data.setdefault(key, val)

        # ==============================
        # Preprocess and Predict
        # ==============================
        processed_data = preprocess_input(data)
        print(f"Processed shape: {processed_data.shape}")
        print("Columns:", processed_data.columns.tolist())

        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]

        # Improved confidence levels
        if probability >= 0.8 or probability <= 0.2:
            confidence = "High"
        elif 0.6 <= probability < 0.8 or 0.2 < probability <= 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"

        result = {
            'prediction': 'Churn' if prediction == 1 else 'No Churn',
            'probability': round(probability * 100, 2),
            'confidence': confidence
        }

        print("\n=== Prediction Result ===")
        print(result)

        return render_template('index.html', prediction=result)

    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        print(error_msg)
        return render_template('index.html', prediction={'error': error_msg})

# =====================================
#  Feature Inspection Route
# =====================================
@app.route('/features')
def show_features():
    if model and hasattr(model, 'feature_names_in_'):
        return jsonify({'features': model.feature_names_in_.tolist()})
    else:
        return jsonify({'features': FEATURE_NAMES, 'note': 'Manual feature list used.'})

# =====================================
#  Run App
# =====================================
if __name__ == '__main__':
    print("Starting Telco Churn Prediction App...")
    print(f"Template folder exists: {os.path.exists('templates')}")
    print(f"Index.html exists: {os.path.exists('templates/index.html')}")
    app.run(debug=True, host='0.0.0.0', port=5000)


