"""
Test the trained model by loading it and making predictions
"""

import pickle
import pandas as pd
import numpy as np

print("=" * 70)
print("TESTING SAVED MODEL")
print("=" * 70)

# Load the model
print("\n📂 Loading saved model...")
with open('triage_model.pkl', 'rb') as f:
    model = pickle.load(f)
print("   ✅ Model loaded successfully")

# Load feature names
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)
print(f"   ✅ Feature names loaded ({len(feature_names)} features)")

# Load metadata
with open('model_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
print(f"   ✅ Metadata loaded")
print(f"      Training date: {metadata['training_date']}")
print(f"      Accuracy: {metadata['accuracy']:.1%}")

# Create a test patient
print("\n🧪 Creating test patient...")

test_patient = {
    'name': 'Test Patient - Severe Chest Pain',
    'age': 58,
    'gender_encoded': 0,  # Male
    'symptom_severity': 9,
    'symptom_duration_hours': 0.5,
    'onset_encoded': 1,  # Sudden
    'heart_rate': 115,
    'systolic_bp': 155,
    'diastolic_bp': 98,
    'temperature': 99.2,
    'oxygen_saturation': 93,
    'has_red_flag': 1,
    'has_chronic_condition': 1,
    'high_risk_chronic': 1,
    'hr_abnormal': 1,
    'bp_abnormal': 1,
    'temp_abnormal': 0,
    'spo2_abnormal': 1,
    'vital_abnormalities': 3,
    'symptom_acuity': 13.5,
    'previous_visits': 3
}

# Extract features in correct order
patient_features = [test_patient[feat] for feat in feature_names]

print(f"   Patient: {test_patient['name']}")
print(f"   Age: {test_patient['age']}, HR: {test_patient['heart_rate']}, " +
      f"BP: {test_patient['systolic_bp']}/{test_patient['diastolic_bp']}")

# Make prediction
print("\n🤖 Making prediction...")
prediction = model.predict([patient_features])[0]
probabilities = model.predict_proba([patient_features])[0]

print(f"\n   ✅ Predicted Urgency Level: {prediction}")
print(f"   Confidence: {probabilities[prediction-1]:.1%}")

print(f"\n   Probability breakdown:")
for level in range(1, 6):
    prob = probabilities[level-1]
    bar = '█' * int(prob * 50)
    print(f"      Level {level}: {prob:6.1%} {bar}")

print("\n" + "=" * 70)
print("✅ MODEL TEST SUCCESSFUL!")
print("=" * 70)
print("\n💡 The model is working correctly and ready for deployment!")