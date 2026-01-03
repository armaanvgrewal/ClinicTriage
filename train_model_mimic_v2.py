"""
ClinicFlow Model Training - MIMIC-IV-ED (Improved Version)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score
)
from sklearn.utils.class_weight import compute_class_weight
import pickle
from datetime import datetime

print("=" * 70)
print("CLINICFLOW MODEL TRAINING V2 - MIMIC-IV-ED")
print("Improved hyperparameters + class weighting")
print("=" * 70)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n📂 Loading MIMIC-IV-ED data...")

df = pd.read_csv('mimic_patients_10k.csv')
print(f"   ✅ Loaded {len(df):,} patient records from MIMIC-IV-ED")

# ============================================================================
# PREPARE FEATURES
# ============================================================================

print("\n🔧 Preparing features...")

feature_columns = [
    'age',
    'symptom_severity',
    'symptom_duration_hours',
    'heart_rate',
    'systolic_bp',
    'diastolic_bp',
    'temperature',
    'oxygen_saturation',
    'has_red_flag',
    'has_chronic_condition',
    'high_risk_chronic',
    'hr_abnormal',
    'bp_abnormal',
    'temp_abnormal',
    'spo2_abnormal',
    'vital_abnormalities',
    'symptom_acuity',
    'previous_visits',
    'gender_encoded',
    'onset_encoded'
]

X = df[feature_columns]
y = df['urgency_level']

print(f"   Feature matrix: {X.shape}")
print(f"   Target variable: {y.shape}")

# Check class distribution
print(f"\n   Class distribution:")
for level in sorted(y.unique()):
    count = (y == level).sum()
    pct = count / len(y) * 100
    print(f"      Level {int(level)}: {count:4d} ({pct:5.1f}%)")

# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================

print("\n✂️  Splitting data (stratified)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"   Training: {len(X_train):,} records")
print(f"   Testing:  {len(X_test):,} records")

# ============================================================================
# IMPROVED MODEL WITH BETTER HYPERPARAMETERS
# ============================================================================

print("\n🤖 Training improved Random Forest...")

# Improved hyperparameters based on best practices for medical data
model = RandomForestClassifier(
    n_estimators=300,           # More trees (was 200)
    max_depth=20,               # Deeper trees (was 15)
    min_samples_split=5,        # Less restrictive (was 10)
    min_samples_leaf=2,         # Less restrictive (was 5)
    max_features='sqrt',        # Feature sampling
    class_weight='balanced',    # Handle imbalanced classes
    random_state=42,
    n_jobs=-1,
    bootstrap=True,
    oob_score=True              # Out-of-bag score for validation
)

start_time = datetime.now()
model.fit(X_train, y_train)
training_time = (datetime.now() - start_time).total_seconds()

print(f"   ✅ Model trained in {training_time:.2f} seconds")
print(f"   Out-of-bag score: {model.oob_score_:.1%}")

# ============================================================================
# EVALUATE
# ============================================================================

print("\n📊 Evaluating performance...")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n   Overall Accuracy: {accuracy:.1%}")
print(f"   Weighted F1 Score: {f1:.1%}")

print("\n   Classification Report:")
print(classification_report(y_test, y_pred, 
      target_names=['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']))

# Critical patient accuracy (most important!)
critical_mask = y_test.isin([1, 2])
critical_accuracy = accuracy_score(y_test[critical_mask], y_pred[critical_mask])
print(f"   🚨 Critical Case Accuracy (L1-2): {critical_accuracy:.1%}")

# Feature importance
print("\n🔍 Top 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']:25s}: {row['importance']:.3f}")

# ============================================================================
# CONFUSION MATRIX
# ============================================================================

print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# ============================================================================
# SAVE MODEL
# ============================================================================

print("\n💾 Saving improved MIMIC-IV trained model...")

with open('triage_model_mimic_v2.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

metadata = {
    'training_date': datetime.now().isoformat(),
    'data_source': 'MIMIC-IV-ED',
    'model_version': 'v2_improved',
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'accuracy': accuracy,
    'f1_score': f1,
    'critical_accuracy': critical_accuracy,
    'oob_score': model.oob_score_,
    'features': feature_columns,
    'hyperparameters': {
        'n_estimators': 300,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'class_weight': 'balanced'
    }
}

with open('model_metadata_mimic_v2.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print(f"   ✅ Saved as 'triage_model_mimic_v2.pkl'")

print("\n" + "=" * 70)
print("IMPROVED MODEL TRAINING COMPLETE!")
print("=" * 70)

print(f"\n🎯 Model Performance Summary:")
print(f"   • Version: v2 (Improved Hyperparameters)")
print(f"   • Overall Accuracy: {accuracy:.1%}")
print(f"   • Critical Case Accuracy: {critical_accuracy:.1%}")
print(f"   • Weighted F1: {f1:.1%}")
print(f"   • OOB Score: {model.oob_score_:.1%}")

if accuracy > 0.75:
    improvement = (accuracy - 0.75) * 100
    print(f"\n   ✅ IMPROVEMENT: +{improvement:.1f} percentage points!")
else:
    print(f"\n   ⚠️  Similar to v1, trying v3 recommended")