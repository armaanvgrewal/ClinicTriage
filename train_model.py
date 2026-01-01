"""
ClinicFlow Machine Learning Model Training
Trains a Random Forest classifier to predict patient urgency levels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score
)
import pickle
from datetime import datetime

print("=" * 70)
print("CLINICFLOW MODEL TRAINING")
print("=" * 70)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\n📂 Step 1: Loading data...")

df = pd.read_csv('synthetic_patients.csv')
print(f"   ✅ Loaded {len(df)} patient records")
print(f"   ✅ Features: {len(df.columns)} columns")

# ============================================================================
# STEP 2: PREPARE DATA FOR MACHINE LEARNING
# ============================================================================

print("\n🔧 Step 2: Preparing data for machine learning...")

# Select features for the model
# We'll use features that are always available at intake
feature_columns = [
    # Demographics
    'age',
    
    # Symptom information
    'symptom_severity',
    'symptom_duration_hours',
    
    # Vital signs
    'heart_rate',
    'systolic_bp',
    'diastolic_bp',
    'temperature',
    'oxygen_saturation',
    
    # Clinical flags
    'has_red_flag',
    'has_chronic_condition',
    'high_risk_chronic',
    
    # Engineered features
    'hr_abnormal',
    'bp_abnormal',
    'temp_abnormal',
    'spo2_abnormal',
    'vital_abnormalities',
    'symptom_acuity',
    
    # History
    'previous_visits'
]

# We need to encode categorical variables
# Convert gender to numeric (0 = Male, 1 = Female)
df['gender_encoded'] = (df['gender'] == 'Female').astype(int)
feature_columns.append('gender_encoded')

# Convert symptom_onset to numeric (0 = Gradual, 1 = Sudden)
df['onset_encoded'] = (df['symptom_onset'] == 'Sudden').astype(int)
feature_columns.append('onset_encoded')

print(f"   Selected {len(feature_columns)} features for model:")
for i, feature in enumerate(feature_columns, 1):
    print(f"      {i:2d}. {feature}")

# Extract features (X) and target (y)
X = df[feature_columns]
y = df['urgency_level']

print(f"\n   ✅ Feature matrix shape: {X.shape}")
print(f"   ✅ Target variable shape: {y.shape}")

# Check for any missing values
if X.isnull().sum().sum() > 0:
    print(f"   ⚠️  Found {X.isnull().sum().sum()} missing values - filling with median")
    X = X.fillna(X.median())
else:
    print(f"   ✅ No missing values")

# ============================================================================
# STEP 3: SPLIT DATA INTO TRAIN AND TEST SETS
# ============================================================================

print("\n✂️  Step 3: Splitting data into train and test sets...")

# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing
    random_state=42,    # Reproducible results
    stratify=y          # Keep same urgency distribution in both sets
)

print(f"   Training set: {len(X_train)} patients ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Test set:     {len(X_test)} patients ({len(X_test)/len(X)*100:.1f}%)")

# Verify urgency distribution is similar
print(f"\n   Urgency distribution in training set:")
train_dist = y_train.value_counts(normalize=True).sort_index()
for level, pct in train_dist.items():
    print(f"      Level {level}: {pct*100:5.1f}%")

print(f"\n   Urgency distribution in test set:")
test_dist = y_test.value_counts(normalize=True).sort_index()
for level, pct in test_dist.items():
    print(f"      Level {level}: {pct*100:5.1f}%")

# ============================================================================
# STEP 4: TRAIN THE MODEL
# ============================================================================

print("\n🤖 Step 4: Training Random Forest model...")
print("   (This may take 30-60 seconds...)")

# Create the Random Forest classifier
model = RandomForestClassifier(
    n_estimators=200,       # Number of trees in the forest
    max_depth=15,           # Maximum depth of each tree
    min_samples_split=10,   # Minimum samples to split a node
    min_samples_leaf=5,     # Minimum samples in leaf node
    random_state=42,        # Reproducible results
    n_jobs=-1,              # Use all CPU cores
    class_weight='balanced' # Handle class imbalance
)

# Train the model
start_time = datetime.now()
model.fit(X_train, y_train)
training_time = (datetime.now() - start_time).total_seconds()

print(f"   ✅ Model trained in {training_time:.2f} seconds")
print(f"   ✅ Random Forest with {model.n_estimators} trees")

# ============================================================================
# STEP 5: EVALUATE THE MODEL
# ============================================================================

print("\n📊 Step 5: Evaluating model performance...")

# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n   Overall Accuracy: {accuracy:.1%}")

# Calculate weighted F1 score (better for imbalanced classes)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"   Weighted F1 Score: {f1:.1%}")

# Detailed classification report
print("\n   📋 Classification Report (per urgency level):")
report = classification_report(y_test, y_pred, target_names=[
    'Level 1 (Critical)',
    'Level 2 (High Risk)', 
    'Level 3 (Moderate)',
    'Level 4 (Low Risk)',
    'Level 5 (Non-urgent)'
])
print(report)

# Calculate accuracy for critical cases (Level 1 and 2)
critical_mask = y_test.isin([1, 2])
critical_accuracy = accuracy_score(
    y_test[critical_mask], 
    y_pred[critical_mask]
)
print(f"   🚨 Critical Case Accuracy (Levels 1-2): {critical_accuracy:.1%}")

# Check if we ever missed a Level 1 as Level 4 or 5 (dangerous!)
level1_mask = y_test == 1
level1_predictions = y_pred[level1_mask]
dangerous_misses = sum(level1_predictions >= 4)
print(f"   ⚠️  Critical patients (Level 1) missed as low priority: {dangerous_misses}")

# ============================================================================
# STEP 6: ANALYZE FEATURE IMPORTANCE
# ============================================================================

print("\n🔍 Step 6: Analyzing feature importance...")

# Get feature importances
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n   Top 10 Most Important Features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"      {row['feature']:25s} {row['importance']:.4f}")

# ============================================================================
# STEP 7: CREATE VISUALIZATIONS
# ============================================================================

print("\n📊 Step 7: Creating visualizations...")

fig = plt.figure(figsize=(16, 12))

# Plot 1: Confusion Matrix
ax1 = plt.subplot(2, 3, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['L1', 'L2', 'L3', 'L4', 'L5'],
            yticklabels=['L1', 'L2', 'L3', 'L4', 'L5'])
ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
ax1.set_ylabel('True Urgency Level')
ax1.set_xlabel('Predicted Urgency Level')

# Plot 2: Feature Importance (Top 15)
ax2 = plt.subplot(2, 3, 2)
top_features = feature_importance.head(15)
ax2.barh(range(len(top_features)), top_features['importance'], color='steelblue')
ax2.set_yticks(range(len(top_features)))
ax2.set_yticklabels(top_features['feature'])
ax2.set_xlabel('Importance Score')
ax2.set_title('Top 15 Feature Importance', fontsize=14, fontweight='bold')
ax2.invert_yaxis()

# Plot 3: Accuracy by Urgency Level
ax3 = plt.subplot(2, 3, 3)
accuracies = []
for level in [1, 2, 3, 4, 5]:
    mask = y_test == level
    if mask.sum() > 0:
        acc = accuracy_score(y_test[mask], y_pred[mask])
        accuracies.append(acc)
    else:
        accuracies.append(0)
colors = ['darkred', 'red', 'orange', 'lightblue', 'lightgreen']
bars = ax3.bar([1, 2, 3, 4, 5], accuracies, color=colors, edgecolor='black')
ax3.set_xlabel('Urgency Level')
ax3.set_ylabel('Accuracy')
ax3.set_title('Accuracy by Urgency Level', fontsize=14, fontweight='bold')
ax3.set_ylim([0, 1.1])
ax3.grid(axis='y', alpha=0.3)
# Add percentage labels on bars
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.1%}',
             ha='center', va='bottom', fontweight='bold')

# Plot 4: Prediction Error Distribution
ax4 = plt.subplot(2, 3, 4)
errors = y_pred - y_test
ax4.hist(errors, bins=np.arange(-4.5, 5.5, 1), color='coral', edgecolor='black')
ax4.set_xlabel('Prediction Error (Predicted - Actual)')
ax4.set_ylabel('Frequency')
ax4.set_title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
ax4.axvline(x=0, color='green', linestyle='--', linewidth=2, label='Perfect Prediction')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# Plot 5: Model Performance Summary
ax5 = plt.subplot(2, 3, 5)
ax5.axis('off')
summary_text = f"""
MODEL PERFORMANCE SUMMARY

Overall Metrics:
  • Accuracy: {accuracy:.1%}
  • F1 Score: {f1:.1%}
  • Critical Accuracy (L1-2): {critical_accuracy:.1%}

Training Info:
  • Training samples: {len(X_train)}
  • Test samples: {len(X_test)}
  • Features used: {len(feature_columns)}
  • Training time: {training_time:.2f}s

Safety Check:
  • L1 missed as low priority: {dangerous_misses}
  
Top 3 Features:
  1. {feature_importance.iloc[0]['feature']}
  2. {feature_importance.iloc[1]['feature']}
  3. {feature_importance.iloc[2]['feature']}
"""
ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes,
         fontsize=11, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Plot 6: Precision-Recall by Level
ax6 = plt.subplot(2, 3, 6)
from sklearn.metrics import precision_score, recall_score

levels = [1, 2, 3, 4, 5]
precisions = []
recalls = []

for level in levels:
    # Create binary classification (this level vs all others)
    y_test_binary = (y_test == level).astype(int)
    y_pred_binary = (y_pred == level).astype(int)
    
    if y_test_binary.sum() > 0:
        prec = precision_score(y_test_binary, y_pred_binary, zero_division=0)
        rec = recall_score(y_test_binary, y_pred_binary, zero_division=0)
    else:
        prec = 0
        rec = 0
    
    precisions.append(prec)
    recalls.append(rec)

x = np.arange(len(levels))
width = 0.35

bars1 = ax6.bar(x - width/2, precisions, width, label='Precision', color='skyblue')
bars2 = ax6.bar(x + width/2, recalls, width, label='Recall', color='lightcoral')

ax6.set_xlabel('Urgency Level')
ax6.set_ylabel('Score')
ax6.set_title('Precision & Recall by Urgency Level', fontsize=14, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(['L1', 'L2', 'L3', 'L4', 'L5'])
ax6.legend()
ax6.set_ylim([0, 1.1])
ax6.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('model_performance.png', dpi=150, bbox_inches='tight')
print("   ✅ Visualizations saved as 'model_performance.png'")

# ============================================================================
# STEP 8: SAVE THE MODEL
# ============================================================================

print("\n💾 Step 8: Saving trained model...")

# Save the model
model_filename = 'triage_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)
print(f"   ✅ Model saved as '{model_filename}'")

# Save feature names (we'll need these for predictions)
feature_names_file = 'feature_names.pkl'
with open(feature_names_file, 'wb') as f:
    pickle.dump(feature_columns, f)
print(f"   ✅ Feature names saved as '{feature_names_file}'")

# Save model metadata
metadata = {
    'training_date': datetime.now().isoformat(),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'accuracy': accuracy,
    'f1_score': f1,
    'features': feature_columns,
    'feature_importance': feature_importance.to_dict('records')
}

metadata_file = 'model_metadata.pkl'
with open(metadata_file, 'wb') as f:
    pickle.dump(metadata, f)
print(f"   ✅ Metadata saved as '{metadata_file}'")

# ============================================================================
# STEP 9: TEST THE MODEL WITH EXAMPLE PATIENTS
# ============================================================================

print("\n🧪 Step 9: Testing model with example patients...")

# Create a few test cases
test_cases = [
    {
        'name': 'Critical - Chest Pain + High HR',
        'age': 65,
        'gender_encoded': 0,
        'symptom_severity': 9,
        'symptom_duration_hours': 1.0,
        'onset_encoded': 1,  # Sudden
        'heart_rate': 120,
        'systolic_bp': 160,
        'diastolic_bp': 95,
        'temperature': 98.6,
        'oxygen_saturation': 92,
        'has_red_flag': 1,
        'has_chronic_condition': 1,
        'high_risk_chronic': 1,
        'hr_abnormal': 1,
        'bp_abnormal': 1,
        'temp_abnormal': 0,
        'spo2_abnormal': 1,
        'vital_abnormalities': 3,
        'symptom_acuity': 13.5,
        'previous_visits': 2,
        'expected': 1
    },
    {
        'name': 'Moderate - Sprained Ankle',
        'age': 28,
        'gender_encoded': 1,
        'symptom_severity': 4,
        'symptom_duration_hours': 2.0,
        'onset_encoded': 1,  # Sudden
        'heart_rate': 75,
        'systolic_bp': 118,
        'diastolic_bp': 78,
        'temperature': 98.4,
        'oxygen_saturation': 99,
        'has_red_flag': 0,
        'has_chronic_condition': 0,
        'high_risk_chronic': 0,
        'hr_abnormal': 0,
        'bp_abnormal': 0,
        'temp_abnormal': 0,
        'spo2_abnormal': 0,
        'vital_abnormalities': 0,
        'symptom_acuity': 6.0,
        'previous_visits': 1,
        'expected': 4
    },
    {
        'name': 'Non-urgent - Medication Refill',
        'age': 45,
        'gender_encoded': 1,
        'symptom_severity': 1,
        'symptom_duration_hours': 48.0,
        'onset_encoded': 0,  # Gradual
        'heart_rate': 72,
        'systolic_bp': 125,
        'diastolic_bp': 82,
        'temperature': 98.6,
        'oxygen_saturation': 98,
        'has_red_flag': 0,
        'has_chronic_condition': 1,
        'high_risk_chronic': 0,
        'hr_abnormal': 0,
        'bp_abnormal': 0,
        'temp_abnormal': 0,
        'spo2_abnormal': 0,
        'vital_abnormalities': 0,
        'symptom_acuity': 1.0,
        'previous_visits': 5,
        'expected': 5
    }
]

print("\n   Testing on example patients:")
for case in test_cases:
    # Extract features in correct order
    patient_features = [case[feat] for feat in feature_columns]
    
    # Make prediction
    prediction = model.predict([patient_features])[0]
    
    # Get prediction probabilities
    probabilities = model.predict_proba([patient_features])[0]
    
    result = "✅ CORRECT" if prediction == case['expected'] else "❌ WRONG"
    print(f"\n   {case['name']}")
    print(f"      Expected: Level {case['expected']}")
    print(f"      Predicted: Level {prediction} {result}")
    print(f"      Confidence: {probabilities[prediction-1]:.1%}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETE!")
print("=" * 70)

print(f"\n✅ Key Results:")
print(f"   • Overall Accuracy: {accuracy:.1%}")
print(f"   • F1 Score: {f1:.1%}")
print(f"   • Critical Case Accuracy: {critical_accuracy:.1%}")
print(f"   • Dangerous Misses (L1→L4/5): {dangerous_misses}")

print(f"\n✅ Files Created:")
print(f"   • triage_model.pkl - Trained model")
print(f"   • feature_names.pkl - Feature list")
print(f"   • model_metadata.pkl - Training info")
print(f"   • model_performance.png - Visualizations")

print(f"\n✅ Model is ready for:")
print(f"   • Integration into Streamlit app")
print(f"   • Real-time patient triage")
print(f"   • Queue optimization")

print("\n💡 Next steps:")
print("   1. Review model_performance.png")
print("   2. Check if accuracy meets requirements (>85% recommended)")
print("   3. Move to Phase 4: Queue Optimization Algorithm")

print("\n" + "=" * 70)