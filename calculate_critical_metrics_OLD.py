"""
Calculate Critical Detection Rate and Critical Exact Accuracy
for any saved triage model
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ============================================================================
# CONFIGURATION - Change this to the model you want to evaluate
# ============================================================================
MODEL_FILE = 'triage_model_mimic_v2_OLD.pkl'  # Change as needed
METADATA_FILE = 'model_metadata_mimic_v2_OLD.pkl'  # Change as needed
DATA_FILE = 'mimic_patients_10k.csv'

# ============================================================================
# LOAD MODEL AND DATA
# ============================================================================

print(f"Loading model: {MODEL_FILE}")
with open(MODEL_FILE, 'rb') as f:
    model = pickle.load(f)

print(f"Loading data: {DATA_FILE}")
df = pd.read_csv(DATA_FILE)

# Get feature names
try:
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
except:
    feature_names = list(model.feature_names_in_)

print(f"Features: {len(feature_names)}")

# ============================================================================
# PREPARE TEST DATA (same split as training)
# ============================================================================

X = df[feature_names]
y = df['urgency_level']

# Use same random state as training for consistent test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Test set size: {len(y_test)}")

# ============================================================================
# MAKE PREDICTIONS
# ============================================================================

y_pred = model.predict(X_test)

# ============================================================================
# CALCULATE METRICS
# ============================================================================

# Overall accuracy
overall_accuracy = (y_pred == y_test).mean()

# Critical cases are Level 1 and Level 2
actual_critical = (y_test <= 2)
predicted_critical = (y_pred <= 2)

# True Positives: Actually critical AND predicted critical
TP = ((actual_critical) & (predicted_critical)).sum()

# False Negatives: Actually critical BUT predicted non-critical
FN = ((actual_critical) & (~predicted_critical)).sum()

# False Positives: Actually non-critical BUT predicted critical
FP = ((~actual_critical) & (predicted_critical)).sum()

# True Negatives: Actually non-critical AND predicted non-critical
TN = ((~actual_critical) & (~predicted_critical)).sum()

# Critical Detection Rate = TP / (TP + FN)
critical_detection_rate = TP / (TP + FN) if (TP + FN) > 0 else 0

# Critical Exact Accuracy = correct L1/L2 predictions / total actual L1/L2
critical_mask = (y_test <= 2)
critical_exact_accuracy = (y_pred[critical_mask] == y_test[critical_mask]).mean()

# Precision for critical
critical_precision = TP / (TP + FP) if (TP + FP) > 0 else 0

# ============================================================================
# PRINT RESULTS
# ============================================================================

print("\n" + "=" * 60)
print(f"METRICS FOR: {MODEL_FILE}")
print("=" * 60)
print(f"\nOverall Accuracy:        {overall_accuracy:.1%}")
print(f"\nCRITICAL CASE METRICS:")
print(f"  Critical Detection Rate: {critical_detection_rate:.1%}")
print(f"  Critical Exact Accuracy: {critical_exact_accuracy:.1%}")
print(f"  Critical Precision:      {critical_precision:.1%}")
print(f"\nBINARY CONFUSION MATRIX:")
print(f"  True Positives (TP):   {TP}")
print(f"  False Negatives (FN):  {FN}")
print(f"  False Positives (FP):  {FP}")
print(f"  True Negatives (TN):   {TN}")
print(f"\nTotal Critical Cases:    {TP + FN}")
print(f"Total Non-Critical:      {FP + TN}")

# ============================================================================
# OPTIONALLY UPDATE METADATA FILE
# ============================================================================

update = input("\nUpdate metadata file with these metrics? (y/n): ")
if update.lower() == 'y':
    try:
        with open(METADATA_FILE, 'rb') as f:
            metadata = pickle.load(f)
    except:
        metadata = {}
    
    metadata['critical_detection_rate'] = critical_detection_rate
    metadata['critical_exact_accuracy'] = critical_exact_accuracy
    metadata['critical_precision'] = critical_precision
    metadata['binary_confusion'] = {'TP': int(TP), 'FN': int(FN), 'FP': int(FP), 'TN': int(TN)}
    
    with open(METADATA_FILE, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✅ Updated {METADATA_FILE}")
else:
    print("Metadata not updated.")