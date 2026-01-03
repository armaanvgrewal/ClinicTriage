"""
Extract 10K sample from MIMIC-IV-ED for ClinicFlow training
FULLY DEBUGGED VERSION
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

print("=" * 70)
print("MIMIC-IV-ED DATA EXTRACTION")
print("=" * 70)

# ============================================================================
# STEP 1: LOAD RAW DATA
# ============================================================================

print("\n📂 Loading MIMIC-IV-ED files...")

# Load ED stays
edstays = pd.read_csv('mimic_data/edstays.csv.gz', compression='gzip')
print(f"   ✅ Loaded {len(edstays):,} ED stays")

# Load triage data
triage = pd.read_csv('mimic_data/triage.csv.gz', compression='gzip')
print(f"   ✅ Loaded {len(triage):,} triage records")

# Load patients (for age calculation)
patients = pd.read_csv('mimic_data/patients.csv.gz', compression='gzip')
print(f"   ✅ Loaded {len(patients):,} patient records")

# ============================================================================
# STEP 2: MERGE TABLES
# ============================================================================

print("\n🔗 Merging tables...")

# Merge edstays with triage on BOTH stay_id AND subject_id
data = edstays.merge(triage, on=['stay_id', 'subject_id'], how='inner')
print(f"   After triage merge: {len(data):,} records")

# Merge with patients to get age
data = data.merge(patients[['subject_id', 'anchor_age', 'anchor_year', 'dod']], 
                  on='subject_id', how='left')
print(f"   After patients merge: {len(data):,} records")

# ============================================================================
# STEP 3: FILTER FOR COMPLETE RECORDS
# ============================================================================

print("\n🔍 Filtering for complete records...")

# Keep only records with acuity level (our target variable)
data = data[data['acuity'].notna()]
print(f"   After acuity filter: {len(data):,} records")

# Keep only records with essential vital signs
essential_vitals = ['temperature', 'heartrate', 'sbp', 'dbp', 'o2sat']
data = data.dropna(subset=essential_vitals)
print(f"   After vital signs filter: {len(data):,} records")

# Keep only valid acuity levels (1-5)
data = data[data['acuity'].isin([1, 2, 3, 4, 5])]
print(f"   After valid acuity filter: {len(data):,} records")

# Keep only records with age data
data = data[data['anchor_age'].notna()]
print(f"   After age filter: {len(data):,} records")

# ============================================================================
# STEP 4: STRATIFIED SAMPLING (10K records)
# ============================================================================

print("\n📊 Creating stratified 10K sample...")

# Sample 10,000 records, maintaining acuity distribution
sample_size = 10000

if len(data) < sample_size:
    print(f"   ⚠️  Only {len(data):,} complete records available")
    sample_size = len(data)
    data_sample = data.copy()
else:
    # Stratified sampling
    data_sample, _ = train_test_split(
        data,
        train_size=sample_size,
        stratify=data['acuity'],
        random_state=42
    )

# Reset index to avoid issues
data_sample = data_sample.reset_index(drop=True)

print(f"   ✅ Sampled {len(data_sample):,} records")

# Show acuity distribution
print(f"\n   Acuity distribution in sample:")
acuity_dist = data_sample['acuity'].value_counts().sort_index()
for level, count in acuity_dist.items():
    pct = (count / len(data_sample)) * 100
    print(f"      Level {int(level)}: {count:4d} ({pct:5.1f}%)")

# ============================================================================
# STEP 5: ENSURE NUMERIC TYPES
# ============================================================================

print("\n🔧 Ensuring numeric types...")

# Convert all vital signs to numeric (force any strings to NaN)
numeric_columns = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 
                   'pain', 'acuity', 'anchor_age', 'anchor_year']

for col in numeric_columns:
    if col in data_sample.columns:
        data_sample[col] = pd.to_numeric(data_sample[col], errors='coerce')

# Drop any rows that became NaN after conversion
data_sample = data_sample.dropna(subset=['acuity', 'temperature', 'heartrate', 
                                          'sbp', 'dbp', 'o2sat', 'anchor_age'])

print(f"   After type checking: {len(data_sample):,} records")

# ============================================================================
# STEP 6: FEATURE ENGINEERING - AGE
# ============================================================================

print("\n🎂 Engineering age feature...")

# Parse intime to get admission year
data_sample['intime'] = pd.to_datetime(data_sample['intime'], errors='coerce')
data_sample['admission_year'] = data_sample['intime'].dt.year

# Calculate age at admission
data_sample['age'] = data_sample['anchor_age'] + (data_sample['admission_year'] - data_sample['anchor_year'])

# Handle ages outside reasonable range (18-100 for adults)
data_sample['age'] = data_sample['age'].clip(18, 100)

print(f"   Age range: {data_sample['age'].min():.0f} - {data_sample['age'].max():.0f} years")
print(f"   Average age: {data_sample['age'].mean():.1f} years")

# ============================================================================
# STEP 7: FEATURE ENGINEERING - OTHER FEATURES
# ============================================================================

print("\n🔧 Engineering other features...")

# Map gender (gender column has 'M' or 'F')
data_sample['gender_encoded'] = (data_sample['gender'] == 'F').astype(int)

# Use pain score as symptom severity (0-10 scale)
# Fill missing pain values with median (5)
data_sample['symptom_severity'] = data_sample['pain'].fillna(5.0).astype(float)

# Calculate vital sign abnormalities
data_sample['hr_abnormal'] = ((data_sample['heartrate'] < 60) | 
                               (data_sample['heartrate'] > 100)).astype(int)
data_sample['bp_abnormal'] = ((data_sample['sbp'] < 90) | (data_sample['sbp'] > 140) |
                               (data_sample['dbp'] < 60) | (data_sample['dbp'] > 90)).astype(int)
data_sample['temp_abnormal'] = ((data_sample['temperature'] < 97.0) | 
                                 (data_sample['temperature'] > 100.4)).astype(int)
data_sample['spo2_abnormal'] = (data_sample['o2sat'] < 95).astype(int)

data_sample['vital_abnormalities'] = (data_sample['hr_abnormal'] + 
                                       data_sample['bp_abnormal'] +
                                       data_sample['temp_abnormal'] + 
                                       data_sample['spo2_abnormal'])

# Check for red flag symptoms in chief complaint
red_flag_keywords = ['chest pain', 'difficulty breathing', 'sob', 'chest pressure',
                     'stroke', 'bleeding', 'unresponsive', 'altered', 'seizure',
                     'unconscious', 'syncope', 'cardiac', 'mi', 'heart attack']

def check_red_flags(complaint):
    if pd.isna(complaint):
        return 0
    complaint_lower = str(complaint).lower()
    return int(any(keyword in complaint_lower for keyword in red_flag_keywords))

data_sample['has_red_flag'] = data_sample['chiefcomplaint'].apply(check_red_flags)

print(f"   Red flag cases: {data_sample['has_red_flag'].sum()} ({data_sample['has_red_flag'].mean()*100:.1f}%)")

# Estimate symptom acuity (combine pain with acuity level)
# FIXED: Ensure both are numeric and handle properly
data_sample['symptom_acuity'] = (
    data_sample['symptom_severity'].astype(float) * 
    (6.0 - data_sample['acuity'].astype(float)) / 5.0
)

# Set default values for fields not easily extractable from MIMIC-IV-ED
data_sample['symptom_duration_hours'] = 6.0  # Reasonable default for ED
data_sample['onset_encoded'] = 1  # Assume sudden onset for ED visits
data_sample['has_chronic_condition'] = 0  # Would need detailed diagnosis analysis
data_sample['high_risk_chronic'] = 0
data_sample['previous_visits'] = 1  # Default

# ============================================================================
# STEP 8: SELECT FINAL FEATURES
# ============================================================================

print("\n✂️  Selecting features for model...")

# Create final dataframe with ClinicFlow feature names
final_data = pd.DataFrame({
    'patient_id': data_sample['stay_id'].astype(str),
    'age': data_sample['age'].astype(float),
    'gender_encoded': data_sample['gender_encoded'].astype(int),
    'chief_complaint': data_sample['chiefcomplaint'].fillna('Unknown').astype(str),
    'symptom_severity': data_sample['symptom_severity'].astype(float),
    'symptom_duration_hours': data_sample['symptom_duration_hours'].astype(float),
    'onset_encoded': data_sample['onset_encoded'].astype(int),
    'heart_rate': data_sample['heartrate'].astype(float),
    'systolic_bp': data_sample['sbp'].astype(float),
    'diastolic_bp': data_sample['dbp'].astype(float),
    'temperature': data_sample['temperature'].astype(float),
    'oxygen_saturation': data_sample['o2sat'].astype(float),
    'has_red_flag': data_sample['has_red_flag'].astype(int),
    'has_chronic_condition': data_sample['has_chronic_condition'].astype(int),
    'high_risk_chronic': data_sample['high_risk_chronic'].astype(int),
    'hr_abnormal': data_sample['hr_abnormal'].astype(int),
    'bp_abnormal': data_sample['bp_abnormal'].astype(int),
    'temp_abnormal': data_sample['temp_abnormal'].astype(int),
    'spo2_abnormal': data_sample['spo2_abnormal'].astype(int),
    'vital_abnormalities': data_sample['vital_abnormalities'].astype(int),
    'symptom_acuity': data_sample['symptom_acuity'].astype(float),
    'previous_visits': data_sample['previous_visits'].astype(int),
    'urgency_level': data_sample['acuity'].astype(int)
})

# ============================================================================
# STEP 9: DATA QUALITY CHECKS
# ============================================================================

print("\n✅ Quality checks...")

print(f"   Records: {len(final_data):,}")
print(f"   Features: {len(final_data.columns)}")
print(f"   Missing values: {final_data.isnull().sum().sum()}")

# Check value ranges
print(f"\n   Value ranges:")
print(f"   Age: {final_data['age'].min():.0f} - {final_data['age'].max():.0f} years")
print(f"   Heart rate: {final_data['heart_rate'].min():.0f} - {final_data['heart_rate'].max():.0f} bpm")
print(f"   Systolic BP: {final_data['systolic_bp'].min():.0f} - {final_data['systolic_bp'].max():.0f} mmHg")
print(f"   Temperature: {final_data['temperature'].min():.1f} - {final_data['temperature'].max():.1f} °F")
print(f"   O2 Sat: {final_data['oxygen_saturation'].min():.0f} - {final_data['oxygen_saturation'].max():.0f} %")

# Clean extreme outliers
issues = []
if (final_data['heart_rate'] < 30).any() or (final_data['heart_rate'] > 200).any():
    issues.append("Heart rate")
    final_data['heart_rate'] = final_data['heart_rate'].clip(30, 200)
    
if (final_data['systolic_bp'] < 60).any() or (final_data['systolic_bp'] > 250).any():
    issues.append("Blood pressure")
    final_data['systolic_bp'] = final_data['systolic_bp'].clip(60, 250)
    final_data['diastolic_bp'] = final_data['diastolic_bp'].clip(30, 150)
    
if (final_data['temperature'] < 90).any() or (final_data['temperature'] > 110).any():
    issues.append("Temperature")
    final_data['temperature'] = final_data['temperature'].clip(90, 110)
    
if (final_data['oxygen_saturation'] < 70).any() or (final_data['oxygen_saturation'] > 100).any():
    issues.append("O2 saturation")
    final_data['oxygen_saturation'] = final_data['oxygen_saturation'].clip(70, 100)

if issues:
    print(f"   🔧 Cleaned outliers in: {', '.join(issues)}")
else:
    print(f"   ✅ All values in valid ranges")

# ============================================================================
# STEP 10: SAVE PROCESSED DATA
# ============================================================================

print("\n💾 Saving processed data...")

output_file = 'mimic_patients_10k.csv'
final_data.to_csv(output_file, index=False)
print(f"   ✅ Saved {len(final_data):,} records to '{output_file}'")

# Save summary statistics
print(f"\n📊 Summary Statistics:")
print(f"   Age: {final_data['age'].mean():.1f} ± {final_data['age'].std():.1f} years")
print(f"   Heart Rate: {final_data['heart_rate'].mean():.1f} ± {final_data['heart_rate'].std():.1f} bpm")
print(f"   Systolic BP: {final_data['systolic_bp'].mean():.1f} ± {final_data['systolic_bp'].std():.1f} mmHg")
print(f"   Temperature: {final_data['temperature'].mean():.1f} ± {final_data['temperature'].std():.1f} °F")
print(f"   Oxygen Sat: {final_data['oxygen_saturation'].mean():.1f} ± {final_data['oxygen_saturation'].std():.1f} %")

print(f"\n   Acuity distribution:")
for level in [1, 2, 3, 4, 5]:
    count = (final_data['urgency_level'] == level).sum()
    pct = count / len(final_data) * 100
    print(f"      Level {level}: {count:4d} ({pct:5.1f}%)")

print("\n" + "=" * 70)
print("DATA EXTRACTION COMPLETE!")
print("=" * 70)

print(f"\n✅ Next steps:")
print(f"   1. Review 'mimic_patients_10k.csv'")
print(f"   2. Train model with: python train_model_mimic.py")
print(f"   3. Validate results")