"""
ClinicFlow Synthetic Patient Data Generator
Generates realistic free clinic patient data for ML training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

print("=" * 70)
print("CLINICFLOW SYNTHETIC DATA GENERATOR")
print("=" * 70)

# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_PATIENTS = 1000  # Number of patients to generate
OUTPUT_FILE = "synthetic_patients.csv"

# ============================================================================
# DEFINE MEDICAL KNOWLEDGE BASE
# ============================================================================

# Chief complaints by urgency level
COMPLAINTS = {
    1: [  # Life-threatening
        "Chest pain with difficulty breathing",
        "Severe difficulty breathing",
        "Unresponsive or confused",
        "Severe bleeding that won't stop",
        "Suspected stroke symptoms"
    ],
    2: [  # High risk
        "Chest pain",
        "Severe asthma attack",
        "High fever with severe headache",
        "Severe abdominal pain",
        "Head injury with loss of consciousness"
    ],
    3: [  # Moderate
        "Moderate abdominal pain",
        "Possible broken bone",
        "Deep cut requiring stitches",
        "High fever (over 103°F)",
        "Severe back pain"
    ],
    4: [  # Low risk
        "Ankle sprain",
        "Minor cut or scrape",
        "Cold or flu symptoms",
        "Mild rash",
        "Ear pain"
    ],
    5: [  # Non-urgent
        "Medication refill",
        "Routine check-up",
        "Minor skin irritation",
        "Mild headache",
        "Follow-up visit"
    ]
}

# Red flag symptoms that elevate urgency
RED_FLAGS = [
    "chest_pain",
    "difficulty_breathing", 
    "altered_mental_status",
    "severe_bleeding",
    "stroke_symptoms"
]

# Chronic conditions
CHRONIC_CONDITIONS = [
    "None",
    "Diabetes",
    "Hypertension",
    "Asthma",
    "Heart disease",
    "COPD",
    "Kidney disease"
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_demographics():
    """Generate patient demographics"""
    age = np.random.choice(
        [
            np.random.randint(18, 35),  # Young adults (30%)
            np.random.randint(35, 55),  # Middle age (40%)
            np.random.randint(55, 80)   # Older adults (30%)
        ],
        p=[0.3, 0.4, 0.3]
    )
    
    gender = np.random.choice(['Male', 'Female'], p=[0.48, 0.52])
    
    return age, gender

def generate_vitals(urgency_level, age):
    """Generate vital signs based on urgency and age"""
    
    # Normal ranges
    normal_hr = 70
    normal_sbp = 120
    normal_dbp = 80
    normal_temp = 98.6
    normal_spo2 = 98
    
    # Add age-related variations
    if age > 60:
        normal_hr += 5
        normal_sbp += 10
    
    # Adjust based on urgency
    if urgency_level == 1:
        # Critical vitals
        hr = normal_hr + np.random.randint(30, 50)
        sbp = normal_sbp + np.random.randint(-30, 40)
        dbp = normal_dbp + np.random.randint(-20, 30)
        temp = normal_temp + np.random.uniform(-1, 4)
        spo2 = normal_spo2 - np.random.randint(10, 20)
    elif urgency_level == 2:
        # Abnormal vitals
        hr = normal_hr + np.random.randint(15, 30)
        sbp = normal_sbp + np.random.randint(-20, 30)
        dbp = normal_dbp + np.random.randint(-15, 20)
        temp = normal_temp + np.random.uniform(-0.5, 3)
        spo2 = normal_spo2 - np.random.randint(5, 10)
    elif urgency_level == 3:
        # Slightly abnormal
        hr = normal_hr + np.random.randint(5, 20)
        sbp = normal_sbp + np.random.randint(-10, 20)
        dbp = normal_dbp + np.random.randint(-5, 15)
        temp = normal_temp + np.random.uniform(0, 2)
        spo2 = normal_spo2 - np.random.randint(0, 5)
    else:
        # Normal vitals
        hr = normal_hr + np.random.randint(-5, 10)
        sbp = normal_sbp + np.random.randint(-10, 10)
        dbp = normal_dbp + np.random.randint(-5, 10)
        temp = normal_temp + np.random.uniform(-0.5, 1)
        spo2 = normal_spo2 + np.random.randint(-1, 1)
    
    # Keep values in realistic ranges
    hr = max(40, min(180, hr))
    sbp = max(70, min(200, sbp))
    dbp = max(40, min(120, dbp))
    temp = max(95, min(105, round(temp, 1)))
    spo2 = max(70, min(100, spo2))
    
    return hr, sbp, dbp, temp, spo2

def generate_symptom_details(urgency_level):
    """Generate symptom severity, duration, and onset"""
    
    if urgency_level == 1:
        severity = np.random.randint(9, 11)
        duration_hours = np.random.uniform(0, 2)
        onset = "Sudden"
    elif urgency_level == 2:
        severity = np.random.randint(7, 9)
        duration_hours = np.random.uniform(1, 6)
        onset = np.random.choice(["Sudden", "Gradual"], p=[0.6, 0.4])
    elif urgency_level == 3:
        severity = np.random.randint(5, 8)
        duration_hours = np.random.uniform(3, 24)
        onset = np.random.choice(["Sudden", "Gradual"], p=[0.4, 0.6])
    else:
        severity = np.random.randint(2, 6)
        duration_hours = np.random.uniform(12, 72)
        onset = "Gradual"
    
    return severity, duration_hours, onset

def check_red_flags(complaint):
    """Check if complaint contains red flag symptoms"""
    complaint_lower = complaint.lower()
    flags = []
    
    if "chest pain" in complaint_lower or "chest" in complaint_lower:
        flags.append("chest_pain")
    if "breathing" in complaint_lower or "breathe" in complaint_lower:
        flags.append("difficulty_breathing")
    if "confused" in complaint_lower or "unresponsive" in complaint_lower:
        flags.append("altered_mental_status")
    if "bleeding" in complaint_lower:
        flags.append("severe_bleeding")
    if "stroke" in complaint_lower:
        flags.append("stroke_symptoms")
    
    return flags

# ============================================================================
# GENERATE PATIENTS
# ============================================================================

print(f"\n📊 Generating {NUM_PATIENTS} synthetic patients...")

patients = []

# Define urgency distribution (realistic for free clinics)
urgency_distribution = [
    (1, 0.05),   # 5% Level 1 (critical)
    (2, 0.15),   # 15% Level 2 (high risk)
    (3, 0.35),   # 35% Level 3 (moderate)
    (4, 0.30),   # 30% Level 4 (low risk)
    (5, 0.15)    # 15% Level 5 (non-urgent)
]

urgency_levels = []
urgency_probs = []
for level, prob in urgency_distribution:
    urgency_levels.append(level)
    urgency_probs.append(prob)

for i in range(NUM_PATIENTS):
    # Assign urgency level
    urgency = np.random.choice(urgency_levels, p=urgency_probs)
    
    # Generate demographics
    age, gender = generate_demographics()
    
    # Select chief complaint
    chief_complaint = random.choice(COMPLAINTS[urgency])
    
    # Check for red flags
    red_flags = check_red_flags(chief_complaint)
    has_red_flag = 1 if red_flags else 0
    
    # Generate vitals
    hr, sbp, dbp, temp, spo2 = generate_vitals(urgency, age)
    
    # Generate symptom details
    severity, duration_hours, onset = generate_symptom_details(urgency)
    
    # Generate medical history
    chronic_condition = np.random.choice(
        CHRONIC_CONDITIONS,
        p=[0.4, 0.15, 0.15, 0.10, 0.10, 0.05, 0.05]
    )
    
    # Previous visits
    previous_visits = np.random.poisson(2)
    
    # Wait time expectation (not used for ML, but for simulation)
    arrival_time = datetime.now() + timedelta(hours=np.random.uniform(0, 4))
    
    # Create patient record
    patient = {
        'patient_id': f'P{i+1:04d}',
        'age': age,
        'gender': gender,
        'chief_complaint': chief_complaint,
        'symptom_severity': severity,
        'symptom_duration_hours': round(duration_hours, 1),
        'symptom_onset': onset,
        'heart_rate': hr,
        'systolic_bp': sbp,
        'diastolic_bp': dbp,
        'temperature': temp,
        'oxygen_saturation': spo2,
        'has_red_flag': has_red_flag,
        'red_flag_type': ','.join(red_flags) if red_flags else 'None',
        'chronic_condition': chronic_condition,
        'previous_visits': previous_visits,
        'urgency_level': urgency,
        'arrival_time': arrival_time.strftime('%H:%M')
    }
    
    patients.append(patient)
    
    # Progress indicator
    if (i + 1) % 100 == 0:
        print(f"   Generated {i + 1}/{NUM_PATIENTS} patients...")

# ============================================================================
# CREATE DATAFRAME AND ENGINEER FEATURES
# ============================================================================

print("\n🔧 Engineering additional features...")

df = pd.DataFrame(patients)

# Feature engineering (creating derived features)

# 1. Age risk categories
df['age_risk'] = pd.cut(
    df['age'], 
    bins=[0, 35, 55, 100], 
    labels=['Low', 'Moderate', 'High']
)

# 2. Vital sign abnormalities
df['hr_abnormal'] = ((df['heart_rate'] < 60) | (df['heart_rate'] > 100)).astype(int)
df['bp_abnormal'] = ((df['systolic_bp'] < 90) | (df['systolic_bp'] > 140) | 
                      (df['diastolic_bp'] < 60) | (df['diastolic_bp'] > 90)).astype(int)
df['temp_abnormal'] = ((df['temperature'] < 97.0) | (df['temperature'] > 100.4)).astype(int)
df['spo2_abnormal'] = (df['oxygen_saturation'] < 95).astype(int)

# 3. Total vital abnormalities
df['vital_abnormalities'] = (df['hr_abnormal'] + df['bp_abnormal'] + 
                               df['temp_abnormal'] + df['spo2_abnormal'])

# 4. Symptom acuity score
df['symptom_acuity'] = df['symptom_severity'] * (1 if 'onset' == 'Sudden' else 0.7)
df['symptom_acuity'] = df.apply(
    lambda row: row['symptom_severity'] * (1.5 if row['symptom_onset'] == 'Sudden' else 1.0),
    axis=1
)

# 5. Duration category
df['duration_category'] = pd.cut(
    df['symptom_duration_hours'],
    bins=[0, 6, 24, float('inf')],
    labels=['Acute', 'Recent', 'Chronic']
)

# 6. Has chronic condition
df['has_chronic_condition'] = (df['chronic_condition'] != 'None').astype(int)

# 7. High risk chronic condition
high_risk_conditions = ['Heart disease', 'COPD', 'Kidney disease']
df['high_risk_chronic'] = df['chronic_condition'].isin(high_risk_conditions).astype(int)

# 8. Frequent visitor (previous visits > 3)
df['frequent_visitor'] = (df['previous_visits'] > 3).astype(int)

# 9. Risk score (composite)
df['risk_score'] = (
    df['symptom_severity'] * 0.3 +
    df['vital_abnormalities'] * 2 +
    df['has_red_flag'] * 5 +
    (df['age'] / 100) * 3 +
    df['has_chronic_condition'] * 1
)

# 10. Priority category (based on risk score)
df['priority_category'] = pd.cut(
    df['risk_score'],
    bins=[0, 5, 10, 15, float('inf')],
    labels=['Low', 'Medium', 'High', 'Critical']
)

print(f"   Engineered {len(df.columns) - len(patients[0])} additional features")

# ============================================================================
# SAVE DATA
# ============================================================================

print(f"\n💾 Saving data to {OUTPUT_FILE}...")
df.to_csv(OUTPUT_FILE, index=False)

# ============================================================================
# GENERATE STATISTICS
# ============================================================================

print("\n" + "=" * 70)
print("DATA GENERATION COMPLETE!")
print("=" * 70)

print(f"\n📊 Dataset Statistics:")
print(f"   Total patients: {len(df)}")
print(f"   Total features: {len(df.columns)}")
print(f"   Date range: {df['arrival_time'].min()} to {df['arrival_time'].max()}")

print(f"\n🚨 Urgency Level Distribution:")
for level in sorted(df['urgency_level'].unique()):
    count = len(df[df['urgency_level'] == level])
    pct = (count / len(df)) * 100
    print(f"   Level {level}: {count:3d} patients ({pct:5.1f}%)")

print(f"\n👥 Demographics:")
print(f"   Age range: {df['age'].min()}-{df['age'].max()} years")
print(f"   Average age: {df['age'].mean():.1f} years")
print(f"   Gender: {(df['gender']=='Male').sum()} Male, {(df['gender']=='Female').sum()} Female")

print(f"\n🔴 Red Flag Symptoms:")
print(f"   Patients with red flags: {df['has_red_flag'].sum()} ({(df['has_red_flag'].sum()/len(df))*100:.1f}%)")

print(f"\n💊 Chronic Conditions:")
chronic_counts = df['chronic_condition'].value_counts()
for condition, count in chronic_counts.head(5).items():
    print(f"   {condition}: {count}")

print(f"\n📈 Feature Summary:")
print(f"   Average symptom severity: {df['symptom_severity'].mean():.1f}/10")
print(f"   Average vital abnormalities: {df['vital_abnormalities'].mean():.1f}")
print(f"   Average risk score: {df['risk_score'].mean():.1f}")

print(f"\n✅ Sample of generated data:")
print(df[['patient_id', 'age', 'chief_complaint', 'urgency_level', 'risk_score']].head(10).to_string(index=False))

print("\n" + "=" * 70)
print(f"✅ Data saved to: {OUTPUT_FILE}")
print("✅ Ready for machine learning!")
print("=" * 70)