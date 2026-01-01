"""
Inspect and validate synthetic patient data
"""

import pandas as pd
import matplotlib.pyplot as plt

print("=" * 70)
print("CLINICFLOW DATA INSPECTOR")
print("=" * 70)

# Load the data
df = pd.read_csv('synthetic_patients.csv')

print(f"\n📊 Dataset Overview:")
print(f"   Rows: {len(df)}")
print(f"   Columns: {len(df.columns)}")
print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

print(f"\n📋 Column Names:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. {col}")

print(f"\n🔍 Data Types:")
print(df.dtypes.value_counts())

print(f"\n❓ Missing Values:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("   ✅ No missing values!")
else:
    print(missing[missing > 0])

print(f"\n📈 Urgency Level Statistics:")
print(df['urgency_level'].value_counts().sort_index())

print(f"\n🎯 Key Feature Statistics:")
numeric_cols = ['age', 'symptom_severity', 'heart_rate', 'systolic_bp', 
                'temperature', 'oxygen_saturation', 'risk_score']
print(df[numeric_cols].describe().round(2))

print(f"\n🔴 Red Flag Analysis:")
print(f"   Total with red flags: {df['has_red_flag'].sum()}")
print(f"   Percentage: {(df['has_red_flag'].sum()/len(df)*100):.1f}%")
print(f"\n   Red flag types:")
red_flag_types = df[df['has_red_flag']==1]['red_flag_type'].value_counts()
for flag_type, count in red_flag_types.head(10).items():
    print(f"      {flag_type}: {count}")

print(f"\n✅ Data Quality Checks:")

# Check 1: Urgency levels are valid
valid_urgency = df['urgency_level'].isin([1, 2, 3, 4, 5]).all()
print(f"   Urgency levels valid (1-5): {'✅ Pass' if valid_urgency else '❌ Fail'}")

# Check 2: Vitals are in realistic ranges
hr_valid = ((df['heart_rate'] >= 40) & (df['heart_rate'] <= 180)).all()
print(f"   Heart rate in range (40-180): {'✅ Pass' if hr_valid else '❌ Fail'}")

bp_valid = ((df['systolic_bp'] >= 70) & (df['systolic_bp'] <= 200)).all()
print(f"   Blood pressure in range: {'✅ Pass' if bp_valid else '❌ Fail'}")

temp_valid = ((df['temperature'] >= 95) & (df['temperature'] <= 105)).all()
print(f"   Temperature in range (95-105): {'✅ Pass' if temp_valid else '❌ Fail'}")

spo2_valid = ((df['oxygen_saturation'] >= 70) & (df['oxygen_saturation'] <= 100)).all()
print(f"   Oxygen saturation in range: {'✅ Pass' if spo2_valid else '❌ Fail'}")

# Check 3: Higher urgency correlates with higher risk
correlation = df['urgency_level'].corr(df['risk_score'])
print(f"   Urgency-risk correlation: {correlation:.3f} {'✅ Strong' if abs(correlation) > 0.6 else '⚠️  Weak'}")

print("\n" + "=" * 70)
print("✅ DATA INSPECTION COMPLETE!")
print("=" * 70)

# Optional: Create visualizations
print("\n📊 Creating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Urgency distribution
urgency_counts = df['urgency_level'].value_counts().sort_index()
axes[0, 0].bar(urgency_counts.index, urgency_counts.values, color='steelblue')
axes[0, 0].set_xlabel('Urgency Level')
axes[0, 0].set_ylabel('Number of Patients')
axes[0, 0].set_title('Distribution of Urgency Levels')
axes[0, 0].grid(axis='y', alpha=0.3)

# Plot 2: Age distribution by urgency
for urgency in [1, 2, 3, 4, 5]:
    ages = df[df['urgency_level'] == urgency]['age']
    axes[0, 1].hist(ages, alpha=0.5, label=f'Level {urgency}', bins=15)
axes[0, 1].set_xlabel('Age')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Age Distribution by Urgency')
axes[0, 1].legend()

# Plot 3: Risk score distribution
axes[1, 0].hist(df['risk_score'], bins=30, color='coral', edgecolor='black')
axes[1, 0].set_xlabel('Risk Score')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Risk Score Distribution')
axes[1, 0].grid(axis='y', alpha=0.3)

# Plot 4: Vital abnormalities by urgency
vital_by_urgency = df.groupby('urgency_level')['vital_abnormalities'].mean()
axes[1, 1].bar(vital_by_urgency.index, vital_by_urgency.values, color='lightcoral')
axes[1, 1].set_xlabel('Urgency Level')
axes[1, 1].set_ylabel('Average Vital Abnormalities')
axes[1, 1].set_title('Vital Abnormalities by Urgency')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('data_analysis.png', dpi=150, bbox_inches='tight')
print("   ✅ Visualization saved as 'data_analysis.png'")

print("\n💡 Next steps:")
print("   1. Review the data in synthetic_patients.csv")
print("   2. Look at data_analysis.png for visualizations")
print("   3. Ready to move to Phase 3: Model Training!")