#!/usr/bin/env python3
"""
ClinicFlow Setup Test
Tests that all required packages are installed and working
"""

import sys
print("=" * 60)
print("CLINICFLOW SETUP TEST")
print("=" * 60)

# Test Python version
print(f"\n✅ Python version: {sys.version}")
if sys.version_info < (3, 11):
    print("⚠️  Warning: Python 3.11+ recommended")
else:
    print("✅ Python version is good!")

# Test imports
print("\n📦 Testing package imports...")

try:
    import numpy as np
    print(f"✅ NumPy {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy failed: {e}")
    sys.exit(1)

try:
    import pandas as pd
    print(f"✅ Pandas {pd.__version__}")
except ImportError as e:
    print(f"❌ Pandas failed: {e}")
    sys.exit(1)

try:
    import sklearn
    print(f"✅ Scikit-learn {sklearn.__version__}")
except ImportError as e:
    print(f"❌ Scikit-learn failed: {e}")
    sys.exit(1)

try:
    import streamlit as st
    print(f"✅ Streamlit {st.__version__}")
except ImportError as e:
    print(f"❌ Streamlit failed: {e}")
    sys.exit(1)

try:
    import plotly
    print(f"✅ Plotly {plotly.__version__}")
except ImportError as e:
    print(f"❌ Plotly failed: {e}")
    sys.exit(1)

try:
    import matplotlib
    print(f"✅ Matplotlib {matplotlib.__version__}")
except ImportError as e:
    print(f"❌ Matplotlib failed: {e}")
    sys.exit(1)

try:
    import seaborn as sns
    print(f"✅ Seaborn {sns.__version__}")
except ImportError as e:
    print(f"❌ Seaborn failed: {e}")
    sys.exit(1)

# Test basic functionality
print("\n🧪 Testing basic functionality...")

# Test DataFrame creation
data = pd.DataFrame({
    'patient_id': [1, 2, 3, 4, 5],
    'age': [25, 45, 67, 34, 52],
    'urgency': [3, 5, 1, 4, 2]
})
print(f"✅ Created test DataFrame with {len(data)} rows")

# Test machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X = data[['age']]
y = data['urgency']

# Simple train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a simple model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

print(f"✅ Trained Random Forest classifier")
print(f"✅ Model test accuracy: {accuracy:.1%}")

# Test plotting (don't display, just create)
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 2])
print("✅ Created test matplotlib plot")
plt.close()

# Success message
print("\n" + "=" * 60)
print("🎉 ALL TESTS PASSED!")
print("=" * 60)
print("\n✅ Your Mac is ready for ClinicFlow development!")
print("✅ All packages installed correctly")
print("✅ Machine learning working")
print("✅ Data processing working")
print("✅ Visualization working")
print("\n📋 Next steps:")
print("   1. Reply to Claude that Phase 1 is complete")
print("   2. Start Phase 2: Data Generation")
print("\n" + "=" * 60)