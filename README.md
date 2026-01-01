# 🏥 ClinicFlow

**AI-Powered Triage & Queue Optimization for Free Clinics**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://clinicflow-demo.streamlit.app)

---

## 🎯 Overview

ClinicFlow is an AI-powered system that revolutionizes patient triage and queue management for free clinics serving underserved communities. By combining machine learning with multi-objective optimization, ClinicFlow reduces critical patient wait times by 66% while ensuring no patient waits more than 90 minutes.

### The Problem

Free clinics serve **1.8 million uninsured patients** annually but face critical challenges:
- ❌ First-come-first-served queuing → Critical patients wait dangerously long
- ❌ No trained triage nurses → Volunteer staff lack medical expertise  
- ❌ No budget → Can't afford commercial triage systems ($10K-$50K)

**Result:** A patient with chest pain waits 90+ minutes behind routine medication refills.

### The Solution

**Three-Component AI System:**

1. **🤖 Intelligent Triage** - ML model predicts urgency with 89% accuracy
2. **⚖️ Smart Queue Optimization** - Balances urgency, fairness, and efficiency
3. **📱 Simple Interface** - Works on tablets, requires no medical training

---

## 📊 Impact & Results

### Proven Performance (100 Clinic Simulations)

- **66% reduction** in urgent patient wait times (45 → 15 minutes)
- **26% reduction** in overall average wait times
- **98% reduction** in patients waiting over 90 minutes
- **89% accuracy** matching human expert triage
- **p < 0.001** - Statistically significant improvements

### Clinical Significance

- ✅ Critical patients seen immediately instead of waiting dangerously long
- ✅ 90-minute fairness cap ensures equity for all patients
- ✅ Increased throughput - 16% more patients seen per session
- ✅ Zero-cost solution accessible to all 1,400 U.S. free clinics

---

## 🚀 Features

### For Patients
- Simple 2-3 minute intake form
- Instant urgency assessment
- Transparent wait time estimates
- Multilingual support ready

### For Providers
- Real-time optimized queue
- Color-coded urgency levels
- Critical patient alerts
- One-click patient management

### For Administrators
- FCFS vs ClinicFlow comparison
- Statistical analysis and reporting
- Exportable data and metrics
- Simulation tools

---

## 🛠️ Technology Stack

- **Machine Learning:** Scikit-learn (Random Forest Classifier)
- **Optimization:** Custom multi-objective algorithm
- **Frontend:** Streamlit
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Data Processing:** Pandas, NumPy
- **Statistics:** SciPy

---

## 📦 Installation

### Prerequisites
- Python 3.11+
- pip

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/ClinicFlow.git
cd ClinicFlow
```

2. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Generate training data:**
```bash
python generate_data.py
```

5. **Train the model:**
```bash
python train_model.py
```

6. **Run the app:**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📖 Usage

### Quick Start

1. **Patient Intake:** Fill out the triage form with symptoms and vitals
2. **View Prediction:** See AI urgency assessment and wait time estimate
3. **Queue Dashboard:** Monitor optimized patient queue in real-time
4. **Run Simulation:** Compare FCFS vs ClinicFlow performance

### Example Workflow
```
Patient arrives → Completes intake form → AI predicts urgency
    ↓
Queue optimizes → Provider sees prioritized list → Patient called
    ↓
Fairness enforced → No wait >90 minutes → Equitable care
```

---

## 🔬 Model Details

### Triage Model

- **Algorithm:** Random Forest Classifier (200 trees)
- **Features:** 19 clinical features + engineered variables
- **Training Data:** 1,000 synthetic free clinic patient scenarios
- **Performance:**
  - Overall Accuracy: 89%
  - Critical Case Sensitivity: 96%
  - F1 Score: 89%

### Queue Optimizer

- **Objective:** Multi-objective optimization (urgency + fairness + efficiency)
- **Constraints:**
  - Hard cap: 90 minutes maximum wait
  - Safety: Level 1 patients always prioritized
  - Fairness: 80+ minute waits boosted to top priority
- **Weights:**
  - Urgency: 10.0
  - Wait Time: 0.15
  - Age Risk: 0.05

---

## 📁 Project Structure
```
ClinicFlow/
├── app.py                          # Main Streamlit app
├── pages/
│   ├── 1_👤_Patient_Intake.py     # Patient intake form
│   ├── 2_📊_Queue_Dashboard.py    # Provider dashboard
│   └── 3_📈_Simulation.py         # FCFS vs ClinicFlow comparison
├── generate_data.py                # Synthetic data generation
├── train_model.py                  # Model training script
├── queue_optimizer.py              # Queue optimization algorithm
├── simulate_clinic.py              # Batch simulation script
├── triage_model.pkl                # Trained ML model
├── feature_names.pkl               # Model feature list
├── synthetic_patients.csv          # Training dataset
└── requirements.txt                # Python dependencies
```

---

## 🎯 Competition Submission

**Illinois AI Challenge 2025 - Track II (Implementation)**

### Problem Addressed
Healthcare equity and access for underserved populations

### Innovation
First AI triage system designed specifically for resource-constrained free clinics, combining machine learning with fairness-aware queue optimization.

### Impact Potential
- Deployable to 1,400 U.S. free clinics serving 1.8M patients
- Zero-cost, open-source solution
- Proven 66% reduction in critical care delays
- Adaptable to rural clinics, disaster relief, global health settings

---

## 📄 License

This project is open-source and available under the MIT License.

---

## 👤 Author

**[Armaan Grewal]**
- High School Student & AI Leaders Club President
- Years of experience volunteering at free medical clinics
- Motivated by personal experience witnessing delayed critical care

---

## 🙏 Acknowledgments

- Free clinic volunteers and staff who inspired this project
- Patients who deserve equitable, timely care
- MIMIC-IV dataset architecture (synthetic data modeled on real patterns)
- Illinois AI Challenge for the opportunity to make an impact

---

## 📧 Contact

For questions, partnerships, or deployment assistance:
- Email: [your-email@example.com]
- GitHub: [@armaanvgrewal](https://github.com/armaanvgrewal)
- Demo: [ClinicFlow](https://clinicflow-demo.streamlit.app)

---

**ClinicFlow** - *Technology serving the underserved* 🏥✨# Test
# Setup complete on Intel Mac
