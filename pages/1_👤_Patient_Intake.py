"""
ClinicFlow - Patient Intake Page
Collects patient information and provides AI triage prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pickle

st.set_page_config(page_title="Patient Intake - ClinicFlow", page_icon="👤", layout="wide")

# ============================================================================
# LOAD MODELS
# ============================================================================

@st.cache_resource
def load_models():
    with open('triage_model.pkl', 'rb') as f:
        triage_model = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return triage_model, feature_names

triage_model, feature_names = load_models()

# ============================================================================
# HEADER
# ============================================================================

st.title("👤 Patient Intake & Triage")
st.markdown("Complete the form below to receive your urgency level and estimated wait time.")

st.markdown("---")

# ============================================================================
# PATIENT INTAKE FORM
# ============================================================================

with st.form("patient_intake_form"):
    st.markdown("### 📋 Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        patient_age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=45,
            help="Patient's age in years"
        )
        
        patient_gender = st.selectbox(
            "Gender",
            options=["Male", "Female"],
            help="Biological sex"
        )
    
    with col2:
        chief_complaint = st.text_input(
            "Chief Complaint",
            placeholder="e.g., Chest pain, Fever, Ankle injury",
            help="Main reason for visit"
        )
    
    st.markdown("---")
    st.markdown("### 🤒 Symptoms")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        symptom_severity = st.slider(
            "Symptom Severity",
            min_value=1,
            max_value=10,
            value=5,
            help="1 = Mild discomfort, 10 = Worst pain imaginable"
        )
    
    with col2:
        symptom_duration = st.number_input(
            "Duration (hours)",
            min_value=0.0,
            max_value=168.0,
            value=2.0,
            step=0.5,
            help="How long have you had symptoms?"
        )
    
    with col3:
        symptom_onset = st.selectbox(
            "Onset",
            options=["Sudden", "Gradual"],
            help="Did symptoms start suddenly or develop gradually?"
        )
    
    st.markdown("---")
    st.markdown("### 🩺 Vital Signs (if available)")
    st.caption("Leave blank if not measured")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        heart_rate = st.number_input(
            "Heart Rate (bpm)",
            min_value=40,
            max_value=200,
            value=75,
            help="Normal: 60-100 bpm"
        )
    
    with col2:
        systolic_bp = st.number_input(
            "Systolic BP",
            min_value=70,
            max_value=200,
            value=120,
            help="Top number, normal: 90-120"
        )
    
    with col3:
        diastolic_bp = st.number_input(
            "Diastolic BP",
            min_value=40,
            max_value=130,
            value=80,
            help="Bottom number, normal: 60-80"
        )
    
    with col4:
        temperature = st.number_input(
            "Temperature (°F)",
            min_value=95.0,
            max_value=106.0,
            value=98.6,
            step=0.1,
            help="Normal: 97-99°F"
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        oxygen_saturation = st.number_input(
            "Oxygen Saturation (%)",
            min_value=70,
            max_value=100,
            value=98,
            help="Normal: 95-100%"
        )
    
    st.markdown("---")
    st.markdown("### 🏥 Medical History")
    
    col1, col2 = st.columns(2)
    
    with col1:
        chronic_condition = st.selectbox(
            "Chronic Conditions",
            options=["None", "Diabetes", "Hypertension", "Asthma", 
                    "Heart disease", "COPD", "Kidney disease"],
            help="Select primary chronic condition"
        )
    
    with col2:
        previous_visits = st.number_input(
            "Previous Clinic Visits",
            min_value=0,
            max_value=50,
            value=1,
            help="Number of previous visits to this clinic"
        )
    
    st.markdown("---")
    st.markdown("### 🚨 Red Flag Symptoms")
    st.caption("Check any that apply:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        chest_pain = st.checkbox("Chest pain")
        difficulty_breathing = st.checkbox("Difficulty breathing")
    
    with col2:
        altered_mental = st.checkbox("Confusion / Altered mental status")
        severe_bleeding = st.checkbox("Severe bleeding")
    
    with col3:
        stroke_symptoms = st.checkbox("Stroke symptoms (FAST)")
    
    # Submit button
    st.markdown("---")
    submitted = st.form_submit_button("🔍 Get Triage Assessment", use_container_width=True)

# ============================================================================
# PROCESS FORM SUBMISSION
# ============================================================================

if submitted:
    # Check red flags
    red_flags = []
    if chest_pain:
        red_flags.append("chest_pain")
    if difficulty_breathing:
        red_flags.append("difficulty_breathing")
    if altered_mental:
        red_flags.append("altered_mental_status")
    if severe_bleeding:
        red_flags.append("severe_bleeding")
    if stroke_symptoms:
        red_flags.append("stroke_symptoms")
    
    has_red_flag = 1 if red_flags else 0
    
    # Calculate engineered features
    gender_encoded = 1 if patient_gender == "Female" else 0
    onset_encoded = 1 if symptom_onset == "Sudden" else 0
    
    hr_abnormal = 1 if (heart_rate < 60 or heart_rate > 100) else 0
    bp_abnormal = 1 if (systolic_bp < 90 or systolic_bp > 140 or 
                       diastolic_bp < 60 or diastolic_bp > 90) else 0
    temp_abnormal = 1 if (temperature < 97.0 or temperature > 100.4) else 0
    spo2_abnormal = 1 if oxygen_saturation < 95 else 0
    
    vital_abnormalities = hr_abnormal + bp_abnormal + temp_abnormal + spo2_abnormal
    
    symptom_acuity = symptom_severity * (1.5 if symptom_onset == "Sudden" else 1.0)
    
    has_chronic_condition = 0 if chronic_condition == "None" else 1
    high_risk_chronic = 1 if chronic_condition in ["Heart disease", "COPD", "Kidney disease"] else 0
    
    # Create feature dictionary
    patient_features = {
        'age': patient_age,
        'symptom_severity': symptom_severity,
        'symptom_duration_hours': symptom_duration,
        'heart_rate': heart_rate,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'temperature': temperature,
        'oxygen_saturation': oxygen_saturation,
        'has_red_flag': has_red_flag,
        'has_chronic_condition': has_chronic_condition,
        'high_risk_chronic': high_risk_chronic,
        'hr_abnormal': hr_abnormal,
        'bp_abnormal': bp_abnormal,
        'temp_abnormal': temp_abnormal,
        'spo2_abnormal': spo2_abnormal,
        'vital_abnormalities': vital_abnormalities,
        'symptom_acuity': symptom_acuity,
        'previous_visits': previous_visits,
        'gender_encoded': gender_encoded,
        'onset_encoded': onset_encoded
    }
    
    # Extract features in correct order
    features_array = [patient_features[feat] for feat in feature_names]
    
    # Make prediction
    urgency_prediction = triage_model.predict([features_array])[0]
    probabilities = triage_model.predict_proba([features_array])[0]
    confidence = probabilities[urgency_prediction - 1]
    
    # Display results
    st.markdown("---")
    st.markdown("## 🎯 Triage Assessment Results")
    
    # Urgency level display
    urgency_colors = {
        1: ("🔴", "CRITICAL - Life Threatening", "#ff4444"),
        2: ("🟠", "HIGH RISK - Urgent Care Needed", "#ff8800"),
        3: ("🟡", "MODERATE - Timely Care Needed", "#ffbb00"),
        4: ("🟢", "LOW RISK - Non-urgent", "#88cc88"),
        5: ("🟢", "MINIMAL - Routine Care", "#66bb66")
    }
    
    icon, label, color = urgency_colors[urgency_prediction]
    
    st.markdown(f"""
    <div style='background-color: {color}; padding: 2rem; border-radius: 1rem; text-align: center;'>
        <h1 style='color: white; margin: 0;'>{icon} URGENCY LEVEL {urgency_prediction}</h1>
        <h2 style='color: white; margin: 0.5rem 0;'>{label}</h2>
        <p style='color: white; font-size: 1.2rem;'>Confidence: {confidence:.0%}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Details
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Estimated Wait Time",
            f"{(urgency_prediction - 1) * 20} minutes",
            help="Based on current queue and urgency level"
        )
    
    with col2:
        st.metric(
            "Queue Position",
            f"#{6 - urgency_prediction}",
            help="Approximate position (updates as patients arrive)"
        )
    
    with col3:
        st.metric(
            "Vital Abnormalities",
            f"{vital_abnormalities} / 4",
            help="Number of abnormal vital signs"
        )
    
    # Clinical reasoning
    st.markdown("---")
    st.markdown("### 🔍 Clinical Reasoning")
    
    reasoning_points = []
    
    if has_red_flag:
        reasoning_points.append(f"⚠️ **Red flag symptoms detected:** {', '.join(red_flags)}")
    
    if vital_abnormalities > 0:
        reasoning_points.append(f"📊 **{vital_abnormalities} abnormal vital signs** requiring attention")
    
    if symptom_severity >= 8:
        reasoning_points.append(f"🤒 **High symptom severity** (rated {symptom_severity}/10)")
    
    if patient_age >= 65:
        reasoning_points.append(f"👴 **Age risk factor** ({patient_age} years old)")
    
    if symptom_onset == "Sudden":
        reasoning_points.append(f"⚡ **Sudden onset** increases urgency")
    
    if high_risk_chronic:
        reasoning_points.append(f"🏥 **High-risk chronic condition:** {chronic_condition}")
    
    if reasoning_points:
        for point in reasoning_points:
            st.markdown(point)
    else:
        st.success("✅ No critical factors identified - routine care appropriate")
    
    # Probability breakdown
    st.markdown("---")
    st.markdown("### 📊 Probability Distribution")
    
    prob_df = pd.DataFrame({
        'Urgency Level': ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'],
        'Probability': probabilities
    })
    
    st.bar_chart(prob_df.set_index('Urgency Level'))
    
    # Add to queue button
    st.markdown("---")
    if st.button("➕ Add to Queue", use_container_width=True):
        # Create patient record
        patient_record = {
            'patient_id': f"P{st.session_state.patient_counter:04d}",
            'name': f"Patient {st.session_state.patient_counter}",
            'age': patient_age,
            'gender': patient_gender,
            'chief_complaint': chief_complaint if chief_complaint else "Not specified",
            'urgency_level': int(urgency_prediction),
            'arrival_time': datetime.now(),
            **patient_features
        }
        
        # Add to session state queue
        st.session_state.queue.append(patient_record)
        st.session_state.patient_counter += 1
        
        st.success(f"✅ Patient added to queue! ID: {patient_record['patient_id']}")
        st.info("👉 Go to **Queue Dashboard** to see optimized patient order")

# ============================================================================
# INFORMATION SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### ℹ️ Urgency Level Guide")
    
    st.markdown("""
    **Level 1 - Critical**
    - Life-threatening condition
    - Immediate care required
    - Examples: Heart attack, stroke, severe bleeding
    
    **Level 2 - High Risk**
    - Potentially dangerous
    - Urgent care needed
    - Examples: Chest pain, severe asthma
    
    **Level 3 - Moderate**
    - Needs timely care
    - Not immediately dangerous
    - Examples: Possible fracture, high fever
    
    **Level 4 - Low Risk**
    - Minor condition
    - Can wait safely
    - Examples: Sprain, cold symptoms
    
    **Level 5 - Minimal**
    - Routine care
    - Non-urgent
    - Examples: Medication refill, follow-up
    """)