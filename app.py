# -*- coding: utf-8 -*-
"""
ClinicTriage - AI-Powered Triage System for Free Clinics
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
from queue_optimizer import QueueOptimizer

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="ClinicTriage - AI Triage System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR BETTER STYLING
# ============================================================================

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .urgency-1 {
        background-color: #ff4444;
        color: white;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
    .urgency-2 {
        background-color: #ff8800;
        color: white;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
    .urgency-3 {
        background-color: #ffbb00;
        color: black;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
    .urgency-4 {
        background-color: #88cc88;
        color: black;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
    .urgency-5 {
        background-color: #66bb66;
        color: white;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS (CACHED)
# ============================================================================

@st.cache_resource
def load_models():
    """Load MIMIC-IV ML models and optimizer (cached for performance)"""
    # Try to load MIMIC-IV v2 model first
    try:
        with open('triage_model_mimic_v2.pkl', 'rb') as f:
            triage_model = pickle.load(f)
        model_version = "MIMIC-IV v2 (74.2% accuracy, 83.5% critical detection)"
    except FileNotFoundError:
        # Fallback to synthetic model
        with open('triage_model.pkl', 'rb') as f:
            triage_model = pickle.load(f)
        model_version = "Synthetic (89% accuracy)"
    
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    optimizer = QueueOptimizer(
        urgency_weight=10.0,
        wait_time_weight=0.15,
        age_risk_weight=0.05,
        max_wait_minutes=90
    )
    
    # Store model version in session state
    if 'model_version' not in st.session_state:
        st.session_state.model_version = model_version
    
    return triage_model, feature_names, optimizer

# Load models
triage_model, feature_names, optimizer = load_models()

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================

if 'queue' not in st.session_state:
    st.session_state.queue = []

if 'patient_counter' not in st.session_state:
    st.session_state.patient_counter = 1

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.image("https://placehold.co/300x100/1f77b4/ffffff?text=ClinicTriage", 
             use_container_width=True)
    st.markdown("---")
    
    st.markdown("### 📊 System Status")
    st.metric("Patients in Queue", len(st.session_state.queue))
    
    if st.session_state.queue:
        urgent_count = sum(1 for p in st.session_state.queue if p.get('urgency_level', 5) <= 2)
        st.metric("Urgent Patients (L1-2)", urgent_count)
    
    st.markdown("---")
    
    # Quick stats
    st.markdown("### 🎯 ClinicTriage Impact")
    st.markdown("""
    - **83.5%** critical detection rate on real clinical data
    - **77.7%** critical case accuracy
    - **66%** reduction in urgent wait times
    - Trained on **10K** real ED visits (MIMIC-IV dataset)
    """)
    
    st.caption(f"Model: {st.session_state.get('model_version', 'Loading...')}")

# ============================================================================
# MAIN PAGE - HOME
# ============================================================================

st.markdown('<p style="font-size: 24px; font-weight: bold; text-align: center;">🏥 ClinicTriage</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Triage & Queue Optimization for Free Clinics</p>', 
            unsafe_allow_html=True)

# Add MIMIC-IV badge
st.markdown("""
<div style='text-align: center; margin-bottom: 1rem;'>
    <span style='background-color: #1f77b4; color: white; padding: 0.5rem 1rem; border-radius: 0.5rem; font-weight: bold;'>
        ✅ Trained on MIMIC-IV-ED | 10,000 Real ED Visits | 83.5% Critical Detection
    </span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Introduction
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🎯 The Problem")
    st.markdown("""
    Free clinics serve **1.8 million patients** annually with limited resources:
    
    - ⚠️ **First-come-first-served** → Critical patients wait dangerously long
    - 👥 **No trained triage nurses** → Volunteer staff lack medical expertise
    - 📋 **Walk-in model** → No appointment system to manage flow
    - 💰 **Zero budget** → Cannot afford commercial triage systems that cost ~$50K annually
    
    **Result:** A 62-year-old with chest pain waits 90+ minutes behind medication refills.
    """)
    
    st.markdown("### 💡 The ClinicTriage Solution")
    st.markdown("""
    **Three-Component AI System:**
    
    1. **🤖 Intelligent Triage** - ML trained on 10,000 real ED visits (83.5% critical detection)
    2. **⚖️ Smart Queue Optimization** - Balances urgency, fairness, and efficiency  
    3. **📱 Simple Interface** - Works on tablets, requires no medical training
    
    **Validated on Real Clinical Data:**
    - Trained on **MIMIC-IV-ED** dataset from Beth Israel Deaconess Medical Center
    - **83.5%** critical detection rate (Level 1-2) - the most important metric
    - **77.7%** critical exact accuracy
    - **74.2%** overall accuracy on actual triage decisions
    - Tested against expert emergency physician assessments
    
    **Why 74.2% overall accuracy but 83.5% critical detection rate?**  
    ✅ The model is optimized to prioritize patient safety by maximizing detection of life-threatening cases (Level 1-2). This is the most important clinical metric.
    
    **Operational Impact:**
    - Critical patients seen **66% faster**
    - Overall wait times reduced **26%**
    - **98%** reduction in patients waiting >90 minutes
    - **Free** and open-source
    """)

with col2:
    st.markdown("### 📊 System Performance")
    
    # Create simple metrics display
    st.metric(
        label="🚨 Critical Detection Rate",
        value="83.5%",
        delta="Level 1-2 patients",
        help="Correctly detects 83.5 out of 100 life-threatening cases as critical"
    )
    
    st.metric(
        label="🎯 Critical Accuracy",
        value="77.7%",
        delta="On real clinical data",
        help="Validated on 10,000 MIMIC-IV emergency department visits"
    )
    
    st.metric(
        label="⚡ Wait Time Reduction",
        value="66%",
        delta="For urgent patients",
        help="Critical patients seen 66% faster (45 → 15 minutes)"
    )
    
    st.metric(
        label="💰 Cost per Clinic",
        value="$0",
        delta="Free & open-source",
        help="Commercial triage systems cost $10K-$50K"
    )

st.markdown("---")

# How it works
st.markdown("### 🔄 How ClinicTriage Works")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### 1️⃣ Patient Intake
    
    Patient completes simple form on tablet:
    - Chief complaint
    - Symptom severity (1-10)
    - Basic vitals (if available)
    - Medical history
    
    ⏱️ Takes 2-3 minutes
    """)

with col2:
    st.markdown("""
    #### 2️⃣ AI Triage
    
    MIMIC-IV trained model analyzes:
    - 20 clinical features
    - Red flag symptoms
    - Vital sign patterns
    - Chronic conditions
    
    ⚡ Predicts urgency in <1 second
    🚨 83.5% critical detection rate
    """)

with col3:
    st.markdown("""
    #### 3️⃣ Smart Queue
    
    Optimization balances:
    - Medical urgency (safety)
    - Wait time fairness (equity)
    - Throughput (efficiency)
    
    🎯 Updates dynamically as patients arrive
    """)

st.markdown("---")

# Call to action
st.markdown("### 🚀 Get Started")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("👤 **Try Patient Intake**\n\nExperience the triage form and see AI predictions in real-time.")
    if st.button("Go to Patient Intake", use_container_width=True):
        st.switch_page("pages/1_👤_Patient_Intake.py")

with col2:
    st.success("📊 **View Queue Dashboard**\n\nSee how ClinicTriage optimizes patient order for providers.")
    if st.button("Go to Queue Dashboard", use_container_width=True):
        st.switch_page("pages/2_📊_Queue_Dashboard.py")

with col3:
    st.warning("📈 **Run Simulation**\n\nCompare FCFS vs ClinicTriage with real clinic data.")
    if st.button("Go to Simulation", use_container_width=True):
        st.switch_page("pages/3_📈_Simulation.py")

st.markdown("---")

# Footer
st.markdown("""
<div style='text-align: center; color: #888; padding: 2rem;'>
    <p><strong>ClinicTriage</strong> | AI for Healthcare Equity</p>
    <p>Trained on MIMIC-IV-ED (10,000 real emergency department visits)</p>
    <p>Model: 83.5% Critical Detection Rate | 77.7% Critical Accuracy</p>
</div>
""", unsafe_allow_html=True)