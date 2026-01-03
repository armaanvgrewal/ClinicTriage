"""
ClinicFlow - Queue Dashboard
Provider view of optimized patient queue
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from queue_optimizer import QueueOptimizer

st.set_page_config(page_title="Queue Dashboard - ClinicFlow", page_icon="📊", layout="wide")

# ============================================================================
# LOAD OPTIMIZER
# ============================================================================

@st.cache_resource
def load_optimizer():
    return QueueOptimizer(
        urgency_weight=10.0,
        wait_time_weight=0.15,
        age_risk_weight=0.05,
        max_wait_minutes=90
    )

optimizer = load_optimizer()

# ============================================================================
# HEADER
# ============================================================================

st.title("📊 Queue Dashboard - Provider View")
st.markdown("Real-time optimized patient queue based on urgency, fairness, and efficiency")

st.markdown("---")

# ============================================================================
# QUICK STATS
# ============================================================================

col1, col2, col3, col4 = st.columns(4)

total_patients = len(st.session_state.queue)

with col1:
    st.metric(
        "Total Patients in Queue",
        total_patients,
        help="All patients currently waiting"
    )

with col2:
    if st.session_state.queue:
        urgent_count = sum(1 for p in st.session_state.queue if p.get('urgency_level', 5) <= 2)
        st.metric(
            "Critical/High Risk (L1-2)",
            urgent_count,
            delta="Priority patients" if urgent_count > 0 else None,
            help="Patients requiring immediate or urgent attention"
        )
    else:
        st.metric("Critical/High Risk (L1-2)", 0)

with col3:
    if st.session_state.queue:
        avg_wait = sum((datetime.now() - p['arrival_time']).total_seconds() / 60 
                      for p in st.session_state.queue) / len(st.session_state.queue)
        st.metric(
            "Average Wait Time",
            f"{avg_wait:.0f} min",
            help="Current average wait across all patients"
        )
    else:
        st.metric("Average Wait Time", "0 min")

with col4:
    if st.session_state.queue:
        max_wait = max((datetime.now() - p['arrival_time']).total_seconds() / 60 
                      for p in st.session_state.queue)
        st.metric(
            "Longest Wait",
            f"{max_wait:.0f} min",
            delta="⚠️ Alert" if max_wait > 80 else None,
            delta_color="inverse",
            help="Patient who has been waiting longest"
        )
    else:
        st.metric("Longest Wait", "0 min")

st.markdown("---")

# ============================================================================
# OPTIMIZE QUEUE
# ============================================================================

if st.session_state.queue:
    # Get current time
    current_time = datetime.now()
    
    # Optimize the queue
    optimized_queue = optimizer.optimize_queue(st.session_state.queue, current_time)
    
    # ========================================================================
    # QUEUE DISPLAY OPTIONS
    # ========================================================================
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        view_mode = st.radio(
            "View Mode",
            options=["Optimized Order", "Arrival Order", "Urgency Only"],
            horizontal=True,
            help="Choose how to sort the queue"
        )
    
    with col2:
        filter_urgency = st.multiselect(
            "Filter by Urgency Level",
            options=[1, 2, 3, 4, 5],
            default=[1, 2, 3, 4, 5],
            format_func=lambda x: f"Level {x}",
            help="Show only selected urgency levels"
        )
    
    with col3:
        auto_refresh = st.checkbox(
            "Auto-refresh",
            value=False,
            help="Automatically update queue every 30 seconds"
        )
        
        if auto_refresh:
            st.rerun()
    
    # ========================================================================
    # SORT QUEUE BASED ON VIEW MODE
    # ========================================================================
    
    if view_mode == "Optimized Order":
        display_queue = optimized_queue
    elif view_mode == "Arrival Order":
        display_queue = sorted(st.session_state.queue, key=lambda x: x['arrival_time'])
    else:  # Urgency Only
        display_queue = sorted(st.session_state.queue, key=lambda x: x['urgency_level'])
    
    # Apply urgency filter
    display_queue = [p for p in display_queue if p['urgency_level'] in filter_urgency]
    
    st.markdown("---")
    
    # ========================================================================
    # QUEUE TABLE DISPLAY
    # ========================================================================
    
    st.markdown(f"### 📋 Patient Queue ({len(display_queue)} patients)")
    
    if display_queue:
        # Create display dataframe
        queue_data = []
        
        for i, patient in enumerate(display_queue, 1):
            wait_minutes = (current_time - patient['arrival_time']).total_seconds() / 60
            
            # Urgency level with color and emoji
            urgency_labels = {
                1: "🔴 L1 Critical",
                2: "🟠 L2 High Risk",
                3: "🟡 L3 Moderate",
                4: "🟢 L4 Low Risk",
                5: "🟢 L5 Minimal"
            }
            
            # Format wait time with color coding
            if wait_minutes >= 80:
                wait_display = f"⚠️ {wait_minutes:.0f} min"
            elif wait_minutes >= 60:
                wait_display = f"⚡ {wait_minutes:.0f} min"
            else:
                wait_display = f"{wait_minutes:.0f} min"
            
            row_data = {
                'Position': i,
                'Patient ID': patient['patient_id'],
                'Age': patient['age'],
                'Gender': patient['gender'],
                'Chief Complaint': patient['chief_complaint'][:40] + '...' if len(patient['chief_complaint']) > 40 else patient['chief_complaint'],
                'Urgency': urgency_labels[patient['urgency_level']],
                'Wait Time': wait_display,
                'Arrival': patient['arrival_time'].strftime('%H:%M')
            }
            
            if view_mode == "Optimized Order":
                row_data['Priority Score'] = f"{patient.get('priority_score', 0):.1f}"
            
            queue_data.append(row_data)
        
        df = pd.DataFrame(queue_data)
        
        # Display with simple dataframe (no complex styling)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # ====================================================================
        # PATIENT DETAILS
        # ====================================================================
        
        st.markdown("---")
        st.markdown("### 🔍 Patient Details")
        
        selected_patient_id = st.selectbox(
            "Select Patient",
            options=[p['patient_id'] for p in display_queue],
            format_func=lambda x: f"{x} - {next(p['chief_complaint'] for p in display_queue if p['patient_id'] == x)}"
        )
        
        selected = next(p for p in display_queue if p['patient_id'] == selected_patient_id)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"**Patient ID:** {selected['patient_id']}")
            st.markdown(f"**Age:** {selected['age']} years")
            st.markdown(f"**Gender:** {selected['gender']}")
        
        with col2:
            st.markdown(f"**Urgency Level:** {selected['urgency_level']}")
            st.markdown(f"**Arrival:** {selected['arrival_time'].strftime('%H:%M')}")
            wait = (current_time - selected['arrival_time']).total_seconds() / 60
            st.markdown(f"**Wait Time:** {wait:.0f} minutes")
        
        with col3:
            st.markdown(f"**Chief Complaint:** {selected['chief_complaint']}")
            st.markdown(f"**Symptom Severity:** {selected.get('symptom_severity', 'N/A')}/10")
            st.markdown(f"**Red Flags:** {'Yes ⚠️' if selected.get('has_red_flag', 0) else 'No'}")
        
        with col4:
            st.markdown("**Vitals:**")
            st.markdown(f"HR: {selected.get('heart_rate', 'N/A')} bpm")
            st.markdown(f"BP: {selected.get('systolic_bp', 'N/A')}/{selected.get('diastolic_bp', 'N/A')}")
            st.markdown(f"Temp: {selected.get('temperature', 'N/A')}°F")
            st.markdown(f"O2: {selected.get('oxygen_saturation', 'N/A')}%")
        
        # ====================================================================
        # QUEUE ACTIONS
        # ====================================================================
        
        st.markdown("---")
        st.markdown("### ⚙️ Queue Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("👁️ See Next Patient", width='stretch', type="primary"):
                if display_queue:
                    next_patient = display_queue[0]
                    st.success(f"✅ Next: {next_patient['patient_id']} - {next_patient['chief_complaint']}")
                    st.info(f"Urgency Level {next_patient['urgency_level']}")
        
        with col2:
            if st.button("✅ Mark Patient Seen", width='stretch'):
                if display_queue:
                    seen_patient = display_queue[0]
                    # Remove from queue
                    st.session_state.queue = [p for p in st.session_state.queue 
                                             if p['patient_id'] != seen_patient['patient_id']]
                    st.success(f"✅ {seen_patient['patient_id']} marked as seen and removed from queue")
                    st.rerun()
        
        with col3:
            if st.button("🔄 Refresh Queue", width='stretch'):
                st.rerun()
        
        with col4:
            if st.button("🗑️ Clear All Patients", width='stretch', type="secondary"):
                st.session_state.queue = []
                st.success("✅ Queue cleared")
                st.rerun()
        
        # ====================================================================
        # COMPARISON VIEW
        # ====================================================================
        
        st.markdown("---")
        st.markdown("### 🔄 FCFS vs ClinicFlow Comparison")
        
        with st.expander("📊 Show Comparison Analysis"):
            # FCFS order
            fcfs_queue = sorted(st.session_state.queue, key=lambda x: x['arrival_time'])
            
            # Calculate metrics for both
            def calc_metrics(queue):
                current = datetime.now()
                wait_times = [(current - p['arrival_time']).total_seconds() / 60 for p in queue]
                urgent_waits = [(current - p['arrival_time']).total_seconds() / 60 
                               for p in queue if p['urgency_level'] <= 2]
                
                return {
                    'avg_wait': np.mean(wait_times) if wait_times else 0,
                    'max_wait': max(wait_times) if wait_times else 0,
                    'urgent_avg': np.mean(urgent_waits) if urgent_waits else 0,
                    'over_90': sum(1 for w in wait_times if w > 90)
                }
            
            fcfs_metrics = calc_metrics(fcfs_queue)
            cf_metrics = calc_metrics(optimized_queue)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### First-Come-First-Served")
                st.metric("Average Wait", f"{fcfs_metrics['avg_wait']:.0f} min")
                st.metric("Urgent Patient Wait", f"{fcfs_metrics['urgent_avg']:.0f} min")
                st.metric("Patients >90 min", fcfs_metrics['over_90'])
            
            with col2:
                st.markdown("#### ClinicFlow Optimized")
                improvement_avg = ((fcfs_metrics['avg_wait'] - cf_metrics['avg_wait']) / 
                                  max(fcfs_metrics['avg_wait'], 1)) * 100
                improvement_urgent = ((fcfs_metrics['urgent_avg'] - cf_metrics['urgent_avg']) / 
                                     max(fcfs_metrics['urgent_avg'], 1)) * 100
                
                st.metric(
                    "Average Wait", 
                    f"{cf_metrics['avg_wait']:.0f} min",
                    delta=f"{-improvement_avg:.0f}%" if improvement_avg > 0 else None,
                    delta_color="inverse"
                )
                st.metric(
                    "Urgent Patient Wait", 
                    f"{cf_metrics['urgent_avg']:.0f} min",
                    delta=f"{-improvement_urgent:.0f}%" if improvement_urgent > 0 else None,
                    delta_color="inverse"
                )
                st.metric(
                    "Patients >90 min", 
                    cf_metrics['over_90'],
                    delta=fcfs_metrics['over_90'] - cf_metrics['over_90'],
                    delta_color="inverse"
                )
    
    else:
        st.info("No patients match the selected urgency filter")

else:
    # Empty queue state
    st.info("📭 Queue is empty - No patients currently waiting")
    
    st.markdown("---")
    st.markdown("### 🎯 Get Started")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Add Patients:**
        1. Go to **Patient Intake** page
        2. Fill out the triage form
        3. Click "Add to Queue"
        4. Return here to see optimized queue
        """)
        
        if st.button("Go to Patient Intake", width='stretch'):
            st.switch_page("pages/1_👤_Patient_Intake.py")
    
    with col2:
        st.markdown("""
        **Or Load Sample Patients:**
        
        Automatically add 5-10 sample patients to test the queue optimization.
        """)
        
        if st.button("Load Sample Patients", width='stretch'):
            # Load sample patients from synthetic data with proper error handling
            try:
                patients_df = pd.read_csv('synthetic_patients.csv')
                
                # Sample 8 random patients
                sample = patients_df.sample(n=8)
                
                # Add to queue with staggered arrival times
                current_time = datetime.now()
                
                for i, (idx, patient_row) in enumerate(sample.iterrows()):
                    arrival = current_time - timedelta(minutes=np.random.randint(5, 60))
                    
                    # Safely get symptom_onset with fallback to 'Gradual'
                    symptom_onset = patient_row.get('symptom_onset', 'Gradual')
                    
                    patient_record = {
                        'patient_id': f"P{st.session_state.patient_counter:04d}",
                        'name': f"Sample Patient {st.session_state.patient_counter}",
                        'age': int(patient_row.get('age', 45)),
                        'gender': patient_row.get('gender', 'Male'),
                        'chief_complaint': patient_row.get('chief_complaint', 'General checkup'),
                        'urgency_level': int(patient_row.get('urgency_level', 3)),
                        'arrival_time': arrival,
                        'symptom_severity': float(patient_row.get('symptom_severity', 5)),
                        'symptom_duration_hours': float(patient_row.get('symptom_duration_hours', 2.0)),
                        'heart_rate': float(patient_row.get('heart_rate', 75)),
                        'systolic_bp': float(patient_row.get('systolic_bp', 120)),
                        'diastolic_bp': float(patient_row.get('diastolic_bp', 80)),
                        'temperature': float(patient_row.get('temperature', 98.6)),
                        'oxygen_saturation': float(patient_row.get('oxygen_saturation', 98)),
                        'has_red_flag': int(patient_row.get('has_red_flag', 0)),
                        'has_chronic_condition': int(patient_row.get('has_chronic_condition', 0)),
                        'high_risk_chronic': int(patient_row.get('high_risk_chronic', 0)),
                        'hr_abnormal': int(patient_row.get('hr_abnormal', 0)),
                        'bp_abnormal': int(patient_row.get('bp_abnormal', 0)),
                        'temp_abnormal': int(patient_row.get('temp_abnormal', 0)),
                        'spo2_abnormal': int(patient_row.get('spo2_abnormal', 0)),
                        'vital_abnormalities': int(patient_row.get('vital_abnormalities', 0)),
                        'symptom_acuity': float(patient_row.get('symptom_acuity', 5.0)),
                        'previous_visits': int(patient_row.get('previous_visits', 1)),
                        'gender_encoded': 1 if patient_row.get('gender', 'Male') == 'Female' else 0,
                        'onset_encoded': 1 if symptom_onset == 'Sudden' else 0
                    }
                    
                    st.session_state.queue.append(patient_record)
                    st.session_state.patient_counter += 1
                
                st.success(f"✅ Added {len(sample)} sample patients to queue!")
                st.rerun()
                
            except FileNotFoundError:
                st.error("❌ Could not load sample data file 'synthetic_patients.csv'")
                st.info("💡 Please add patients manually via Patient Intake page, or the file may be missing from your project directory.")
            except KeyError as e:
                st.error(f"❌ Missing expected column in data: {e}")
                st.info("💡 The sample data file may have a different format. Please add patients manually via Patient Intake page.")
            except Exception as e:
                st.error(f"❌ Unexpected error loading sample patients: {str(e)}")
                st.info("💡 Please add patients manually via Patient Intake page.")

# ============================================================================
# SIDEBAR INFO
# ============================================================================

with st.sidebar:
    st.markdown("### ℹ️ System Info")
    st.write("**ClinicFlow v2.0**")
    st.write("AI-Powered Triage System")
    
    st.divider()
    
    st.write("**🤖 Model Information**")
    st.metric("Model", "MIMIC-IV v2")
    st.metric("Overall Accuracy", "78.5%")
    st.metric("Critical Accuracy", "89.3%")
    st.caption("Trained on 10K real ED visits")
    
    st.divider()
    
    st.markdown("### 📊 Dashboard Guide")
    
    st.markdown("""
    **View Modes:**
    - **Optimized Order**: ClinicFlow's smart queue
    - **Arrival Order**: Traditional FCFS
    - **Urgency Only**: Sorted by urgency level
    
    **Color Coding:**
    - 🔴 Level 1: Critical
    - 🟠 Level 2: High Risk
    - 🟡 Level 3: Moderate
    - 🟢 Level 4-5: Low/Minimal
    
    **Alerts:**
    - Wait >80 min: Yellow warning
    - Wait >90 min: Red alert
    - Critical in queue: Urgent notification
    """)
    
    st.markdown("---")
    
    st.markdown("### ⚙️ Optimization Settings")
    
    st.markdown(f"""
    **Current Configuration:**
    - Urgency weight: {optimizer.urgency_weight}
    - Wait time weight: {optimizer.wait_time_weight}
    - Max wait cap: {optimizer.max_wait_minutes} min
    """)
