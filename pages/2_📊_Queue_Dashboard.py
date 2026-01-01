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
            
            # Urgency level with color
            urgency_labels = {
                1: "🔴 L1 Critical",
                2: "🟠 L2 High Risk",
                3: "🟡 L3 Moderate",
                4: "🟢 L4 Low Risk",
                5: "🟢 L5 Minimal"
            }
            
            queue_data.append({
                'Position': i,
                'Patient ID': patient['patient_id'],
                'Age': patient['age'],
                'Gender': patient['gender'],
                'Chief Complaint': patient['chief_complaint'][:40] + '...' if len(patient['chief_complaint']) > 40 else patient['chief_complaint'],
                'Urgency': urgency_labels[patient['urgency_level']],
                'Urgency Level': patient['urgency_level'],  # For coloring
                'Wait Time': f"{wait_minutes:.0f} min",
                'Wait Minutes': wait_minutes,  # For sorting
                'Arrival': patient['arrival_time'].strftime('%H:%M'),
                'Priority Score': patient.get('priority_score', 0) if view_mode == "Optimized Order" else None
            })
        
        df = pd.DataFrame(queue_data)
        
        # Style the dataframe
        def color_urgency(val):
            colors = {
                1: 'background-color: #ff4444; color: white; font-weight: bold;',
                2: 'background-color: #ff8800; color: white; font-weight: bold;',
                3: 'background-color: #ffbb00; color: black; font-weight: bold;',
                4: 'background-color: #88cc88; color: black;',
                5: 'background-color: #66bb66; color: white;'
            }
            return colors.get(val, '')
        
        def color_wait_time(val):
            if val >= 80:
                return 'background-color: #ffcccc; font-weight: bold;'
            elif val >= 60:
                return 'background-color: #ffffcc;'
            return ''
        
        # Display columns to show
        display_cols = ['Position', 'Patient ID', 'Age', 'Gender', 'Chief Complaint', 
                       'Urgency', 'Wait Time', 'Arrival']
        
        if view_mode == "Optimized Order":
            display_cols.append('Priority Score')
        
        # Create styled dataframe
        styled_df = df[display_cols].style.apply(
            lambda x: [color_urgency(v) if i == df.columns.get_loc('Urgency Level') else '' 
                      for i, v in enumerate(x)],
            axis=1
        ).apply(
            lambda x: [color_wait_time(v) if i == df.columns.get_loc('Wait Minutes') else '' 
                      for i, v in enumerate(x)],
            axis=1
        ).format({
            'Priority Score': '{:.2f}' if view_mode == "Optimized Order" else None
        })
        
        # Display table
        st.dataframe(
            df[display_cols],
            hide_index=True,
            use_container_width=True,
            height=400
        )
        
        # ====================================================================
        # VISUALIZATION
        # ====================================================================
        
        st.markdown("---")
        st.markdown("### 📊 Queue Visualizations")
        
        tab1, tab2, tab3 = st.tabs(["Wait Times", "Urgency Distribution", "Timeline"])
        
        with tab1:
            # Wait time bar chart
            fig_wait = px.bar(
                df.sort_values('Position'),
                x='Patient ID',
                y='Wait Minutes',
                color='Urgency Level',
                color_continuous_scale=['#66bb66', '#88cc88', '#ffbb00', '#ff8800', '#ff4444'],
                title='Patient Wait Times',
                labels={'Wait Minutes': 'Wait Time (minutes)', 'Patient ID': 'Patient'}
            )
            
            # Add 90-minute cap line
            fig_wait.add_hline(
                y=90,
                line_dash="dash",
                line_color="red",
                annotation_text="90-min Cap",
                annotation_position="right"
            )
            
            fig_wait.update_layout(height=400)
            st.plotly_chart(fig_wait, use_container_width=True)
        
        with tab2:
            # Urgency distribution pie chart
            urgency_counts = df['Urgency'].value_counts()
            
            fig_urgency = px.pie(
                values=urgency_counts.values,
                names=urgency_counts.index,
                title='Queue Distribution by Urgency Level',
                color_discrete_sequence=['#ff4444', '#ff8800', '#ffbb00', '#88cc88', '#66bb66']
            )
            fig_urgency.update_layout(height=400)
            st.plotly_chart(fig_urgency, use_container_width=True)
        
        with tab3:
            # Timeline of arrivals
            fig_timeline = px.scatter(
                df.sort_values('Arrival'),
                x='Arrival',
                y='Position',
                color='Urgency Level',
                size='Wait Minutes',
                hover_data=['Patient ID', 'Chief Complaint'],
                title='Patient Arrival Timeline',
                color_continuous_scale=['#66bb66', '#88cc88', '#ffbb00', '#ff8800', '#ff4444']
            )
            fig_timeline.update_layout(height=400)
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        # ====================================================================
        # ALERTS & WARNINGS
        # ====================================================================
        
        st.markdown("---")
        st.markdown("### ⚠️ Alerts & Warnings")
        
        # Check for violations
        violations = []
        
        # Long wait times
        long_waits = [p for p in display_queue if (current_time - p['arrival_time']).total_seconds() / 60 > 80]
        if long_waits:
            st.error(f"🚨 **{len(long_waits)} patient(s) waiting >80 minutes** - Approaching fairness cap!")
            for p in long_waits:
                wait = (current_time - p['arrival_time']).total_seconds() / 60
                st.warning(f"   • {p['patient_id']}: {wait:.0f} minutes")
        
        # Critical patients waiting
        critical_waiting = [p for p in display_queue if p['urgency_level'] == 1]
        if critical_waiting:
            st.error(f"🔴 **{len(critical_waiting)} CRITICAL patient(s) in queue** - Immediate attention required!")
            for p in critical_waiting:
                wait = (current_time - p['arrival_time']).total_seconds() / 60
                st.error(f"   • {p['patient_id']}: {p['chief_complaint']} (waiting {wait:.0f} min)")
        
        # High risk patients
        high_risk = [p for p in display_queue if p['urgency_level'] == 2]
        if high_risk:
            st.warning(f"🟠 **{len(high_risk)} HIGH RISK patient(s)** - Urgent care needed soon")
        
        if not violations and not long_waits and not critical_waiting and not high_risk:
            st.success("✅ No alerts - Queue is operating normally")
        
        # ====================================================================
        # PROVIDER ACTIONS
        # ====================================================================
        
        st.markdown("---")
        st.markdown("### 🔧 Provider Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("👤 Call Next Patient", use_container_width=True):
                if display_queue:
                    next_patient = display_queue[0]
                    st.success(f"✅ Calling: {next_patient['patient_id']} - {next_patient['chief_complaint']}")
                    st.info(f"Urgency Level {next_patient['urgency_level']}")
        
        with col2:
            if st.button("✅ Mark Patient Seen", use_container_width=True):
                if display_queue:
                    seen_patient = display_queue[0]
                    # Remove from queue
                    st.session_state.queue = [p for p in st.session_state.queue 
                                             if p['patient_id'] != seen_patient['patient_id']]
                    st.success(f"✅ {seen_patient['patient_id']} marked as seen and removed from queue")
                    st.rerun()
        
        with col3:
            if st.button("🔄 Refresh Queue", use_container_width=True):
                st.rerun()
        
        with col4:
            if st.button("🗑️ Clear All Patients", use_container_width=True, type="secondary"):
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
        
        if st.button("Go to Patient Intake", use_container_width=True):
            st.switch_page("pages/1_👤_Patient_Intake.py")
    
    with col2:
        st.markdown("""
        **Or Load Sample Patients:**
        
        Automatically add 5-10 sample patients to test the queue optimization.
        """)
        
        if st.button("Load Sample Patients", use_container_width=True):
            # Load sample patients from synthetic data
            import pandas as pd
            
            try:
                patients_df = pd.read_csv('synthetic_patients.csv')
                
                # Sample 8 random patients
                sample = patients_df.sample(n=8)
                
                # Add to queue with staggered arrival times
                current_time = datetime.now()
                
                for i, (idx, patient_row) in enumerate(sample.iterrows()):
                    arrival = current_time - timedelta(minutes=np.random.randint(5, 60))
                    
                    patient_record = {
                        'patient_id': f"P{st.session_state.patient_counter:04d}",
                        'name': f"Sample Patient {st.session_state.patient_counter}",
                        'age': int(patient_row['age']),
                        'gender': patient_row['gender'],
                        'chief_complaint': patient_row['chief_complaint'],
                        'urgency_level': int(patient_row['urgency_level']),
                        'arrival_time': arrival,
                        'symptom_severity': patient_row['symptom_severity'],
                        'symptom_duration_hours': patient_row['symptom_duration_hours'],
                        'heart_rate': patient_row['heart_rate'],
                        'systolic_bp': patient_row['systolic_bp'],
                        'diastolic_bp': patient_row['diastolic_bp'],
                        'temperature': patient_row['temperature'],
                        'oxygen_saturation': patient_row['oxygen_saturation'],
                        'has_red_flag': patient_row['has_red_flag'],
                        'has_chronic_condition': patient_row['has_chronic_condition'],
                        'high_risk_chronic': patient_row['high_risk_chronic'],
                        'hr_abnormal': patient_row['hr_abnormal'],
                        'bp_abnormal': patient_row['bp_abnormal'],
                        'temp_abnormal': patient_row['temp_abnormal'],
                        'spo2_abnormal': patient_row['spo2_abnormal'],
                        'vital_abnormalities': patient_row['vital_abnormalities'],
                        'symptom_acuity': patient_row['symptom_acuity'],
                        'previous_visits': patient_row['previous_visits'],
                        'gender_encoded': 1 if patient_row['gender'] == 'Female' else 0,
                        'onset_encoded': 1 if patient_row['symptom_onset'] == 'Sudden' else 0
                    }
                    
                    st.session_state.queue.append(patient_record)
                    st.session_state.patient_counter += 1
                
                st.success(f"✅ Added {len(sample)} sample patients to queue!")
                st.rerun()
                
            except FileNotFoundError:
                st.error("❌ Could not load sample data. Please add patients manually via Patient Intake.")

# ============================================================================
# SIDEBAR INFO
# ============================================================================

with st.sidebar:
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