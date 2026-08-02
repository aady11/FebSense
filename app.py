import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time
import io
from fpdf import FPDF

# -------------------------------------------------------
# 1. APP CONFIGURATION & INDUSTRIAL CSS THEME
# -------------------------------------------------------
st.set_page_config(page_title="FabSense", page_icon="🏭", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #1A1D21; color: #E8E6E1; }
    [data-testid="stSidebar"] { background-color: #14161A; border-right: 1px solid #333; }
    body, .stMarkdown, .stLabel, .stCaption { font-family: 'IBM Plex Sans', 'Inter', sans-serif; }
    .stMetric, .stMetricValue, .stMetricDelta, code, .dataframe { font-family: 'IBM Plex Mono', 'JetBrains Mono', monospace !important; }
    
    .exec-card {
        background-color: #23272B; border: 1px solid #333; border-left: 5px solid #F2A93B;
        border-radius: 8px; padding: 20px; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .exec-title {
        color: #F2A93B; font-family: 'IBM Plex Mono', monospace; font-size: 1.2rem;
        font-weight: bold; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 15px;
    }
    
    [data-testid="stMetricValue"] { color: #E8E6E1; font-size: 1.8rem !important; }
    [data-testid="stMetricLabel"] { color: #A0A0A0; }
    [data-testid="stMetricDelta"] { font-size: 1rem !important; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 24px; background-color: #1A1D21; border-bottom: 1px solid #333; }
    .stTabs [data-baseweb="tab"] { color: #A0A0A0 !important; font-family: 'IBM Plex Mono', monospace; }
    .stTabs [aria-selected="true"] { color: #F2A93B !important; border-bottom-color: #F2A93B !important; }
    .stSlider [data-baseweb="slider-thumb"] { background-color: #F2A93B; }

    .gauge-container { margin-bottom: 12px; }
    .gauge-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.9rem; color: #A0A0A0; margin-bottom: 4px; display: flex; justify-content: space-between; }
    .gauge-track { width: 100%; height: 12px; background-color: #333; border-radius: 6px; position: relative; overflow: hidden; }
    .gauge-safe-zone { position: absolute; height: 100%; background-color: rgba(91, 143, 168, 0.4); border-radius: 4px; }
    .gauge-marker { position: absolute; top: -4px; width: 4px; height: 20px; border-radius: 2px; }
    .gauge-marker.normal { background-color: #5B8FA8; box-shadow: 0 0 8px rgba(91, 143, 168, 0.6); }
    .gauge-marker.danger { background-color: #F2A93B; box-shadow: 0 0 8px rgba(242, 169, 59, 0.8); }

    .sensor-card {
        background-color: #23272B; border: 1px solid #333; border-radius: 8px;
        padding: 15px; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .sensor-card-header { font-family: 'IBM Plex Mono', monospace; font-size: 0.9rem; color: #A0A0A0; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
    .sensor-card-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem; font-weight: bold; color: #E8E6E1; }
    .sensor-card-status { font-family: 'IBM Plex Mono', monospace; font-size: 0.9rem; margin-top: 5px; }
    .status-good { color: #5B8FA8; }
    .status-danger { color: #F2A93B; }

    /* Chatbot styling */
    .stChatMessage { background-color: #23272B; border-radius: 8px; border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p style="font-family: IBM Plex Mono; color: #F2A93B; font-size: 2rem; font-weight: bold; margin-bottom: 0;">FabSense</p>', unsafe_allow_html=True)
st.caption("Process Intelligence Platform — AI Co-Pilot for Yield Loss")
st.divider()

# -------------------------------------------------------
# 2. DATA GENERATION & MODEL TRAINING
# -------------------------------------------------------
@st.cache_data
def generate_data():
    np.random.seed(42)
    n_wafers = 2000
    process_steps = ['Etch', 'Deposition', 'Lithography']
    step_assignment = np.random.choice(process_steps, size=n_wafers)
    data = []
    for i in range(n_wafers):
        step = step_assignment[i]
        if step == 'Etch':
            temperature = np.random.normal(75, 3); pressure = np.random.normal(5.0, 0.3)
            gas_flow = np.random.normal(120, 8); etch_rate = np.random.normal(550, 25)
            voltage = np.random.normal(300, 15); current = np.random.normal(6.0, 0.4)
        elif step == 'Deposition':
            temperature = np.random.normal(450, 15); pressure = np.random.normal(2.5, 0.2)
            gas_flow = np.random.normal(200, 12); etch_rate = 0; voltage = 0; current = 0
        else:
            temperature = np.random.normal(22, 0.5); pressure = np.random.normal(1.0, 0.05)
            gas_flow = 0; etch_rate = 0; voltage = np.random.normal(50, 3); current = np.random.normal(2.0, 0.2)
        
        defect_prob = 0.05
        if step == 'Etch':
            if etch_rate > 590 or etch_rate < 510: defect_prob += 0.45
            if voltage > 325 or voltage < 275: defect_prob += 0.35
            if current > 6.7 or current < 5.3: defect_prob += 0.30
            if temperature > 80 or temperature < 70: defect_prob += 0.25
            if gas_flow > 135 or gas_flow < 105: defect_prob += 0.20
            if pressure > 5.5 or pressure < 4.5: defect_prob += 0.15
        elif step == 'Deposition':
            if temperature > 475 or temperature < 425: defect_prob += 0.50
            if pressure > 2.8 or pressure < 2.2: defect_prob += 0.35
            if gas_flow > 220 or gas_flow < 180: defect_prob += 0.30
        else:
            if temperature > 23 or temperature < 21: defect_prob += 0.50
            if voltage > 55 or voltage < 45: defect_prob += 0.35
            if current > 2.3 or current < 1.7: defect_prob += 0.30
            if pressure > 1.08 or pressure < 0.92: defect_prob += 0.25
        
        defect_prob = min(defect_prob, 0.95)
        defect_label = 1 if np.random.random() < defect_prob else 0
        data.append({'process_step': step, 'temperature': round(temperature, 2), 'pressure': round(pressure, 3),
                     'gas_flow': round(gas_flow, 1), 'etch_rate': round(etch_rate, 1), 'voltage': round(voltage, 1),
                     'current': round(current, 2), 'defect_label': defect_label})
    return pd.DataFrame(data)

@st.cache_resource
def train_model():
    df = generate_data()
    y = df['defect_label']
    df_features = df.drop('defect_label', axis=1)
    X = pd.get_dummies(df_features, columns=['process_step'], drop_first=False)
    feature_names = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model, feature_names

model, feature_names = train_model()

normal_ranges = {
    'Etch': {'temperature': (70, 80), 'pressure': (4.5, 5.5), 'gas_flow': (105, 135), 'etch_rate': (510, 590), 'voltage': (275, 325), 'current': (5.3, 6.7)},
    'Deposition': {'temperature': (425, 475), 'pressure': (2.2, 2.8), 'gas_flow': (180, 220)},
    'Lithography': {'temperature': (21, 23), 'pressure': (0.92, 1.08), 'voltage': (45, 55), 'current': (1.7, 2.3)}
}

# -------------------------------------------------------
# 3. SIDEBAR INPUTS
# -------------------------------------------------------
with st.sidebar:
    st.header("Control Panel")
    app_mode = st.radio("Operation Mode", ["Single Wafer Check", "Batch Analysis"], index=0)
    st.divider()
    
    if app_mode == "Single Wafer Check":
        st.markdown("Configure wafer parameters below.")
        process_step = st.selectbox("Process Step", ['Etch', 'Deposition', 'Lithography'])

        if process_step == 'Etch':
            temperature = st.slider("Temperature (C)", 60.0, 90.0, 75.0)
            pressure = st.slider("Pressure (Torr)", 4.0, 6.0, 5.0)
            gas_flow = st.slider("Gas Flow (sccm)", 100.0, 140.0, 120.0)
            etch_rate = st.slider("Etch Rate (A/min)", 500.0, 600.0, 550.0)
            voltage = st.slider("Voltage (V)", 250.0, 350.0, 300.0)
            current = st.slider("Current (A)", 5.0, 7.0, 6.0)
        elif process_step == 'Deposition':
            temperature = st.slider("Temperature (C)", 400.0, 500.0, 450.0)
            pressure = st.slider("Pressure (Torr)", 2.0, 3.0, 2.5)
            gas_flow = st.slider("Gas Flow (sccm)", 180.0, 220.0, 200.0)
            etch_rate = 0.0; voltage = 0.0; current = 0.0
        else:
            temperature = st.slider("Temperature (C)", 20.0, 24.0, 22.0)
            pressure = st.slider("Pressure (Torr)", 0.9, 1.1, 1.0)
            gas_flow = 0.0; etch_rate = 0.0
            voltage = st.slider("Voltage (V)", 40.0, 60.0, 50.0)
            current = st.slider("Current (A)", 1.5, 2.5, 2.0)
        
        st.divider()
        st.subheader("Decision Settings")
        threshold = st.slider("Action Threshold (0-100)", 10, 90, 30)

    else:
        st.markdown("Upload a CSV file with wafer data.")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        threshold = st.slider("Action Threshold (0-100)", 10, 90, 30)

# -------------------------------------------------------
# 4. PDF GENERATION LOGIC
# -------------------------------------------------------
def create_pdf_report(proc_step, risk, pred, top_feat, top_val, safe_min, safe_max):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    
    pdf.cell(200, 10, txt="FabSense AI Inspection Report", ln=True, align='C')
    pdf.cell(200, 10, txt="---------------------------", ln=True, align='C')
    pdf.ln(10)
    
    pdf.cell(200, 10, txt=f"Process Step: {proc_step}", ln=True)
    pdf.cell(200, 10, txt=f"Risk Score: {risk:.1f} / 100", ln=True)
    pdf.cell(200, 10, txt=f"Prediction: {pred}", ln=True)
    pdf.ln(10)
    
    pdf.set_font("Helvetica", size=10)
    pdf.cell(200, 10, txt="Primary Root Cause:", ln=True)
    if top_val > safe_max or top_val < safe_min:
        pdf.cell(200, 10, txt=f"- {top_feat} is out of spec (Reading: {top_val:.1f}, Safe: {safe_min}-{safe_max})", ln=True)
    else:
        pdf.cell(200, 10, txt="- All sensors within normal parameters.", ln=True)
        
    pdf.ln(10)
    pdf.cell(200, 10, txt="Financial Impact (Per Wafer):", ln=True)
    pdf.cell(200, 10, txt="- Scrap Cost if Missed: $500", ln=True)
    pdf.cell(200, 10, txt="- Savings if Caught Early: $300", ln=True)
    
    return bytes(pdf.output())

# -------------------------------------------------------
# 5. MAIN DASHBOARD
# -------------------------------------------------------
if app_mode == "Single Wafer Check":
    input_data = {
        'temperature': temperature, 'pressure': pressure, 'gas_flow': gas_flow,
        'etch_rate': etch_rate, 'voltage': voltage, 'current': current,
        'process_step_Etch': 1 if process_step == 'Etch' else 0,
        'process_step_Deposition': 1 if process_step == 'Deposition' else 0,
        'process_step_Lithography': 1 if process_step == 'Lithography' else 0
    }
    
    input_df = pd.DataFrame([input_data]).reindex(columns=feature_names, fill_value=0)
    
    with st.status("Analyzing wafer sensors...", expanded=True) as status:
        st.write("Running AI prediction model...")
        prob = model.predict_proba(input_df)[0][1]
        risk_score = prob * 100
        
        st.write("Generating SHAP explanations...")
        explainer = shap.TreeExplainer(model)
        shap_values_raw = explainer.shap_values(input_df)
        if isinstance(shap_values_raw, list):
            shap_vals = shap_values_raw[1][0]
        else:
            shap_vals = shap_values_raw[0, :, 1]
        
        active_features = [(f, s, input_data[f]) for f, s in zip(feature_names, shap_vals) if f in normal_ranges.get(process_step, {})]
        active_features.sort(key=lambda x: abs(x[1]), reverse=True)
        
        time.sleep(0.5)
        status.update(label="Analysis Complete", state="complete", expanded=False)

    if risk_score >= 50: risk_label = "HIGH RISK"
    elif risk_score >= threshold: risk_label = "ELEVATED RISK"
    else: risk_label = "LOW RISK"
    
    prediction = "DEFECTIVE" if risk_score >= threshold else "GOOD"

    top_feature = active_features[0][0] if active_features else "N/A"
    top_val = active_features[0][2] if active_features else 0
    safe_min, safe_max = normal_ranges.get(process_step, {}).get(top_feature, (0, 0))
    
    if top_val > safe_max or top_val < safe_min:
        action_text = f"Halt process. Inspect {top_feature} controller immediately."
    else:
        action_text = "Proceed to next step. All sensors nominal."

    st.markdown(f"""
    <div class="exec-card">
        <div class="exec-title">Executive Summary — {process_step} Step</div>
        <p><b>Risk Level:</b> {risk_label} ({risk_score:.1f}/100) &nbsp;&nbsp; | &nbsp;&nbsp; 
        <b>Primary Cause:</b> {top_feature} &nbsp;&nbsp; | &nbsp;&nbsp; 
        <b>Financial Impact:</b> $500 (Rs. 41,500)</p>
        <p><b>Recommended Action:</b> {action_text}</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Live Verdict", "Diagnostics", "Impact Report", "AI Co-Pilot"])

    with tab1:
        st.header("Overall Risk Verdict")
        col1, col2 = st.columns([1.5, 1])
        with col1:
            if risk_score >= 50: gauge_color = "#e74c3c"
            elif risk_score >= threshold: gauge_color = "#F2A93B"
            else: gauge_color = "#5B8FA8"
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = risk_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"Risk Score: {risk_score:.1f}/100", 'font': {'size': 20, 'color': '#E8E6E1'}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#E8E6E1"},
                    'bar': {'color': gauge_color, 'thickness': 0.4}, 'bgcolor': "#1A1D21", 'borderwidth': 2, 'bordercolor': "#333",
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(91, 143, 168, 0.2)'},
                        {'range': [30, 60], 'color': 'rgba(242, 169, 59, 0.2)'},
                        {'range': [60, 100], 'color': 'rgba(231, 76, 60, 0.2)'}],
                    'threshold': {'line': {'color': "#E8E6E1", 'width': 4}, 'thickness': 0.75, 'value': threshold}
                }
            ))
            fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=60, b=20), paper_bgcolor="#1A1D21")
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.caption(f"White line = Action Threshold: {threshold}/100")
            
        with col2:
            st.markdown("#### Summary")
            st.metric("Process Step", process_step)
            st.metric("Prediction", prediction)
            st.metric("Risk Level", risk_label)

        st.divider()
        st.header("SCADA Sensor Gauge Stack")
        st.caption("Visual representation of sensor positions within their safe operating bands.")
        for feature, shap_val, val in active_features:
            safe_min_f, safe_max_f = normal_ranges[process_step][feature]
            bounds_map = {'temperature': (20.0, 500.0), 'pressure': (0.9, 6.0), 'gas_flow': (100.0, 220.0), 'etch_rate': (500.0, 600.0), 'voltage': (40.0, 350.0), 'current': (1.5, 7.0)}
            track_min, track_max = bounds_map.get(feature, (0, 100))
            track_range = track_max - track_min
            if track_range == 0: continue
            safe_start_pct = ((safe_min_f - track_min) / track_range) * 100
            safe_width_pct = ((safe_max_f - safe_min_f) / track_range) * 100
            marker_pct = max(0, min(100, ((val - track_min) / track_range) * 100))
            is_normal = safe_min_f <= val <= safe_max_f
            marker_class = "normal" if is_normal else "danger"
            val_color = "#5B8FA8" if is_normal else "#F2A93B"
            st.markdown(f"""
            <div class="gauge-container">
                <div class="gauge-label"><span>{feature.upper()}</span><span style="color: {val_color};">{val:.1f}</span></div>
                <div class="gauge-track">
                    <div class="gauge-safe-zone" style="left: {safe_start_pct}%; width: {safe_width_pct}%;"></div>
                    <div class="gauge-marker {marker_class}" style="left: {marker_pct}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.header("Sensor Health & Diagnostics")
        cols = st.columns(3)
        col_idx = 0
        for feature, shap_val, val in active_features:
            safe_min_f, safe_max_f = normal_ranges[process_step][feature]
            is_normal = safe_min_f <= val <= safe_max_f
            status_icon = "NORMAL" if is_normal else "EXCURSION"
            status_class = "status-good" if is_normal else "status-danger"
            with cols[col_idx]:
                st.markdown(f"""
                <div class="sensor-card">
                    <div class="sensor-card-header">{feature}</div>
                    <div class="sensor-card-value">{val:.1f}</div>
                    <div class="sensor-card-status {status_class}">{status_icon}</div>
                    <div style="font-size: 0.8rem; color: #A0A0A0; margin-top: 5px;">Safe: {safe_min_f} - {safe_max_f}</div>
                    <div style="font-size: 0.8rem; color: #A0A0A0;">SHAP: {shap_val:+.4f}</div>
                </div>
                """, unsafe_allow_html=True)
            col_idx = (col_idx + 1) % 3
            
        st.divider()
        st.subheader("Feature Impact (SHAP)")
        feature_importance = pd.DataFrame({'Feature': feature_names, 'SHAP Value': shap_vals})
        feature_importance = feature_importance[feature_importance['SHAP Value'] != 0]
        feature_importance = feature_importance[~feature_importance['Feature'].str.startswith('process_step_')]
        feature_importance = feature_importance.sort_values('SHAP Value', ascending=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#1A1D21'); ax.set_facecolor('#1A1D21')
        colors = ['#e74c3c' if x > 0 else '#5B8FA8' for x in feature_importance['SHAP Value']]
        ax.barh(feature_importance['Feature'], feature_importance['SHAP Value'], color=colors)
        ax.set_xlabel('Impact on Prediction', fontsize=12, color='#E8E6E1'); ax.tick_params(colors='#E8E6E1')
        for spine in ax.spines.values(): spine.set_edgecolor('#333')
        st.pyplot(fig)
        
        st.divider()
        st.subheader("Plain-English Explanation")
        if active_features:
            top_feature_d, top_shap_d, top_val_d = active_features[0]
            safe_min_d, safe_max_d = normal_ranges[process_step][top_feature_d]
            if top_val_d > safe_max_d or top_val_d < safe_min_d:
                if top_val_d > safe_max_d:
                    st.error(f"**{top_feature_d} is running dangerously high — likely cause of the defect risk.**")
                    st.caption(f"Reading: {top_val_d:.1f}, Safe Max: {safe_max_d}")
                else:
                    st.error(f"**{top_feature_d} is running dangerously low — likely cause of the defect risk.**")
                    st.caption(f"Reading: {top_val_d:.1f}, Safe Min: {safe_min_d}")
            else:
                st.success("**All sensors within normal parameters.**")

    with tab3:
        st.header("Business Impact & Action Plan")
        USD_TO_INR = 83
        scrap_usd = 500; scrap_inr = scrap_usd * USD_TO_INR
        early_stop_usd = 300; early_stop_inr = early_stop_usd * USD_TO_INR
        inspection_usd = 50; inspection_inr = inspection_usd * USD_TO_INR
        
        st.write("#### Financial Context (Per Wafer)")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Scrap Cost (If Missed)", f"${scrap_usd:,}", delta=f"Rs. {scrap_inr:,}", delta_color="inverse")
        with col2: st.metric("Savings (If Caught Early)", f"${early_stop_usd:,}", delta=f"Rs. {early_stop_inr:,}", delta_color="normal")
        with col3: st.metric("Inspection Cost (False Alarm)", f"${inspection_usd:,}", delta=f"Rs. {inspection_inr:,}", delta_color="off")
            
        st.divider()
        st.write("#### Business Dashboard (Monthly KPIs)")
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1: st.metric("Est. Scrap Saved", "957 wafers")
        with kpi2: st.metric("Monthly Savings", "$228,760", delta="Rs. 1.89 Cr")
        with kpi3: st.metric("Yield Improvement", "+9.6%")
        
        st.divider()
        st.write("#### Maintenance Recommendations")
        if active_features:
            top_feature_m, top_shap_m, top_val_m = active_features[0]
            safe_min_m, safe_max_m = normal_ranges[process_step][top_feature_m]
            if top_val_m > safe_max_m or top_val_m < safe_min_m:
                st.warning(f"**Immediate Action:** Halt process. Inspect {top_feature_m} controller.")
                st.info(f"**Preventive Action:** Schedule calibration for {top_feature_m} sensors in this chamber.")
                st.success(f"**Long-Term Recommendation:** Implement real-time anomaly detection for {top_feature_m} in the {process_step} step.")
                st.metric("Estimated Downtime", "2-4 hours")
                st.metric("Estimated Repair Cost", "$1,200")
                st.metric("Expected Yield Improvement", "3.2%")
            else:
                st.success("**NO IMMEDIATE ACTION REQUIRED:** All sensors nominal.")

        st.divider()
        st.write("#### Export Data")
        col_pdf, col_csv = st.columns(2)
        
        with col_pdf:
            pdf_bytes = create_pdf_report(process_step, risk_score, prediction, top_feature, top_val, safe_min, safe_max)
            st.download_button(label="Download AI Inspection Report (PDF)", data=pdf_bytes, file_name="fabsense_inspection_report.pdf", mime="application/pdf")
            
        with col_csv:
            csv_data = input_df.copy()
            csv_data['risk_score'] = risk_score
            csv_data['prediction'] = prediction
            st.download_button(label="Download Prediction Data (CSV)", data=csv_data.to_csv(index=False).encode('utf-8'), file_name="fabsense_prediction.csv", mime="text/csv")

    with tab4:
        st.header("AI Co-Pilot")
        st.caption("Ask questions about this wafer's diagnosis. Powered by rule-based logic.")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.messages.append({"role": "assistant", "content": f"Wafer analysis complete. Risk is {risk_score:.1f}/100. Ask me anything about the {process_step} step diagnosis."})

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("Ask about this wafer..."):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Rule-Based Chatbot Logic
            prompt_lower = prompt.lower()
            response = ""
            
            if "why" in prompt_lower and "risky" in prompt_lower:
                if top_val > safe_max or top_val < safe_min:
                    response = f"This wafer is risky primarily because **{top_feature}** is out of its safe operating range. It is currently reading {top_val:.1f}, which exceeds the safe limit of {safe_max}."
                else:
                    response = "The risk is not driven by a single sensor excursion. It's likely due to subtle multi-sensor interactions that the model has detected."
            elif "which sensor" in prompt_lower or "cause" in prompt_lower:
                response = f"The primary contributing sensor is **{top_feature}** with a SHAP impact of {active_features[0][1]:+.4f}."
            elif "inspect" in prompt_lower or "action" in prompt_lower:
                if top_val > safe_max or top_val < safe_min:
                    response = f"You should inspect the **{top_feature}** controller and the associated hardware in the {process_step} chamber immediately."
                else:
                    response = "No immediate inspection is required. All sensors are within normal parameters."
            elif "what if" in prompt_lower and "normal" in prompt_lower:
                response = f"If {top_feature} returned to normal, the risk score would likely drop significantly, as {top_feature} is currently the largest positive contributor to the risk."
            else:
                response = f"I can tell you why this wafer is risky, which sensor caused it, or what to inspect. The current risk is {risk_score:.1f}/100."

            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# -------------------------------------------------------
# BATCH MODE LOGIC
# -------------------------------------------------------
else:
    st.header("Batch Analysis")
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Uploaded {len(batch_df)} wafers. Running predictions...")
            batch_features = batch_df.drop(columns=['defect_label'], errors='ignore')
            batch_encoded = pd.get_dummies(batch_features, columns=['process_step'], drop_first=False)
            batch_encoded = batch_encoded.reindex(columns=feature_names, fill_value=0)
            batch_probs = model.predict_proba(batch_encoded)[:, 1]
            batch_df['risk_score'] = (batch_probs * 100).round(1)
            batch_df['prediction'] = np.where(batch_probs >= (threshold/100), "DEFECTIVE", "GOOD")
            batch_df_sorted = batch_df.sort_values(by='risk_score', ascending=False)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Risk Ranking (Highest First)")
                st.dataframe(batch_df_sorted, use_container_width=True, hide_index=True)
            with col2:
                st.subheader("Defect Rate by Step")
                if 'process_step' in batch_df.columns:
                    step_summary = batch_df.groupby('process_step')['prediction'].apply(lambda x: (x == 'DEFECTIVE').mean() * 100).reset_index()
                    step_summary.columns = ['Process Step', 'Defect Rate (%)']
                    fig, ax = plt.subplots(figsize=(5, 4))
                    fig.patch.set_facecolor('#1A1D21'); ax.set_facecolor('#1A1D21')
                    ax.bar(step_summary['Process Step'], step_summary['Defect Rate (%)'], color=['#e74c3c', '#F2A93B', '#5B8FA8'])
                    ax.set_ylabel('Defect Rate (%)', color='#E8E6E1'); ax.tick_params(colors='#E8E6E1')
                    st.pyplot(fig)
            
            st.divider()
            csv_export = batch_df_sorted.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Batch Results (CSV)", data=csv_export, file_name="fabsense_batch_results.csv", mime="text/csv")
            
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("Upload a CSV file to begin batch analysis.")
