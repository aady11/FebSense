import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

# -------------------------------------------------------
# 1. APP CONFIGURATION & CUSTOM CSS (The UI Magic)
# -------------------------------------------------------
st.set_page_config(page_title="YieldGuard", page_icon="🏭", layout="wide")

# Custom CSS for animations, smooth bars, and premium feel
st.markdown("""
<style>
/* Smooth fade-in for all elements */
.stMarkdown, .stMetric, .stAlert {
    animation: fadeIn 0.6s ease-in-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Pulsing live dot */
.pulsing-dot {
    width: 10px;
    height: 10px;
    background-color: #2ecc71;
    border-radius: 50%;
    display: inline-block;
    animation: pulse 1.5s infinite;
    margin-right: 8px;
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(46, 204, 113, 0.7); }
    70% { box-shadow: 0 0 0 10px rgba(46, 204, 113, 0); }
    100% { box-shadow: 0 0 0 0 rgba(46, 204, 113, 0); }
}

/* Custom Risk Bar */
.risk-bar-background {
    background-color: #e0e0e0;
    border-radius: 20px;
    height: 24px;
    width: 100%;
    overflow: hidden;
    margin-bottom: 20px;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);
}
.risk-bar-fill {
    height: 100%;
    border-radius: 20px;
    transition: width 0.8s ease-in-out, background-color 0.5s ease;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 10px;
    color: white;
    font-weight: bold;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("YieldGuard - Process Intelligence Platform")
st.markdown('<span class="pulsing-dot"></span> **System Active**', unsafe_allow_html=True)
st.markdown("---")

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
            temperature = np.random.normal(75, 3)
            pressure = np.random.normal(5.0, 0.3)
            gas_flow = np.random.normal(120, 8)
            etch_rate = np.random.normal(550, 25)
            voltage = np.random.normal(300, 15)
            current = np.random.normal(6.0, 0.4)
        elif step == 'Deposition':
            temperature = np.random.normal(450, 15)
            pressure = np.random.normal(2.5, 0.2)
            gas_flow = np.random.normal(200, 12)
            etch_rate = 0
            voltage = 0
            current = 0
        else:
            temperature = np.random.normal(22, 0.5)
            pressure = np.random.normal(1.0, 0.05)
            gas_flow = 0
            etch_rate = 0
            voltage = np.random.normal(50, 3)
            current = np.random.normal(2.0, 0.2)
        
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
        data.append({
            'process_step': step,
            'temperature': round(temperature, 2),
            'pressure': round(pressure, 3),
            'gas_flow': round(gas_flow, 1),
            'etch_rate': round(etch_rate, 1),
            'voltage': round(voltage, 1),
            'current': round(current, 2),
            'defect_label': defect_label
        })
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
        etch_rate = 0.0
        voltage = 0.0
        current = 0.0
    else:
        temperature = st.slider("Temperature (C)", 20.0, 24.0, 22.0)
        pressure = st.slider("Pressure (Torr)", 0.9, 1.1, 1.0)
        gas_flow = 0.0
        etch_rate = 0.0
        voltage = st.slider("Voltage (V)", 40.0, 60.0, 50.0)
        current = st.slider("Current (A)", 1.5, 2.5, 2.0)
    
    st.markdown("---")
    st.subheader("Decision Settings")
    threshold = st.slider("Risk Threshold (%)", 10, 90, 30, help="Lower = catch more defects but more false alarms. Higher = fewer false alarms but miss more defects.")
    
    st.markdown("---")
    analyze_button = st.button("Analyze Wafer", type="primary", use_container_width=True)

# -------------------------------------------------------
# 4. MAIN DASHBOARD
# -------------------------------------------------------
if analyze_button:
    input_data = {
        'temperature': temperature,
        'pressure': pressure,
        'gas_flow': gas_flow,
        'etch_rate': etch_rate,
        'voltage': voltage,
        'current': current,
        'process_step_Etch': 1 if process_step == 'Etch' else 0,
        'process_step_Deposition': 1 if process_step == 'Deposition' else 0,
        'process_step_Lithography': 1 if process_step == 'Lithography' else 0
    }
    
    # THE FIX: Force columns to match exactly what the model trained on
    input_df = pd.DataFrame([input_data]).reindex(columns=feature_names, fill_value=0)
    
    prob = model.predict_proba(input_df)[0][1]
    threshold_val = threshold / 100.0
    prediction = "DEFECTIVE" if prob >= threshold_val else "GOOD"
    
    # Top Metrics
    st.subheader("Verdict")
    col1, col2 = st.columns(2)
    
    if prob >= 0.5: risk_label = "HIGH RISK"
    elif prob >= threshold_val: risk_label = "ELEVATED RISK"
    else: risk_label = "LOW RISK"
    
    col1.metric("Defect Probability", f"{prob*100:.1f}%")
    col2.metric("Status", f"{prediction} ({risk_label})")
    
    # Animated Risk Bar
    if prob >= 0.5: bar_color = "#e74c3c"
    elif prob >= threshold_val: bar_color = "#f39c12"
    else: bar_color = "#2ecc71"
    
    st.markdown(f"""
    <div class="risk-bar-background">
        <div class="risk-bar-fill" style="width: {prob*100}%; background-color: {bar_color};">
            {prob*100:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # SHAP Explanation
    st.subheader("Root Cause Analysis")
    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(input_df)
    
    if isinstance(shap_values_raw, list):
        shap_vals = shap_values_raw[1][0]
    else:
        shap_vals = shap_values_raw[0, :, 1]
    
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': shap_vals
    })
    feature_importance = feature_importance[feature_importance['SHAP Value'] != 0]
    feature_importance = feature_importance[~feature_importance['Feature'].str.startswith('process_step_')]
    feature_importance = feature_importance.sort_values('SHAP Value', ascending=True)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#e74c3c' if x > 0 else '#3498db' for x in feature_importance['SHAP Value']]
    ax.barh(feature_importance['Feature'], feature_importance['SHAP Value'], color=colors)
    ax.set_xlabel('Impact on Prediction')
    ax.set_title('Sensor Impact on Defect Risk')
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Detailed Diagnostic Report
    st.subheader("Diagnostic Report & Action Plan")
    
    active_features = [(f, s, input_data[f]) for f, s in zip(feature_names, shap_vals) if f in normal_ranges.get(process_step, {})]
    active_features.sort(key=lambda x: abs(x[1]), reverse=True)
    
    USD_TO_INR = 83
    scrap_usd = 500
    scrap_inr = scrap_usd * USD_TO_INR
    early_stop_usd = 300
    early_stop_inr = early_stop_usd * USD_TO_INR
    inspection_usd = 50
    inspection_inr = inspection_usd * USD_TO_INR
    
    if active_features:
        top_feature, top_shap, top_val = active_features[0]
        safe_min, safe_max = normal_ranges[process_step][top_feature]
        
        if top_val > safe_max or top_val < safe_min:
            with st.expander("⚠️ VIEW CRITICAL ALERT", expanded=True):
                if top_val > safe_max:
                    delta = top_val - safe_max
                    pct = (delta / safe_max) * 100
                    st.error(f"**Primary Cause:** {top_feature} excursion in {process_step} step.")
                    st.markdown(f"**Technical Detail:** Sensor reads **{top_val:.1f}**, which exceeds the safe upper limit of **{safe_max}** by **{delta:.1f} ({pct:.1f}%)**.")
                else:
                    delta = safe_min - top_val
                    pct = (delta / safe_min) * 100
                    st.error(f"**Primary Cause:** {top_feature} excursion in {process_step} step.")
                    st.markdown(f"**Technical Detail:** Sensor reads **{top_val:.1f}**, which is below the safe lower limit of **{safe_min}** by **{delta:.1f} ({pct:.1f}%)**.")
                
                st.markdown("---")
                st.markdown("**Business Impact:**")
                st.markdown(f"- If this wafer is scrapped at end-of-line: **Loss of ${scrap_usd:,} (Rs. {scrap_inr:,})**")
                st.markdown(f"- If caught and stopped now: **Save ${early_stop_usd:,} (Rs. {early_stop_inr:,})**")
                st.markdown(f"- If this is a false alarm: **Inspection cost of ${inspection_usd:,} (Rs. {inspection_inr:,})**")
                
                st.markdown("---")
                st.success(f"**Actionable Recommendation:** Halt processing. Inspect the system controlling **{top_feature}**. Do not proceed to the next step until {top_feature} stabilizes within the {safe_min}-{safe_max} range.")
        else:
            with st.expander("✅ STATUS: NOMINAL", expanded=True):
                st.success("All sensors are operating within normal parameters.")
                st.info("Risk is likely driven by subtle multi-sensor interactions. Continue monitoring, but no immediate action required.")
    else:
        st.info("No active sensors found for this step.")

else:
    st.info("Configure the control panel on the left and click 'Analyze Wafer' to run a diagnostic.")
