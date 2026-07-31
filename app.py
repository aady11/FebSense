import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# -------------------------------------------------------
# 1. APP CONFIGURATION
# -------------------------------------------------------
st.set_page_config(page_title="FabSense", page_icon="🏭", layout="wide")

st.title("FabSense — AI Co-Pilot for Yield Loss")
st.caption("Predict defect risk, identify root causes, and calculate financial impact in USD & INR.")
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
    threshold = st.slider("Risk Threshold (%)", 10, 90, 30, help="Lower = catch more defects but more false alarms. Higher = fewer false alarms but miss more defects.")
    
    st.divider()
    analyze_button = st.button("Analyze Wafer", type="primary", use_container_width=True)

# -------------------------------------------------------
# 4. MAIN DASHBOARD
# -------------------------------------------------------
if analyze_button:
    input_data = {
        'temperature': temperature, 'pressure': pressure, 'gas_flow': gas_flow,
        'etch_rate': etch_rate, 'voltage': voltage, 'current': current,
        'process_step_Etch': 1 if process_step == 'Etch' else 0,
        'process_step_Deposition': 1 if process_step == 'Deposition' else 0,
        'process_step_Lithography': 1 if process_step == 'Lithography' else 0
    }
    
    input_df = pd.DataFrame([input_data]).reindex(columns=feature_names, fill_value=0)
    
    prob = model.predict_proba(input_df)[0][1]
    threshold_val = threshold / 100.0
    prediction = "DEFECTIVE" if prob >= threshold_val else "GOOD"
    
    if prob >= 0.5: risk_label = "HIGH RISK"
    elif prob >= threshold_val: risk_label = "ELEVATED RISK"
    else: risk_label = "LOW RISK"

    # SHAP Calculation (done once to use across tabs)
    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(input_df)
    if isinstance(shap_values_raw, list):
        shap_vals = shap_values_raw[1][0]
    else:
        shap_vals = shap_values_raw[0, :, 1]
    
    active_features = [(f, s, input_data[f]) for f, s in zip(feature_names, shap_vals) if f in normal_ranges.get(process_step, {})]
    active_features.sort(key=lambda x: abs(x[1]), reverse=True)

    # Create Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Verdict & Risk", "🔬 Root Cause Analysis", "💰 Business Impact & Action"])

    with tab1:
        st.header("Wafer Verdict")
        
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            # Plotly Gauge Chart
            if prob >= 0.5: gauge_color = "red"
            elif prob >= threshold_val: gauge_color = "orange"
            else: gauge_color = "green"
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"Status: {risk_label}", 'font': {'size': 24}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': gauge_color, 'thickness': 0.4},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(46, 204, 113, 0.2)'},
                        {'range': [30, 60], 'color': 'rgba(243, 156, 18, 0.2)'},
                        {'range': [60, 100], 'color': 'rgba(231, 76, 60, 0.2)'}],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold_val * 100}
                }
            ))
            fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with col2:
            st.markdown("#### Summary")
            st.metric("Process Step", process_step)
            st.metric("Prediction", prediction)
            st.metric("Risk Threshold", f"{threshold}%")
            st.info(f"The black line on the gauge shows your decision threshold. The model is {prob*100:.1f}% confident this wafer is defective.")

    with tab2:
        st.header("Root Cause Analysis")
        
        # Sensor Status Table
        st.subheader("Sensor Status Table")
        sensor_table_data = []
        for f, s, v in active_features:
            safe_min, safe_max = normal_ranges[process_step][f]
            status = "🟢 Normal" if safe_min <= v <= safe_max else "🔴 Out of Spec"
            sensor_table_data.append({
                'Sensor': f, 'Reading': f"{v:.2f}", 
                'Safe Range': f"{safe_min} - {safe_max}", 
                'Status': status, 'SHAP Impact': f"{s:+.4f}"
            })
        
        df_sensors = pd.DataFrame(sensor_table_data)
        st.dataframe(df_sensors, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # SHAP Chart
        st.subheader("Feature Impact (SHAP)")
        feature_importance = pd.DataFrame({'Feature': feature_names, 'SHAP Value': shap_vals})
        feature_importance = feature_importance[feature_importance['SHAP Value'] != 0]
        feature_importance = feature_importance[~feature_importance['Feature'].str.startswith('process_step_')]
        feature_importance = feature_importance.sort_values('SHAP Value', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#e74c3c' if x > 0 else '#3498db' for x in feature_importance['SHAP Value']]
        ax.barh(feature_importance['Feature'], feature_importance['SHAP Value'], color=colors)
        ax.set_xlabel('Impact on Prediction (SHAP Value)', fontsize=12)
        ax.set_title('Sensor Impact on Defect Risk', fontsize=14)
        st.pyplot(fig)
        
        st.divider()
        st.subheader("Plain-English Explanation")
        if active_features:
            top_feature, top_shap, top_val = active_features[0]
            safe_min, safe_max = normal_ranges[process_step][top_feature]
            
            if top_val > safe_max or top_val < safe_min:
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
            else:
                st.success("All sensors are operating within normal parameters.")
                st.info("Risk is likely driven by subtle multi-sensor interactions. Continue monitoring, but no immediate action required.")

    with tab3:
        st.header("Business Impact & Action Plan")
        
        USD_TO_INR = 83
        scrap_usd = 500
        scrap_inr = scrap_usd * USD_TO_INR
        early_stop_usd = 300
        early_stop_inr = early_stop_usd * USD_TO_INR
        inspection_usd = 50
        inspection_inr = inspection_usd * USD_TO_INR
        
        st.write("#### Financial Context (Per Wafer)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Scrap Cost (If Missed)", f"${scrap_usd:,}", delta=f"Rs. {scrap_inr:,}", delta_color="inverse")
        with col2:
            st.metric("Savings (If Caught Early)", f"${early_stop_usd:,}", delta=f"Rs. {early_stop_inr:,}", delta_color="normal")
        with col3:
            st.metric("Inspection Cost (False Alarm)", f"${inspection_usd:,}", delta=f"Rs. {inspection_inr:,}", delta_color="off")
            
        st.divider()
        st.write("#### Actionable Recommendation")
        
        if active_features:
            top_feature, top_shap, top_val = active_features[0]
            safe_min, safe_max = normal_ranges[process_step][top_feature]
            
            if top_val > safe_max or top_val < safe_min:
                st.warning(f"**IMMEDIATE ACTION REQUIRED:** Halt processing. Inspect the system controlling **{top_feature}**.")
                st.markdown(f"- Do not proceed to the next step until {top_feature} stabilizes within the **{safe_min}-{safe_max}** range.")
                st.markdown(f"- Catching this now saves **${early_stop_usd:,} (Rs. {early_stop_inr:,})** vs. scrapping at end-of-line.")
            else:
                st.success("**NO IMMEDIATE ACTION REQUIRED:** All sensors nominal. Proceed to next step.")
                st.markdown("- Continue standard monitoring protocols.")

else:
    st.markdown("### Welcome to FabSense")
    st.info("Configure the control panel on the left and click **Analyze Wafer** to run a diagnostic.")
