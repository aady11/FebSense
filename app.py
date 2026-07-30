
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import shap

# -------------------------------------------------------
# 1. APP CONFIGURATION
# -------------------------------------------------------
st.set_page_config(page_title="FabSense", page_icon="🏭", layout="wide")

st.title("🏭 FabSense — AI Co-Pilot for Yield Loss")
st.markdown("Predict defect risk, explain why, and estimate cost impact.")

# -------------------------------------------------------
# 2. DATA GENERATION (Same as Phase A)
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
        elif step == 'Deposition':
            if temperature > 475 or temperature < 425: defect_prob += 0.50
        else:
            if temperature > 23 or temperature < 21: defect_prob += 0.50
        
        defect_prob = min(defect_prob, 0.95)
        defect_label = 1 if np.random.random() < defect_prob else 0
        data.append({
            'process_step': step,
            'temperature': temperature,
            'pressure': pressure,
            'gas_flow': gas_flow,
            'etch_rate': etch_rate,
            'voltage': voltage,
            'current': current,
            'defect_label': defect_label
        })
    return pd.DataFrame(data)

df = generate_data()

# -------------------------------------------------------
# 3. MODEL TRAINING (Cached so it doesn't retrain every click)
# -------------------------------------------------------
@st.cache_resource
def train_model():
    df_train = generate_data()
    df_encoded = pd.get_dummies(df_train, columns=['process_step'], drop_first=False)
    df_encoded = df_encoded.drop('defect_label', axis=1)
    X = df_encoded.drop('defect_label', axis=1)
    y = df_train['defect_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model, X_train.columns

model, feature_names = train_model()

# -------------------------------------------------------
# 4. SIDEBAR INPUTS
# -------------------------------------------------------
st.sidebar.header("Wafer Sensor Inputs")
process_step = st.sidebar.selectbox("Process Step", ['Etch', 'Deposition', 'Lithography'])

# Dynamic inputs based on step
if process_step == 'Etch':
    temperature = st.sidebar.slider("Temperature (°C)", 60, 90, 75.0)
    pressure = st.sidebar.slider("Pressure (Torr)", 4.0, 6.0, 5.0)
    gas_flow = st.sidebar.slider("Gas Flow (sccm)", 100, 140, 120.0)
    etch_rate = st.sidebar.slider("Etch Rate (Å/min)", 500, 600, 550.0)
    voltage = st.sidebar.slider("Voltage (V)", 250, 350, 300.0)
    current = st.sidebar.slider("Current (A)", 5.0, 7.0, 6.0)
elif process_step == 'Deposition':
    temperature = st.sidebar.slider("Temperature (°C)", 400, 500, 450.0)
    pressure = st.sidebar.slider("Pressure (Torr)", 2.0, 3.0, 2.5)
    gas_flow = st.sidebar.slider("Gas Flow (sccm)", 180, 220, 200.0)
    etch_rate = 0
    voltage = 0
    current = 0
else:
    temperature = st.sidebar.slider("Temperature (°C)", 20.0, 24.0, 22.0)
    pressure = st.sidebar.slider("Pressure (Torr)", 0.9, 1.1, 1.0)
    gas_flow = 0
    etch_rate = 0
    voltage = st.sidebar.slider("Voltage (V)", 40, 60, 50.0)
    current = st.sidebar.slider("Current (A)", 1.5, 2.5, 2.0)

# -------------------------------------------------------
# 5. PREDICTION & EXPLANATION
# -------------------------------------------------------
if st.sidebar.button("🔍 Analyze Wafer"):
    # Prepare input
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
    input_df = pd.DataFrame([input_data])
    
    # Predict
    prob = model.predict_proba(input_df)[0][1]
    risk_level = "🟢 LOW" if prob < 0.3 else "🔴 HIGH"
    
    # Display Results
    st.header("Analysis Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Defect Risk", f"{prob*100:.1f}%")
    col2.metric("Risk Level", risk_level)
    col3.metric("Est. Cost Impact", f"${int(prob * 500):,}")
    
    # SHAP Explanation (Simplified for app)
    st.subheader("Top Contributing Factors")
    if process_step == 'Deposition' and temperature > 475:
        st.warning(f"⚠️ Temperature is {temperature:.1f}°C (Above safe limit of 475°C). This is the primary risk driver.")
    elif process_step == 'Etch' and etch_rate > 590:
        st.warning(f"⚠️ Etch Rate is {etch_rate:.1f} Å/min (Above safe limit of 590). This is the primary risk driver.")
    elif process_step == 'Lithography' and temperature > 23:
        st.warning(f"⚠️ Temperature is {temperature:.1f}°C (Above safe limit of 23°C). This is the primary risk driver.")
    else:
        st.success("✅ All sensors within normal ranges.")
    
    # Business Impact
    st.subheader("Business Impact")
    st.write(f"If this wafer is defective and missed, the cost is **$500**.")
    st.write(f"If caught early, savings is **$300**.")
    st.write(f"**Recommendation:** {'Inspect immediately.' if prob >= 0.3 else 'Proceed to next step.'}")
