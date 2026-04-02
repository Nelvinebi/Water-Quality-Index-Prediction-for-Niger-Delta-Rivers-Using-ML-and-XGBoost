"""
app/app.py — Streamlit Web App for Water Quality Index Prediction
Niger Delta Rivers — ML & XGBoost Project

HOW TO RUN:
    conda activate water_quality
    cd Water_Quality_ML
    streamlit run app/app.py
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Niger Delta WQI Predictor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: linear-gradient(135deg, #0a0f2e 0%, #0d1b4b 40%, #0f2460 70%, #1a3a7a 100%) !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060d28 0%, #0a1535 100%) !important;
    border-right: 1px solid rgba(100,160,255,0.15) !important;
}
[data-testid="stSidebar"] * { color: #c8d8f8 !important; }
h1, h2, h3, h4, h5, h6 { color: #e8f0ff !important; font-family: 'Inter', sans-serif !important; font-weight: 600 !important; }
p, li, span, div, label { color: #c8d8f8 !important; font-family: 'Inter', sans-serif !important; }
.stMarkdown p { color: #b8ccf4 !important; }
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(100,160,255,0.2) !important;
    border-radius: 12px !important; padding: 16px !important;
}
[data-testid="stMetricValue"] { color: #7dd3fc !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #94b8e8 !important; }
[data-testid="stMetricDelta"] { color: #34d399 !important; }
[data-testid="stForm"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(100,160,255,0.15) !important;
    border-radius: 16px !important; padding: 20px !important;
}
.stSelectbox > div > div, .stNumberInput > div > div > input {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(100,160,255,0.25) !important;
    color: #e8f0ff !important; border-radius: 8px !important;
}
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg, #1e5aba, #2563eb) !important;
    color: #ffffff !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(37,99,235,0.35) !important;
}
[data-testid="stAlert"] { border-radius: 10px !important; background: rgba(255,255,255,0.06) !important; }
details { background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(100,160,255,0.15) !important; border-radius: 10px !important; }
summary { color: #90b8f8 !important; }
hr { border-color: rgba(100,160,255,0.15) !important; }
.stCaption { color: #6888b8 !important; }
</style>
""", unsafe_allow_html=True)

MODEL_PATH  = "models/model.pkl"
SCALER_PATH = "models/scaler.pkl"
DATA_PATH   = "data/Niger_Delta_Water_Quality_Enriched.xlsx"
SHAP_PATH   = "outputs/shap_summary.png"
MAP_PATH    = "outputs/pollution_map.html"

# WHO professional WQI scale (0–100, higher = better)
WQI_BINS   = [0, 44, 64, 79, 94, 100]
WQI_LABELS = ["Hazardous", "Poor", "Fair", "Good", "Excellent"]
WQI_COLORS = {
    "Excellent": "#1D9E75",  # green
    "Good":      "#378ADD",  # blue
    "Fair":      "#F0C427",  # yellow
    "Poor":      "#EF9F27",  # orange
    "Hazardous": "#E24B4A",  # red
}

FEATURE_ORDER = [
    "pH","Temperature_C","Turbidity_NTU","Electrical_Conductivity_uS_cm",
    "TDS_mg_L","Dissolved_Oxygen_mg_L","BOD_mg_L","Nitrate_mg_L",
    "Phosphate_mg_L","Iron_Fe_mg_L","Lead_Pb_mg_L","Zinc_Zn_mg_L",
    "Cadmium_Cd_mg_L","Chromium_Cr_mg_L","Total_Coliform_CFU_100mL",
    "E_coli_CFU_100mL","Proximity_To_Settlement_km","Oil_Spill_History",
    "Oil_Spill_Count","Land_Use_Type","Collection_Month","Collection_Year",
    "River_Zone_Encoded","State_Bayelsa","State_Delta","State_Imo",
    "State_Rivers","Heavy_Metal_Index","Oxygen_Stress","Microbial_Risk",
    "Human_Impact","Season","Latitude","Longitude",
]
LAND_USE_MAP = {"Mangrove":0,"Agricultural":1,"Urban":2,"Industrial":3,"Riverine Forest":4}
ZONE_MAP     = {"Upstream":0,"Midstream":1,"Downstream":2}
STATE_LIST   = ["Bayelsa","Delta","Imo","Rivers"]

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)

@st.cache_data
def load_data():
    return pd.read_excel(DATA_PATH)

def wqi_category(score):
    # WHO 5-tier scale: bins are [0, 44, 64, 79, 94, 100]
    # Labels:           Hazardous, Poor, Fair, Good, Excellent
    for i, label in enumerate(WQI_LABELS):
        if score <= WQI_BINS[i + 1]:
            return label
    return "Excellent"

def styled_fig(w=6, h=3.5):
    fig, ax = plt.subplots(figsize=(w, h), facecolor="none")
    ax.set_facecolor("#0d1b4b")
    ax.tick_params(colors="#94b8e8")
    ax.xaxis.label.set_color("#94b8e8")
    ax.yaxis.label.set_color("#94b8e8")
    ax.title.set_color("#e8f0ff")
    for sp in ax.spines.values(): sp.set_edgecolor("#1a3a7a")
    return fig, ax

# Sidebar
st.sidebar.markdown("## 💧 Niger Delta WQI")
st.sidebar.markdown("*ML-powered Water Quality Prediction*")
st.sidebar.divider()
page = st.sidebar.radio("Navigate", [
    "🔬  WQI Predictor","📊  Model Performance",
    "🧠  SHAP Feature Importance","🗺️  Pollution Map","📂  Dataset Explorer"
], label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.markdown("**Model:** XGBoost Regressor")
st.sidebar.markdown("**Test R²:** `0.9704`  |  **RMSE:** `1.29`")
st.sidebar.markdown("**Data:** Niger Delta Rivers, 2019–2023")
st.sidebar.markdown("**Stations:** 35 sampling points")
st.sidebar.divider()
st.sidebar.markdown("---")
st.sidebar.markdown("*Created by **Agbozu Ebingiye Nelvin***")

# ── PAGE 1: WQI PREDICTOR ──────────────────────────────────────────────────────
if page == "🔬  WQI Predictor":
    st.title("🔬 Water Quality Index Predictor")
    st.markdown("Enter water sample parameters to get an instant **WQI prediction** from the XGBoost model. Scale: 0–100 — higher = better quality.")
    st.divider()
    model, scaler = load_model()
    with st.form("prediction_form"):
        st.markdown("### 📍 Sample Location")
        c1,c2,c3 = st.columns(3)
        with c1:
            state      = st.selectbox("State", STATE_LIST)
            river_zone = st.selectbox("River Zone", ["Upstream","Midstream","Downstream"])
        with c2:
            land_use   = st.selectbox("Land Use Type", list(LAND_USE_MAP.keys()))
            oil_history= st.selectbox("Oil Spill History", ["No","Yes"])
        with c3:
            oil_count  = st.number_input("Oil Spill Count", 0, 50, 0)
            prox       = st.slider("Proximity to Settlement (km)", 0.2, 25.0, 5.0)
        st.markdown("### 📅 Sampling Period")
        c1,c2,c3 = st.columns(3)
        with c1: month  = st.selectbox("Month", list(range(1,13)), format_func=lambda m: pd.Timestamp(2024,m,1).strftime("%B"))
        with c2: year   = st.selectbox("Year", [2019,2020,2021,2022,2023,2024])
        with c3: season = st.selectbox("Season", ["Dry","Wet"])
        st.markdown("### 🧪 Physical Parameters")
        c1,c2,c3 = st.columns(3)
        with c1:
            ph   = st.slider("pH", 5.5, 8.5, 7.0, 0.01)
            temp = st.slider("Temperature (°C)", 24.0, 34.0, 28.0, 0.1)
        with c2:
            turb = st.slider("Turbidity (NTU)", 0.0, 300.0, 50.0, 0.5)
            ec   = st.slider("Conductivity (µS/cm)", 500.0, 2500.0, 1200.0, 1.0)
        with c3:
            tds  = st.slider("TDS (mg/L)", 200.0, 1500.0, 700.0, 1.0)
        st.markdown("### ⚗️ Chemical Parameters")
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            do_val = st.slider("Dissolved O₂ (mg/L)", 2.0, 9.0, 5.5, 0.01)
            bod    = st.slider("BOD (mg/L)", 0.5, 15.0, 4.0, 0.1)
        with c2:
            nit  = st.slider("Nitrate (mg/L)", 0.5, 25.0, 8.0, 0.1)
            phos = st.slider("Phosphate (mg/L)", 0.0, 5.0, 1.5, 0.01)
        with c3:
            iron = st.slider("Iron Fe (mg/L)", 0.05, 5.0, 0.5, 0.01)
            lead = st.slider("Lead Pb (mg/L)", 0.001, 0.15, 0.02, 0.001)
        with c4:
            zinc = st.slider("Zinc Zn (mg/L)", 0.01, 3.0, 0.5, 0.001)
            cd   = st.slider("Cadmium Cd (mg/L)", 0.0001, 0.02, 0.003, 0.0001)
            cr   = st.slider("Chromium Cr (mg/L)", 0.001, 0.1, 0.02, 0.001)
        st.markdown("### 🦠 Biological Parameters")
        c1,c2 = st.columns(2)
        with c1: coli  = st.number_input("Total Coliform (CFU/100mL)", 0, 50000, 500)
        with c2: ecoli = st.number_input("E. coli (CFU/100mL)", 0, 20000, 150)
        st.markdown("### 📡 GPS Coordinates")
        c1,c2 = st.columns(2)
        with c1: lat = st.number_input("Latitude",  4.0, 6.5, 5.0, 0.0001, format="%.4f")
        with c2: lon = st.number_input("Longitude", 5.0, 8.0, 6.5, 0.0001, format="%.4f")
        submitted = st.form_submit_button("🔍 Predict WQI", use_container_width=True)
    if submitted:
        # Calculate engineered features
        hmi = lead + cd + cr
        oxy = bod / (do_val + 1e-5)
        mic = coli + ecoli
        him = oil_count + 1/(prox+1)
        
        # Build input dict with ALL features in EXACT order scaler expects
        inp = {
            'Latitude': lat,
            'Longitude': lon,
            'Season': int(season == "Wet"),
            'Proximity_To_Settlement_km': prox,
            'Oil_Spill_History': int(oil_history == "Yes"),
            'Oil_Spill_Count': oil_count,
            'Land_Use_Type': LAND_USE_MAP[land_use],
            'pH': ph,
            'Temperature_C': temp,
            'Turbidity_NTU': turb,
            'Electrical_Conductivity_uS_cm': ec,
            'TDS_mg_L': tds,
            'Dissolved_Oxygen_mg_L': do_val,
            'BOD_mg_L': bod,
            'Nitrate_mg_L': nit,
            'Phosphate_mg_L': phos,
            'Iron_Fe_mg_L': iron,
            'Lead_Pb_mg_L': lead,
            'Zinc_Zn_mg_L': zinc,
            'Cadmium_Cd_mg_L': cd,
            'Chromium_Cr_mg_L': cr,
            'Total_Coliform_CFU_100mL': coli,
            'E_coli_CFU_100mL': ecoli,
            'Collection_Month': month,
            'Collection_Year': year,
            'River_Zone_Encoded': ZONE_MAP[river_zone],
            'State_Bayelsa': int(state == "Bayelsa"),
            'State_Delta': int(state == "Delta"),
            'State_Imo': int(state == "Imo"),
            'State_Rivers': int(state == "Rivers"),
            'Heavy_Metal_Index': hmi,
            'Oxygen_Stress': oxy,
            'Microbial_Risk': mic,
            'Human_Impact': him,
        }
        
        # Create DataFrame in exact order
        Xdf = pd.DataFrame([inp])
        
        # Verify order matches
        expected_order = [
            'Latitude', 'Longitude', 'Season', 'Proximity_To_Settlement_km', 
            'Oil_Spill_History', 'Oil_Spill_Count', 'Land_Use_Type', 'pH', 
            'Temperature_C', 'Turbidity_NTU', 'Electrical_Conductivity_uS_cm', 
            'TDS_mg_L', 'Dissolved_Oxygen_mg_L', 'BOD_mg_L', 'Nitrate_mg_L', 
            'Phosphate_mg_L', 'Iron_Fe_mg_L', 'Lead_Pb_mg_L', 'Zinc_Zn_mg_L', 
            'Cadmium_Cd_mg_L', 'Chromium_Cr_mg_L', 'Total_Coliform_CFU_100mL', 
            'E_coli_CFU_100mL', 'Collection_Month', 'Collection_Year', 
            'River_Zone_Encoded', 'State_Bayelsa', 'State_Delta', 'State_Imo', 
            'State_Rivers', 'Heavy_Metal_Index', 'Oxygen_Stress', 
            'Microbial_Risk', 'Human_Impact'
        ]
        
        # Reorder to match exactly
        Xdf = Xdf[expected_order]
        
        # Predict
        X_scaled = scaler.transform(Xdf)
        wqi = float(np.clip(model.predict(X_scaled)[0], 0, 100))
        
        # Continue with display code...
        cat = wqi_category(wqi)
        col = WQI_COLORS[cat]
        # ... rest of your display code
        # avail = [f for f in FEATURE_ORDER if f in inp]
        # Xdf   = pd.DataFrame([[inp[f] for f in avail]], columns=avail)
        # wqi   = float(np.clip(model.predict(scaler.transform(Xdf))[0], 0, 100))
        # cat   = wqi_category(wqi)
        # col   = WQI_COLORS[cat]
        st.divider()
        st.markdown("### 📋 Prediction Result")
        c1,c2,c3 = st.columns(3)
        c1.metric("WQI Score", f"{wqi:.2f} / 100")
        c2.metric("Category",  cat)
        c3.metric("River Zone", river_zone)
        fig, ax = plt.subplots(figsize=(9,1.3), facecolor="none")
        ax.set_facecolor("#0a1535")
        ax.barh(0, 100, color="#1a2a5e", height=0.5)
        ax.barh(0, wqi,  color=col,      height=0.5)
        ax.set_xlim(0,100); ax.set_yticks([])
        ax.tick_params(colors="#94b8e8")
        for sp in ax.spines.values(): sp.set_edgecolor("none")
        for b,l,c in zip([0,44,64,79,94], WQI_LABELS, [WQI_COLORS[x] for x in WQI_LABELS]):
            ax.axvline(b, color=c, linestyle="--", alpha=0.4, linewidth=1)
            ax.text(b+1, 0.32, l, fontsize=7, color=c)
        ax.set_title(f"WQI: {wqi:.1f}  —  {cat}", fontsize=11, color=col, fontweight="bold")
        fig.tight_layout(); st.pyplot(fig); plt.close()
        msgs = {
            "Excellent": "✅ Water quality is **excellent** (WQI 95–100). Near-pristine. Safe for drinking and aquatic life.",
            "Good":      "🟦 Water quality is **good** (WQI 80–94). Low threat. Suitable for most uses with standard treatment.",
            "Fair":      "🟨 Water quality is **fair** (WQI 65–79). Minor impairment detected. Treatment required before use.",
            "Poor":      "🟧 Water quality is **poor** (WQI 45–64). Significant pollution. Not suitable for direct use.",
            "Hazardous": "🔴 Water quality is **hazardous** (WQI 0–44). Severely polluted. Immediate intervention required.",
        }
        st.info(msgs[cat])
        with st.expander("🔎 View engineered features"):
            st.dataframe(pd.DataFrame({
                "Feature":["Heavy Metal Index","Oxygen Stress","Microbial Risk","Human Impact"],
                "Value":[f"{hmi:.4f}",f"{oxy:.4f}",f"{mic}",f"{him:.4f}"]
            }), use_container_width=True, hide_index=True)

# ── PAGE 2: MODEL PERFORMANCE ──────────────────────────────────────────────────
elif page == "📊  Model Performance":
    st.title("📊 Model Performance Dashboard")
    st.markdown("Comparison of 3 models trained with **5-fold cross-validation** on the Niger Delta dataset.")
    st.divider()
    results = {
        "Model":    ["Linear Regression","Random Forest","XGBoost"],
        "CV RMSE":  [2.3130,2.1358,1.4657], "CV RMSE ±":[0.6778,0.7191,0.3241],
        "CV R²":    [0.9026,0.9172,0.9609], "CV R² ±":  [0.0149,0.0137,0.0053],
        "Test RMSE":[2.3476,1.9125,1.2867], "Test MAE": [1.9089,1.4623,0.9475],
        "Test R²":  [0.9016,0.9347,0.9704],
    }
    df_r = pd.DataFrame(results)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Best Model","XGBoost")
    c2.metric("Test R²","0.9704","+0.036 vs Random Forest")
    c3.metric("Test RMSE","1.2867","-0.626 vs Random Forest")
    c4.metric("CV Strategy","5-fold KFold")
    st.divider()
    st.subheader("Comparison Table")
    st.dataframe(df_r.style
        .highlight_min(subset=["CV RMSE","Test RMSE","Test MAE"], color="#3b1a1a")
        .highlight_max(subset=["CV R²","Test R²"],               color="#0f2d1f")
        .format({"CV RMSE":"%.4f","CV RMSE ±":"%.4f","CV R²":"%.4f",
                 "CV R² ±":"%.4f","Test RMSE":"%.4f","Test MAE":"%.4f","Test R²":"%.4f"}),
        use_container_width=True, hide_index=True)
    st.divider()
    st.subheader("Visual Comparison")
    c1,c2 = st.columns(2)
    with c1:
        fig, ax = styled_fig()
        bars = ax.bar(df_r["Model"], df_r["Test R²"], color=["#3b6bbf","#5ba89a","#1D9E75"], width=0.5)
        ax.errorbar(df_r["Model"], df_r["CV R²"], yerr=df_r["CV R² ±"],
                    fmt="o", color="#e8f0ff", capsize=5, label="CV R² ± std")
        ax.set_ylim(0.88,1.0); ax.set_ylabel("R² Score"); ax.set_title("Test R² vs CV R²")
        ax.legend(fontsize=8, facecolor="#0a1535", labelcolor="#c8d8f8")
        for bar,v in zip(bars, df_r["Test R²"]):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.001, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=8, color="#e8f0ff")
        fig.tight_layout(); st.pyplot(fig); plt.close()
    with c2:
        fig, ax = styled_fig()
        bars = ax.bar(df_r["Model"], df_r["Test RMSE"], color=["#7a3535","#8a6a20","#1D9E75"], width=0.5)
        ax.errorbar(df_r["Model"], df_r["CV RMSE"], yerr=df_r["CV RMSE ±"],
                    fmt="o", color="#e8f0ff", capsize=5, label="CV RMSE ± std")
        ax.set_ylabel("RMSE"); ax.set_title("Test RMSE vs CV RMSE")
        ax.legend(fontsize=8, facecolor="#0a1535", labelcolor="#c8d8f8")
        for bar,v in zip(bars, df_r["Test RMSE"]):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=8, color="#e8f0ff")
        fig.tight_layout(); st.pyplot(fig); plt.close()
    st.divider()
    c1,c2,c3 = st.columns(3)
    c1.info("**R²** — XGBoost explains **97%** of WQI variance.")
    c2.info("**RMSE** — XGBoost is off by only **±1.29 WQI points** on average.")
    c3.info("**CV ± std** — XGBoost's low std (**±0.32**) means very consistent results.")

# ── PAGE 3: SHAP ───────────────────────────────────────────────────────────────
elif page == "🧠  SHAP Feature Importance":
    st.title("🧠 SHAP Feature Importance")
    st.markdown(
        "SHAP reveals **which parameters drive WQI predictions** and in which direction. "
        "Each dot is one water sample — 🔴 red = high feature value, 🔵 blue = low. "
        "X-axis shows impact on the predicted WQI score."
    )
    st.divider()
    if os.path.exists(SHAP_PATH):
        st.image(SHAP_PATH, use_column_width=True)   # use_column_width works on all versions
    else:
        st.warning("SHAP plot not found. Run `python main.py --run` first.")
    st.divider()
    st.subheader("How to read this plot")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("""
**Top features** have the largest overall impact on WQI.

**Dot colour:**
- 🔴 Red = high value in that sample
- 🔵 Blue = low value in that sample

**X-axis position:**
- Positive (+) = pushes WQI **higher** (cleaner water)
- Negative (−) = pushes WQI **lower** (more polluted)
        """)
    with c2:
        st.markdown("""
**Key findings from your Niger Delta model:**

- 💧 **Dissolved O₂** — high DO strongly improves WQI ✅
- ⚠️ **Oxygen Stress** — high BOD/DO ratio sharply reduces WQI
- ☠️ **Cadmium & Lead** — degrade quality even at trace levels
- 🏘️ **Proximity to Settlement** — closer to towns = lower WQI
- 🛢️ **Heavy Metal Index** — composite metal pollution shows clear negative impact
        """)

# ── PAGE 4: POLLUTION MAP ──────────────────────────────────────────────────────
elif page == "🗺️  Pollution Map":
    st.title("🗺️ Niger Delta Pollution Map")
    st.markdown("Interactive heatmap of **WQI scores** across 35 sampling stations. Click any marker for station details.")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.success("🟢 Excellent — WQI 95–100")
    c2.info("🔵 Good — WQI 80–94")
    c3.warning("🟡 Fair — WQI 65–79")
    c4.warning("🟠 Poor — WQI 45–64")
    c5.error("🔴 Hazardous — WQI 0–44")
    st.divider()
    if os.path.exists(MAP_PATH):
        with open(MAP_PATH, "r", encoding="utf-8") as f:
            html_data = f.read()
        components.html(html_data, height=580, scrolling=False)
    else:
        st.warning("Pollution map not found. Run `python main.py --run` first.")

# ── PAGE 5: DATASET EXPLORER ───────────────────────────────────────────────────
elif page == "📂  Dataset Explorer":
    st.title("📂 Dataset Explorer")
    st.markdown("Filter, explore, and analyse the Niger Delta water quality dataset.")
    st.divider()
    df = load_data()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Samples",     f"{len(df):,}")
    c2.metric("Sampling Stations", df["Station_Name"].nunique() if "Station_Name" in df.columns else "N/A")
    c3.metric("States Covered",    df["State"].nunique() if "State" in df.columns else "N/A")
    c4.metric("Date Range",        "2019 – 2023")
    st.divider()
    st.subheader("🔽 Filters")
    c1,c2,c3 = st.columns(3)
    with c1:
        states    = ["All"] + sorted(df["State"].unique().tolist()) if "State" in df.columns else ["All"]
        sel_state = st.selectbox("State", states)
    with c2:
        zones    = ["All"] + sorted(df["River_Zone"].unique().tolist()) if "River_Zone" in df.columns else ["All"]
        sel_zone = st.selectbox("River Zone", zones)
    with c3:
        seasons    = ["All"] + sorted(df["Season"].unique().tolist()) if "Season" in df.columns else ["All"]
        sel_season = st.selectbox("Season", seasons)
    df_f = df.copy()
    if sel_state  != "All" and "State"      in df.columns: df_f = df_f[df_f["State"]      == sel_state]
    if sel_zone   != "All" and "River_Zone" in df.columns: df_f = df_f[df_f["River_Zone"] == sel_zone]
    if sel_season != "All" and "Season"     in df.columns: df_f = df_f[df_f["Season"]     == sel_season]
    st.caption(f"Showing {len(df_f):,} of {len(df):,} samples")
    st.dataframe(df_f, use_container_width=True, height=320)
    st.divider()
    st.subheader("📈 Parameter Distribution")
    num_cols  = df_f.select_dtypes(include=np.number).columns.tolist()
    sel_param = st.selectbox("Select parameter", num_cols,
                             index=num_cols.index("Dissolved_Oxygen_mg_L") if "Dissolved_Oxygen_mg_L" in num_cols else 0)
    c1,c2 = st.columns(2)
    with c1:
        fig, ax = styled_fig()
        ax.hist(df_f[sel_param].dropna(), bins=30, color="#378ADD", edgecolor="#0a1535", alpha=0.9)
        ax.set_xlabel(sel_param); ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {sel_param}")
        fig.tight_layout(); st.pyplot(fig); plt.close()
    with c2:
        if "River_Zone" in df_f.columns:
            fig, ax = styled_fig()
            zc = {"Upstream":"#1D9E75","Midstream":"#378ADD","Downstream":"#E24B4A"}
            for zone, grp in df_f.groupby("River_Zone"):
                ax.hist(grp[sel_param].dropna(), bins=20, alpha=0.65,
                        label=zone, color=zc.get(zone,"#888"))
            ax.set_xlabel(sel_param); ax.set_ylabel("Count")
            ax.set_title(f"{sel_param} by River Zone")
            ax.legend(fontsize=8, facecolor="#0a1535", labelcolor="#c8d8f8")
            fig.tight_layout(); st.pyplot(fig); plt.close()
    st.divider()
    st.subheader("🔗 Correlation Heatmap")
    core = [c for c in ["pH","Dissolved_Oxygen_mg_L","BOD_mg_L","Turbidity_NTU","TDS_mg_L",
                        "Nitrate_mg_L","Phosphate_mg_L","Lead_Pb_mg_L","Cadmium_Cd_mg_L",
                        "Total_Coliform_CFU_100mL"] if c in df_f.columns]
    if core:
        corr  = df_f[core].corr()
        short = [c.replace("_mg_L","").replace("_NTU","").replace("_CFU_100mL","")
                  .replace("Dissolved_Oxygen","DO").replace("Total_Coliform","Coliform") for c in core]
        fig, ax = plt.subplots(figsize=(9,6), facecolor="none")
        ax.set_facecolor("#0d1b4b")
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(core))); ax.set_yticks(range(len(core)))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8, color="#94b8e8")
        ax.set_yticklabels(short, fontsize=8, color="#94b8e8")
        for i in range(len(core)):
            for j in range(len(core)):
                ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(corr.iloc[i,j])>0.5 else "#c8d8f8")
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Parameter Correlation Matrix", color="#e8f0ff")
        fig.tight_layout(); st.pyplot(fig); plt.close()
    st.divider()
    csv = df_f.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Filtered Data as CSV", csv,
                       file_name="niger_delta_wq_filtered.csv",
                       mime="text/csv", use_container_width=True)
