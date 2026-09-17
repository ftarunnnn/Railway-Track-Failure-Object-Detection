import os
import io
import json
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from src.pipeline.integrated_pipeline import IntegratedRailwayInspectionPipeline
from src.utils.logger import get_logger

# Streamlit Page Config
st.set_page_config(
    page_title="Railway Track Safety & Object Detection",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #3B82F6;
    }
    .danger-card {
        border-left: 5px solid #EF4444;
    }
    .warning-card {
        border-left: 5px solid #F59E0B;
    }
    .success-card {
        border-left: 5px solid #10B981;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    return IntegratedRailwayInspectionPipeline()

pipeline = load_pipeline()

st.markdown('<div class="main-header">🚆 Railway Track Failure & Object Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Modal AI Dashboard combining Telemetry ML Failure Risk & YOLO Camera Defect Detection</div>', unsafe_allow_html=True)

st.sidebar.header("🕹️ Inspection Telemetry Inputs")
vibration_g = st.sidebar.slider("Vibration Sensor (g-force)", 0.1, 6.0, 2.4, 0.1)
temperature_c = st.sidebar.slider("Rail Temperature (°C)", -10.0, 70.0, 38.0, 0.5)
pressure_psi = st.sidebar.slider("Joint Pressure (PSI)", 2000, 6000, 4200, 50)
maintenance_days = st.sidebar.slider("Days Since Service", 1, 365, 120, 1)
track_stress_index = st.sidebar.slider("Track Stress Index", 0.5, 6.0, 3.1, 0.1)

st.sidebar.markdown("---")
st.sidebar.header("📷 Track Camera Input")
uploaded_file = st.sidebar.file_uploader("Upload Track Inspection Image", type=["jpg", "png", "jpeg"])

# Load sample image if not uploaded
if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")
else:
    sample_path = "data/raw/images/track_0000.jpg"
    if os.path.exists(sample_path):
        input_image = Image.open(sample_path).convert("RGB")
    else:
        input_image = Image.new("RGB", (640, 640), color=(100, 100, 100))

# Execute Multi-Modal Inspection
sensor_inputs = {
    "vibration_g": vibration_g,
    "temperature_c": temperature_c,
    "pressure_psi": pressure_psi,
    "maintenance_days": maintenance_days,
    "track_stress_index": track_stress_index
}

inspection_result = pipeline.inspect(sensor_inputs, input_image)

health_idx = inspection_result["track_health_index"]
status = inspection_result["overall_status"]
sensor_res = inspection_result["sensor_telemetry_analysis"]
visual_res = inspection_result["visual_inspection_analysis"]
annotated_img = inspection_result["annotated_image"]

# Dashboard Layout: Top 4 KPI Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    card_class = "success-card" if health_idx >= 70 else ("warning-card" if health_idx >= 40 else "danger-card")
    st.markdown(f"""
    <div class="metric-card {card_class}">
        <h3>Track Health Index</h3>
        <h1 style="margin:0;">{health_idx:.1f} / 100</h1>
        <p>Overall System Health</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    risk_pct = sensor_res["failure_probability_pct"]
    risk_icon = "🔴" if risk_pct > 70 else ("🟡" if risk_pct > 35 else "🟢")
    st.markdown(f"""
    <div class="metric-card">
        <h3>{risk_icon} Sensor Failure Risk</h3>
        <h1 style="margin:0;">{risk_pct:.1f}%</h1>
        <p>ML Probability Score</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    defect_type = visual_res["defect_type"].upper()
    defect_icon = "🛤️" if defect_type == "NORMAL" else "⚠️"
    st.markdown(f"""
    <div class="metric-card">
        <h3>{defect_icon} Track Defect Type</h3>
        <h2 style="margin:0; color:#1F2937;">{defect_type}</h2>
        <p>CNN Confidence: {visual_res["defect_confidence"]*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    num_objs = visual_res["total_objects_found"]
    obj_icon = "🚧" if num_objs > 0 else "✅"
    st.markdown(f"""
    <div class="metric-card">
        <h3>{obj_icon} Obstacles / Defects</h3>
        <h1 style="margin:0;">{num_objs}</h1>
        <p>YOLO Bounding Boxes</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Main Content Layout: Left = Image & Detection Canvas, Right = Telemetry Trend Graphs
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("🚧 YOLO Bounding Box Defect Detection")
    st.image(annotated_img, caption="Live Track Feed with Localized Bounding Boxes", use_container_width=True)
    
    if visual_res["detected_objects"]:
        st.write("#### 🎯 Detected Bounding Box Coordinates:")
        det_df = pd.DataFrame(visual_res["detected_objects"])
        st.dataframe(det_df, use_container_width=True)
    else:
        st.info("No foreign obstacles or track anomalies detected in camera frame.")

with col_right:
    st.subheader("📈 Real-time & Historical Sensor Telemetry Trends")
    
    # Load processed sensor data for trend graph
    processed_path = "data/processed/cleaned_sensor_data.csv"
    if os.path.exists(processed_path):
        df_sensor = pd.read_csv(processed_path).head(100)
        
        fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax[0].plot(df_sensor.index, df_sensor["vibration_g"], color="#3B82F6", label="Vibration (g)")
        ax[0].axhline(y=vibration_g, color="red", linestyle="--", label="Current Live Input")
        ax[0].set_ylabel("Vibration (g)")
        ax[0].legend(loc="upper right")
        ax[0].set_title("Track Vibration Signal History")
        
        ax[1].plot(df_sensor.index, df_sensor["temperature_c"], color="#EF4444", label="Rail Temp (°C)")
        ax[1].axhline(y=temperature_c, color="blue", linestyle="--", label="Current Live Input")
        ax[1].set_ylabel("Temp (°C)")
        ax[1].set_xlabel("Time Index / Reading")
        ax[1].legend(loc="upper right")
        ax[1].set_title("Rail Temperature History")
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Sensor dataset not found. Please run data collector first.")

st.markdown("---")
st.markdown("🚀 **Railway Track Failure & Object Detection System** | Production Multi-Modal AI Engine")
