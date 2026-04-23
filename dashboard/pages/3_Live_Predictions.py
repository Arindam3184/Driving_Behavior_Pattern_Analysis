"""
Live Predictions Page — Light Admin Theme
===========================================
Interactive prediction interface with sliders, presets, and visual results.
"""


import sys
import os
import streamlit as st
import plotly.graph_objects as go
from src.predictor import DrivingBehaviorPredictor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Live Predictions", page_icon="🎯", layout="wide")

# ─── Theme CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif !important; }
    .stApp { background-color: #E8EDEB !important; }
    [data-testid="stSidebar"] { background: #2D3436 !important; }
    [data-testid="stSidebar"] * { color: #B2BEC3 !important; }
    [data-testid="stSidebar"] h2 { color: #FFFFFF !important; }
    h1 { color: #2D3436 !important; font-weight: 800 !important; -webkit-text-fill-color: #2D3436 !important; background: none !important; }
    h2, h3 { color: #2D3436 !important; font-weight: 600 !important; }
    .stButton > button[kind="primary"], .stButton > button {
        background: #E17055 !important; color: #FFFFFF !important;
        border: none !important; border-radius: 8px !important; font-weight: 600 !important;
        padding: 8px 32px !important; font-size: 1rem !important;
    }
    .stButton > button:hover { background: #D35400 !important; box-shadow: 0 4px 12px rgba(225,112,85,0.3) !important; }
    .section-card { background: #FFFFFF; border-radius: 10px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); border: 1px solid #DFE6E9; padding: 20px; margin-bottom: 16px; }
    .section-card-header { background: #2D3436; color: #FFFFFF; padding: 10px 16px; border-radius: 10px 10px 0 0; margin: -20px -20px 16px -20px; font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; }
    .breadcrumb { color: #636E72; font-size: 0.85rem; margin-bottom: 1rem; }
    .breadcrumb a { color: #E17055; text-decoration: none; }
    hr { border-color: #DFE6E9 !important; }

    .result-safe { background: #FFFFFF; border: 2px solid #00B894; border-radius: 12px; padding: 30px; text-align: center; box-shadow: 0 4px 20px rgba(0,184,148,0.15); }
    .result-normal { background: #FFFFFF; border: 2px solid #E17055; border-radius: 12px; padding: 30px; text-align: center; box-shadow: 0 4px 20px rgba(225,112,85,0.15); }
    .result-aggressive { background: #FFFFFF; border: 2px solid #2D3436; border-radius: 12px; padding: 30px; text-align: center; box-shadow: 0 4px 20px rgba(45,52,54,0.15); }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="breadcrumb">🏠 <a href="/">HOME</a> &gt; LIVE PREDICTIONS</p>', unsafe_allow_html=True)
st.markdown("# 🎯 Live Predictions")
st.markdown("---")

# ─── Load Predictor ──────────────────────────────────────────────────────


@st.cache_resource
def get_predictor():
    p = DrivingBehaviorPredictor()
    p.load()
    return p

try:
    predictor = get_predictor()
except FileNotFoundError as e:
    st.error(f"⚠️ {str(e)}")
    st.stop()

# ─── Feature definitions ─────────────────────────────────────────────────
feature_info = {
    "avg_speed": {"label": "🚗 Avg Speed (km/h)", "min": 0.0, "max": 200.0, "default": 60.0},
    "max_speed": {"label": "🏎️ Max Speed (km/h)", "min": 0.0, "max": 250.0, "default": 90.0},
    "speed_std": {"label": "📊 Speed Variability", "min": 0.0, "max": 50.0, "default": 10.0},
    "avg_acceleration": {"label": "⚡ Avg Accel (m/s²)", "min": 0.0, "max": 8.0, "default": 2.0},
    "max_acceleration": {"label": "💨 Max Accel (m/s²)", "min": 0.0, "max": 10.0, "default": 4.0},
    "acceleration_std": {"label": "📊 Accel Variability", "min": 0.0, "max": 5.0, "default": 1.0},
    "avg_braking": {"label": "🛑 Avg Braking", "min": 0.0, "max": 12.0, "default": 3.0},
    "max_braking": {"label": "⛔ Max Braking", "min": 0.0, "max": 15.0, "default": 6.0},
    "braking_std": {"label": "📊 Braking Var.", "min": 0.0, "max": 5.0, "default": 1.5},
    "avg_steering_angle": {"label": "🔄 Avg Steering (°)", "min": 0.0, "max": 70.0, "default": 10.0},
    "max_steering_angle": {"label": "↩️ Max Steering (°)", "min": 0.0, "max": 90.0, "default": 25.0},
    "steering_variability": {"label": "📊 Steering Var.", "min": 0.0, "max": 30.0, "default": 5.0},
    "avg_rpm": {"label": "🔧 Avg RPM", "min": 500.0, "max": 7000.0, "default": 2500.0},
    "max_rpm": {"label": "🔧 Max RPM", "min": 500.0, "max": 8000.0, "default": 4000.0},
    "avg_throttle": {"label": "🎚️ Avg Throttle (%)", "min": 0.0, "max": 100.0, "default": 40.0},
    "throttle_variability": {"label": "📊 Throttle Var.", "min": 0.0, "max": 35.0, "default": 10.0},
    "harsh_braking_count": {"label": "⚠️ Harsh Braking", "min": 0.0, "max": 10.0, "default": 1.0},
    "rapid_acceleration_count": {"label": "⚠️ Rapid Accel", "min": 0.0, "max": 10.0, "default": 1.0},
    "avg_fuel_consumption": {"label": "⛽ Fuel (L/100km)", "min": 2.0, "max": 25.0, "default": 8.0},
    "total_distance": {"label": "📏 Distance (km)", "min": 1.0, "max": 100.0, "default": 25.0},
}

presets = {
    "Custom": None,
    "🟢 Safe Driver": {"avg_speed": 50.0, "max_speed": 70.0, "speed_std": 8.0, "avg_acceleration": 1.0, "max_acceleration": 2.5, "acceleration_std": 0.5, "avg_braking": 2.0, "max_braking": 4.0, "braking_std": 0.8, "avg_steering_angle": 5.0, "max_steering_angle": 15.0, "steering_variability": 3.0, "avg_rpm": 2000.0, "max_rpm": 3000.0, "avg_throttle": 30.0, "throttle_variability": 6.0, "harsh_braking_count": 0.0, "rapid_acceleration_count": 0.0, "avg_fuel_consumption": 6.0, "total_distance": 20.0},
    "🟠 Normal Driver": {"avg_speed": 70.0, "max_speed": 100.0, "speed_std": 15.0, "avg_acceleration": 2.0, "max_acceleration": 4.5, "acceleration_std": 1.0, "avg_braking": 4.0, "max_braking": 7.0, "braking_std": 1.5, "avg_steering_angle": 12.0, "max_steering_angle": 30.0, "steering_variability": 6.0, "avg_rpm": 3000.0, "max_rpm": 4500.0, "avg_throttle": 50.0, "throttle_variability": 12.0, "harsh_braking_count": 2.0, "rapid_acceleration_count": 2.0, "avg_fuel_consumption": 8.5, "total_distance": 35.0},
    "⚫ Aggressive Driver": {"avg_speed": 110.0, "max_speed": 160.0, "speed_std": 25.0, "avg_acceleration": 4.0, "max_acceleration": 7.0, "acceleration_std": 2.0, "avg_braking": 7.5, "max_braking": 11.0, "braking_std": 2.5, "avg_steering_angle": 25.0, "max_steering_angle": 55.0, "steering_variability": 12.0, "avg_rpm": 4500.0, "max_rpm": 6500.0, "avg_throttle": 75.0, "throttle_variability": 15.0, "harsh_braking_count": 6.0, "rapid_acceleration_count": 5.0, "avg_fuel_consumption": 13.0, "total_distance": 50.0},
}

# ─── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Quick Presets")
    selected_preset = st.radio("Load a profile:", list(presets.keys()), index=0)
    st.markdown("---")
    st.markdown("**How to use:**\n1. Select a preset or customize\n2. Adjust sliders\n3. Click **Predict**")

active_preset = presets.get(selected_preset)

# ─── Input Form ──────────────────────────────────────────────────────────
feature_columns = predictor.get_feature_names()
input_features = {}

groups = {
    "🚗 Speed": ["avg_speed", "max_speed", "speed_std"],
    "⚡ Acceleration": ["avg_acceleration", "max_acceleration", "acceleration_std"],
    "🛑 Braking": ["avg_braking", "max_braking", "braking_std"],
    "🔄 Steering": ["avg_steering_angle", "max_steering_angle", "steering_variability"],
    "🔧 Engine": ["avg_rpm", "max_rpm", "avg_throttle", "throttle_variability"],
    "⚠️ Events & Other": ["harsh_braking_count", "rapid_acceleration_count", "avg_fuel_consumption", "total_distance"],
}

for group_name, group_feats in groups.items():
    st.markdown(f"### {group_name}")
    cols = st.columns(len(group_feats))
    for i, feat in enumerate(group_feats):
        if feat in feature_columns:
            info = feature_info.get(feat, {"label": feat, "min": 0.0, "max": 100.0, "default": 50.0})
            default = active_preset.get(feat, info["default"]) if active_preset else info["default"]
            with cols[i]:
                input_features[feat] = st.slider(info["label"], min_value=info["min"], max_value=info["max"], value=float(default), key=f"s_{feat}")

st.markdown("---")
predict_clicked = st.button("🔮 **Predict Behavior**", use_container_width=False, type="primary")

if predict_clicked:
    with st.spinner("Analyzing..."):
        result = predictor.predict(input_features)

    if result.get("error"):
        st.error(f"Error: {result['error']}")
    else:
        label = result["label"]
        confidence = result["confidence"]
        probabilities = result.get("probabilities", {})

        config = {
            "Safe": {"css": "result-safe", "color": "#00B894", "emoji": "🟢"},
            "Normal": {"css": "result-normal", "color": "#E17055", "emoji": "🟠"},
            "Aggressive": {"css": "result-aggressive", "color": "#2D3436", "emoji": "⚫"},
        }.get(label, {"css": "result-normal", "color": "#E17055", "emoji": "🟠"})

        st.markdown("---")

        col_result, col_gauge = st.columns(2)

        with col_result:
            st.markdown(f"""
            <div class="{config['css']}">
                <p style="font-size: 3rem; margin: 0;">{config['emoji']}</p>
                <h2 style="color: {config['color']}; font-size: 2.2rem; margin: 8px 0;">{label}</h2>
                <p style="color: #636E72; font-size: 1rem;">
                    Confidence: <b style="color: {config['color']}; font-size: 1.5rem;">{confidence*100:.1f}%</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col_gauge:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence * 100,
                number={"suffix": "%", "font": {"color": config["color"], "size": 36}},
                title={"text": "Confidence", "font": {"color": "#636E72", "size": 16}},
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor="#B2BEC3"),
                    bar=dict(color=config["color"]),
                    bgcolor="#F0F0F0",
                    bordercolor="#DFE6E9",
                    steps=[
                        dict(range=[0, 33], color="#FFF5F5"),
                        dict(range=[33, 66], color="#FFF9F0"),
                        dict(range=[66, 100], color="#F0FFF5"),
                    ],
                )
            ))
            fig.update_layout(height=280, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"), margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        if probabilities:
            st.markdown("""
            <div class="section-card" style="margin-top: 16px;">
                <div class="section-card-header">PROBABILITY BREAKDOWN</div>
            </div>
            """, unsafe_allow_html=True)

            prob_colors = {"Safe": "#00B894", "Normal": "#E17055", "Aggressive": "#2D3436"}
            fig = go.Figure(data=[go.Bar(
                x=list(probabilities.keys()), y=list(probabilities.values()),
                marker=dict(color=[prob_colors.get(k, "#636E72") for k in probabilities.keys()], line=dict(width=0), cornerradius=4),
                text=[f"{v*100:.1f}%" for v in probabilities.values()], textposition="outside", textfont=dict(color="#2D3436", size=16)
            )])
            fig.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color="#2D3436"),
                              xaxis=dict(gridcolor="#F0F0F0", showline=True, linecolor="#DFE6E9"),
                              yaxis=dict(title="Probability", gridcolor="#F0F0F0", showline=True, linecolor="#DFE6E9", range=[0, 1.15]),
                              height=300, margin=dict(t=10, b=30))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 💡 Driving Insights")
        if label == "Safe":
            st.success("✅ **Safe driving detected!** Parameters indicate cautious, controlled driving. Keep it up!")
        elif label == "Normal":
            st.warning("⚡ **Normal driving.** Within typical ranges. Consider reducing speed variability for safer driving.")
        else:
            st.error("⚠️ **Aggressive driving detected!** High speed, harsh braking, and rapid acceleration. Reduce speed and anticipate stops.")
