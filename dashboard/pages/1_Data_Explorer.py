"""
Data Explorer Page — Light Admin Theme
========================================
Interactive data exploration with filters, distribution charts,
correlation heatmap, and summary statistics.
"""


import sys
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from src.config import ENGINEERED_FEATURES
from src.db_manager import query_to_dataframe, init_database

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Data Explorer", page_icon="🔍", layout="wide")

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
    h2, h3 { color: #2D3436 !important; }
    [data-testid="stMetric"] { background: #FFFFFF; border-radius: 10px; border: 1px solid #DFE6E9; box-shadow: 0 2px 12px rgba(0,0,0,0.06); }
    [data-testid="stMetricLabel"] { background: #2D3436; color: #FFFFFF !important; padding: 8px 16px !important; font-size: 0.7rem !important; text-transform: uppercase !important; letter-spacing: 1.5px !important; }
    [data-testid="stMetricLabel"] > div, [data-testid="stMetricLabel"] p { color: #FFFFFF !important; }
    [data-testid="stMetricValue"] { color: #2D3436 !important; font-weight: 700 !important; padding: 12px 16px 8px 16px !important; }
    .stTabs [data-baseweb="tab-list"] { background: #FFFFFF; border-radius: 10px 10px 0 0; }
    .stTabs [data-baseweb="tab"] { color: #636E72 !important; font-weight: 500; }
    .stTabs [aria-selected="true"] { background: #E17055 !important; color: #FFFFFF !important; border-radius: 8px; }
    .stTabs [data-baseweb="tab-panel"] { background: #FFFFFF; border: 1px solid #DFE6E9; border-top: none; border-radius: 0 0 10px 10px; padding: 20px; }
    .stButton > button { background: #E17055 !important; color: #FFFFFF !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; }
    .stDownloadButton > button { background: #2D3436 !important; color: #FFFFFF !important; border-radius: 8px !important; border: none !important; }
    .section-card { background: #FFFFFF; border-radius: 10px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); border: 1px solid #DFE6E9; padding: 20px; margin-bottom: 16px; }
    .section-card-header { background: #2D3436; color: #FFFFFF; padding: 10px 16px; border-radius: 10px 10px 0 0; margin: -20px -20px 16px -20px; font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; }
    .breadcrumb { color: #636E72; font-size: 0.85rem; margin-bottom: 1rem; }
    .breadcrumb a { color: #E17055; text-decoration: none; }
    hr { border-color: #DFE6E9 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="breadcrumb">🏠 <a href="/">HOME</a> &gt; DATA EXPLORER</p>', unsafe_allow_html=True)
st.markdown("# 🔍 Data Explorer")
st.markdown("---")

# ─── Load Data ────────────────────────────────────────────────────────────
init_database()



@st.cache_data(ttl=300)
def load_data():
    raw = query_to_dataframe("SELECT * FROM driving_data")
    features = query_to_dataframe("SELECT * FROM features")
    return raw, features

try:
    raw_data, features_data = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

if raw_data.empty:
    st.warning("No data found. Run `python src/run_pipeline.py` first.")
    st.stop()

chart_colors = {"Safe": "#00B894", "Normal": "#E17055", "Aggressive": "#2D3436"}
chart_layout_base = dict(
    plot_bgcolor="#FFFFFF", paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#2D3436", size=11),
    xaxis=dict(gridcolor="#F0F0F0", showline=True, linecolor="#DFE6E9"),
    yaxis=dict(gridcolor="#F0F0F0", showline=True, linecolor="#DFE6E9"),
    legend=dict(font=dict(size=10, color="#636E72"))
)

# ─── Sidebar Filters ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Filters")
    st.markdown("---")
    behavior_options = ["All"] + sorted(raw_data["behavior_label"].unique().tolist())
    selected_behavior = st.selectbox("Behavior Type", behavior_options, key="explorer_behavior")
    min_speed, max_speed = float(raw_data["speed_kmh"].min()), float(raw_data["speed_kmh"].max())
    speed_range = st.slider("Speed Range (km/h)", min_value=min_speed, max_value=max_speed, value=(min_speed, max_speed))

filtered_data = raw_data.copy()
if selected_behavior != "All":
    filtered_data = filtered_data[filtered_data["behavior_label"] == selected_behavior]
filtered_data = filtered_data[(filtered_data["speed_kmh"] >= speed_range[0]) & (filtered_data["speed_kmh"] <= speed_range[1])]

# ─── Summary Statistics ──────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
with col1: st.metric("Records", f"{len(filtered_data):,}")
with col2: st.metric("Unique Trips", f"{filtered_data['trip_id'].nunique():,}")
with col3: st.metric("Avg Speed", f"{filtered_data['speed_kmh'].mean():.1f} km/h")
with col4: st.metric("Avg Accel", f"{filtered_data['acceleration_ms2'].mean():.2f} m/s²")
with col5: st.metric("Avg Braking", f"{filtered_data['braking_force'].mean():.2f}")

st.markdown("---")

# ─── Tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Table", "📊 Distributions", "🔥 Correlation", "📈 Trends"])

with tab1:
    display_cols = st.multiselect(
        "Select columns:", filtered_data.columns.tolist(),
        default=["trip_id", "speed_kmh", "acceleration_ms2", "braking_force", "rpm", "throttle_pct", "behavior_label"]
    )
    if display_cols:
        st.dataframe(filtered_data[display_cols].head(500), use_container_width=True, height=400)
        st.caption(f"Showing top 500 of {len(filtered_data):,} records")
    csv = filtered_data.to_csv(index=False)
    st.download_button("📥 Download Filtered Data", data=csv, file_name="filtered_driving_data.csv", mime="text/csv")

with tab2:
    dist_col = st.selectbox("Feature:", ["speed_kmh", "acceleration_ms2", "braking_force", "steering_angle_deg", "rpm", "throttle_pct", "fuel_consumption"])
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(filtered_data, x=dist_col, color="behavior_label", color_discrete_map=chart_colors, nbins=40, barmode="overlay", opacity=0.7, marginal="box")
        fig.update_layout(**chart_layout_base, height=400, title=f"Distribution of {dist_col}")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.violin(filtered_data, x="behavior_label", y=dist_col, color="behavior_label", color_discrete_map=chart_colors, box=True, points="outliers")
        fig.update_layout(**chart_layout_base, height=400, showlegend=False, title=f"{dist_col} by Behavior")
        st.plotly_chart(fig, use_container_width=True)
    stats = filtered_data.groupby("behavior_label")[dist_col].describe().round(2)
    st.dataframe(stats, use_container_width=True)

with tab3:
    if not features_data.empty:
        numeric_cols = [c for c in ENGINEERED_FEATURES if c in features_data.columns]
        if numeric_cols:
            corr = features_data[numeric_cols].corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr.values, x=[c.replace("_", " ").title() for c in corr.columns],
                y=[c.replace("_", " ").title() for c in corr.index],
                colorscale=[[0, "#2D3436"], [0.5, "#FFFFFF"], [1, "#E17055"]],
                zmin=-1, zmax=1,
                text=np.round(corr.values, 2), texttemplate="%{text}", textfont={"size": 8, "color": "#2D3436"},
            ))
            fig.update_layout(**chart_layout_base, height=650, title="Pearson Correlation Heatmap", margin=dict(l=10, r=10, t=40, b=10))
            fig.update_xaxes(tickangle=45, gridcolor="#F0F0F0")
            st.plotly_chart(fig, use_container_width=True)

            corr_pairs = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    corr_pairs.append({"Feature 1": corr.columns[i], "Feature 2": corr.columns[j], "Correlation": round(corr.iloc[i, j], 4)})
            st.dataframe(pd.DataFrame(corr_pairs).sort_values("Correlation", key=abs, ascending=False).head(10), use_container_width=True)

with tab4:
    c1, c2 = st.columns(2)
    opts = ["speed_kmh", "acceleration_ms2", "braking_force", "steering_angle_deg", "rpm", "throttle_pct"]
    with c1: x_axis = st.selectbox("X-axis", opts, index=0)
    with c2: y_axis = st.selectbox("Y-axis", opts, index=1)

    fig = px.scatter(filtered_data.sample(min(1000, len(filtered_data)), random_state=42), x=x_axis, y=y_axis, color="behavior_label", color_discrete_map=chart_colors, opacity=0.6, hover_data=["trip_id"])
    fig.update_layout(**chart_layout_base, height=450)
    st.plotly_chart(fig, use_container_width=True)

    if not features_data.empty:
        sample = features_data.sample(min(200, len(features_data)), random_state=42)
        dims = [d for d in ["avg_speed", "avg_acceleration", "avg_braking", "avg_steering_angle", "avg_rpm", "avg_throttle"] if d in sample.columns]
        label_map = {"Safe": 0, "Normal": 1, "Aggressive": 2}
        sample["label_num"] = sample["behavior_label"].map(label_map)
        fig = go.Figure(data=go.Parcoords(
            line=dict(color=sample["label_num"], colorscale=[[0, "#00B894"], [0.5, "#E17055"], [1, "#2D3436"]], showscale=False),
            dimensions=[dict(label=d.replace("_", " ").title(), values=sample[d]) for d in dims]
        ))
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color="#2D3436"), height=450, title="Feature Profiles by Behavior")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🟢 Safe · 🟠 Normal · ⚫ Aggressive")
