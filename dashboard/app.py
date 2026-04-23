"""
Driving Behavior Pattern Analysis — Dashboard
===============================================
Main Streamlit application entry point.
Overview page with KPIs, behavior distribution, and model summary.
Redesigned: Light theme, card-based layout inspired by admin dashboard.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from src.config import METRICS_PATH
from src.db_manager import query_to_dataframe, init_database, table_exists

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Driving Behavior Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Theme CSS — Light Admin Dashboard ──────────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* ── Root variables ── */
    :root {
        --bg-main: #E8EDEB;
        --bg-card: #FFFFFF;
        --bg-sidebar: #2D3436;
        --bg-header-dark: #2D3436;
        --text-primary: #2D3436;
        --text-secondary: #636E72;
        --text-light: #B2BEC3;
        --accent-coral: #E17055;
        --accent-mint: #00B894;
        --accent-dark: #2D3436;
        --accent-peach: #FAB1A0;
        --border-light: #DFE6E9;
        --shadow: 0 2px 12px rgba(0,0,0,0.06);
        --radius: 10px;
    }

    * { font-family: 'Inter', sans-serif !important; }

    /* ── Main app background ── */
    .stApp {
        background-color: var(--bg-main) !important;
    }

    .main .block-container {
        padding-top: 1.5rem;
        max-width: 1400px;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--bg-sidebar) !important;
        border-right: none;
    }
    [data-testid="stSidebar"] * {
        color: #B2BEC3 !important;
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #FFFFFF !important;
        font-size: 1.1rem;
        letter-spacing: 1px;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.1) !important;
    }
    [data-testid="stSidebar"] a {
        color: var(--accent-peach) !important;
    }

    /* ── Headers ── */
    h1 {
        color: var(--text-primary) !important;
        font-weight: 800 !important;
        font-size: 1.8rem !important;
        -webkit-text-fill-color: var(--text-primary) !important;
        background: none !important;
    }
    h2 {
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
    }
    h3 {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
    }

    /* ── Metric Cards ── */
    [data-testid="stMetric"] {
        background: var(--bg-card);
        border-radius: var(--radius);
        padding: 0;
        box-shadow: var(--shadow);
        border: 1px solid var(--border-light);
        overflow: hidden;
    }

    [data-testid="stMetricLabel"] {
        background: var(--bg-header-dark);
        color: #FFFFFF !important;
        padding: 8px 16px !important;
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
        margin: 0 !important;
        display: block;
    }
    [data-testid="stMetricLabel"] > div {
        color: #FFFFFF !important;
    }
    [data-testid="stMetricLabel"] p {
        color: #FFFFFF !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        padding: 12px 16px 4px 16px !important;
    }

    [data-testid="stMetricDelta"] {
        color: var(--accent-mint) !important;
        font-size: 0.75rem !important;
        padding: 0 16px 12px 16px !important;
    }

    /* ── Generic Card Styling ── */
    .stPlotlyChart, .stDataFrame {
        background: var(--bg-card);
        border-radius: var(--radius);
        box-shadow: var(--shadow);
        border: 1px solid var(--border-light);
        padding: 8px;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-card);
        border-radius: var(--radius) var(--radius) 0 0;
        padding: 4px;
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--text-secondary) !important;
        font-weight: 500;
        border-radius: 8px;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: var(--accent-coral) !important;
        color: #FFFFFF !important;
        border-bottom: none;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background: var(--bg-card);
        border-radius: 0 0 var(--radius) var(--radius);
        border: 1px solid var(--border-light);
        border-top: none;
        padding: 20px;
    }

    /* ── Selectbox / Inputs ── */
    .stSelectbox > div > div, .stMultiSelect > div > div {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: var(--accent-coral) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 24px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: #D35400 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(225,112,85,0.3) !important;
    }

    /* ── Download Button ── */
    .stDownloadButton > button {
        background: var(--bg-header-dark) !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
    }

    /* ── Dividers ── */
    hr {
        border-color: var(--border-light) !important;
    }

    /* ── Slider ── */
    .stSlider > div > div > div {
        color: var(--text-primary) !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: var(--bg-card);
        border-radius: var(--radius);
    }

    /* ── Success/Warning/Error ── */
    .stAlert {
        border-radius: var(--radius);
    }

    /* ── Breadcrumb ── */
    .breadcrumb {
        color: var(--text-secondary);
        font-size: 0.85rem;
        margin-bottom: 1.5rem;
    }
    .breadcrumb a {
        color: var(--accent-coral);
        text-decoration: none;
    }

    /* ── Section card ── */
    .section-card {
        background: var(--bg-card);
        border-radius: var(--radius);
        box-shadow: var(--shadow);
        border: 1px solid var(--border-light);
        padding: 20px;
        margin-bottom: 16px;
    }
    .section-card-header {
        background: var(--bg-header-dark);
        color: #FFFFFF;
        padding: 10px 16px;
        border-radius: var(--radius) var(--radius) 0 0;
        margin: -20px -20px 16px -20px;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .icon-circle {
        width: 40px; height: 40px; border-radius: 50%;
        display: inline-flex; align-items: center; justify-content: center;
        font-size: 1.2rem;
    }
    .icon-coral { background: var(--accent-peach); color: var(--accent-coral); }
    .icon-mint { background: #DFFEF5; color: var(--accent-mint); }
    .icon-dark { background: #E0E0E0; color: var(--accent-dark); }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚗 DBPA")
    st.markdown("---")
    st.markdown("""
    **📊 Overview** ← _current_

    **🔍 Data Explorer**

    **🧠 Model Insights**

    **🎯 Live Predictions**

    ---

    **Stack:** Python · scikit-learn · Streamlit

    **Models:** RF · GBM · LR

    **Data:** Vehicle Telemetry
    """)
    st.markdown("---")
    st.caption("Driving Behavior v1.0")


# ─── Breadcrumb ─────────────────────────────────────────────────────────────────
st.markdown('<p class="breadcrumb">🏠 <a href="#">HOME</a> &gt; DASHBOARD</p>', unsafe_allow_html=True)
st.markdown("# Driving Behavior Analytics")
st.markdown("---")

# Check if data exists
init_database()

if not table_exists("driving_data") or not table_exists("features"):
    st.warning("⚠️ No data found. Please run the pipeline first:")
    st.code("python src/run_pipeline.py", language="bash")
    st.stop()

# Load data

@st.cache_data(ttl=300)
def load_overview_data():
    driving_data = query_to_dataframe("SELECT * FROM driving_data")
    features_data = query_to_dataframe("SELECT * FROM features")
    return driving_data, features_data


@st.cache_data(ttl=300)
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return None

try:
    driving_data, features_data = load_overview_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

if driving_data.empty:
    st.warning("⚠️ Database tables exist but are empty. Run the pipeline first.")
    st.stop()

metrics = load_metrics()

# ─── KPI Cards Row ─────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Trips", f"{features_data['trip_id'].nunique():,}", "+100%")
with col2:
    st.metric("Total Records", f"{len(driving_data):,}", f"{len(driving_data)} pts")
with col3:
    if metrics:
        st.metric("Model Accuracy", f"{metrics.get('test_accuracy', 0)*100:.1f}%", "+High")
    else:
        st.metric("Model Accuracy", "N/A")
with col4:
    if metrics:
        st.metric("F1 Score", f"{metrics.get('test_f1', 0)*100:.1f}%", "+Excellent")
    else:
        st.metric("F1 Score", "N/A")

st.markdown("---")

# ─── Row 2: Behavior Distribution + Speed Box Plot ─────────────────────────────
col_left, col_right = st.columns(2)

# Chart color palette matching reference
chart_colors = {"Safe": "#00B894", "Normal": "#E17055", "Aggressive": "#2D3436"}

with col_left:
    st.markdown("""
    <div class="section-card">
        <div class="section-card-header">BEHAVIOR DISTRIBUTION</div>
    </div>
    """, unsafe_allow_html=True)

    if "behavior_label" in features_data.columns:
        behavior_counts = features_data["behavior_label"].value_counts()
        fig_pie = px.pie(
            values=behavior_counts.values,
            names=behavior_counts.index,
            color=behavior_counts.index,
            color_discrete_map=chart_colors,
            hole=0.5,
        )
        fig_pie.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#2D3436", size=12),
            margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.15,
                font=dict(color="#636E72", size=11)
            ),
            height=320
        )
        fig_pie.update_traces(
            textposition="inside", textinfo="percent+label",
            textfont_size=12, marker=dict(line=dict(color="#FFFFFF", width=2))
        )
        st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.markdown("""
    <div class="section-card">
        <div class="section-card-header">SPEED BY BEHAVIOR TYPE</div>
    </div>
    """, unsafe_allow_html=True)

    if "avg_speed" in features_data.columns:
        fig_box = px.box(
            features_data, x="behavior_label", y="avg_speed",
            color="behavior_label", color_discrete_map=chart_colors,
        )
        fig_box.update_layout(
            plot_bgcolor="#FFFFFF", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#2D3436", size=12),
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis=dict(title="", gridcolor="#F0F0F0", showline=True, linecolor="#DFE6E9"),
            yaxis=dict(title="Avg Speed (km/h)", gridcolor="#F0F0F0", showline=True, linecolor="#DFE6E9"),
            showlegend=False, height=320
        )
        st.plotly_chart(fig_box, use_container_width=True)

st.markdown("---")

# ─── Row 3: Feature Distributions (3 columns) ──────────────────────────────────
st.markdown("""
<div class="section-card" style="margin-bottom: 0;">
    <div class="section-card-header">FEATURE DISTRIBUTIONS</div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

chart_layout = dict(
    plot_bgcolor="#FFFFFF", paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#2D3436", size=11),
    margin=dict(t=30, b=20, l=20, r=20),
    xaxis=dict(gridcolor="#F0F0F0", showline=True, linecolor="#DFE6E9"),
    yaxis=dict(gridcolor="#F0F0F0", showline=True, linecolor="#DFE6E9"),
    legend=dict(font=dict(size=9, color="#636E72"), orientation="h", y=-0.2),
    height=280
)

with col1:
    if "avg_speed" in features_data.columns:
        fig = px.histogram(
            features_data, x="avg_speed", color="behavior_label",
            color_discrete_map=chart_colors, nbins=25, barmode="overlay", opacity=0.75,
            title="Speed"
        )
        fig.update_layout(**chart_layout)
        st.plotly_chart(fig, use_container_width=True)

with col2:
    if "harsh_braking_count" in features_data.columns:
        fig = px.histogram(
            features_data, x="harsh_braking_count", color="behavior_label",
            color_discrete_map=chart_colors, nbins=15, barmode="overlay", opacity=0.75,
            title="Harsh Braking"
        )
        fig.update_layout(**chart_layout)
        st.plotly_chart(fig, use_container_width=True)

with col3:
    if "avg_rpm" in features_data.columns:
        fig = px.histogram(
            features_data, x="avg_rpm", color="behavior_label",
            color_discrete_map=chart_colors, nbins=25, barmode="overlay", opacity=0.75,
            title="Engine RPM"
        )
        fig.update_layout(**chart_layout)
        st.plotly_chart(fig, use_container_width=True)

# ─── Row 4: Model Summary ──────────────────────────────────────────────────────
if metrics:
    st.markdown("---")

    col_info, col_chart = st.columns([1, 2])

    with col_info:
        model_name = metrics.get("model_name", "N/A")
        acc = metrics.get("test_accuracy", 0) * 100
        f1 = metrics.get("test_f1", 0) * 100
        roc = (metrics.get("roc_auc", 0) or 0) * 100

        st.markdown(f"""
        <div class="section-card">
            <div class="section-card-header">BEST MODEL</div>
            <p style="font-size: 1.5rem; font-weight: 700; color: #2D3436; margin: 8px 0 4px 0;">
                {model_name}
            </p>
            <hr style="margin: 12px 0;">
            <table style="width: 100%; color: #636E72; font-size: 0.9rem;">
                <tr><td>Accuracy</td><td style="text-align: right; font-weight: 700; color: #00B894;">{acc:.1f}%</td></tr>
                <tr><td>F1 Score</td><td style="text-align: right; font-weight: 700; color: #E17055;">{f1:.1f}%</td></tr>
                <tr><td>ROC-AUC</td><td style="text-align: right; font-weight: 700; color: #2D3436;">{roc:.1f}%</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col_chart:
        st.markdown("""
        <div class="section-card" style="margin-bottom: 0;">
            <div class="section-card-header">MODEL COMPARISON</div>
        </div>
        """, unsafe_allow_html=True)

        comparison = metrics.get("model_comparison", {})
        if comparison:
            model_names = list(comparison.keys())
            f1_scores = [comparison[m].get("f1_score", 0) for m in model_names]

            fig_comp = go.Figure(data=[
                go.Bar(
                    x=model_names, y=f1_scores,
                    marker=dict(
                        color=["#E17055", "#2D3436", "#00B894"],
                        line=dict(width=0),
                        cornerradius=4
                    ),
                    text=[f"{s*100:.1f}%" for s in f1_scores],
                    textposition="outside",
                    textfont=dict(color="#2D3436", size=14, family="Inter")
                )
            ])
            fig_comp.update_layout(
                plot_bgcolor="#FFFFFF", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#2D3436"),
                margin=dict(t=10, b=30, l=30, r=30),
                xaxis=dict(gridcolor="#F0F0F0", showline=True, linecolor="#DFE6E9"),
                yaxis=dict(title="F1 Score", gridcolor="#F0F0F0", showline=True,
                           linecolor="#DFE6E9", range=[0, 1.15]),
                height=280
            )
            st.plotly_chart(fig_comp, use_container_width=True)

st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #B2BEC3; font-size: 0.85rem;">'
    'Navigate using the sidebar to explore data, model insights, and live predictions.</p>',
    unsafe_allow_html=True
)
