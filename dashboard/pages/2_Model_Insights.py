"""
Model Insights Page — Light Admin Theme
=========================================
Model evaluation metrics, confusion matrix, feature importance, and comparison.
"""


import sys
import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.config import METRICS_PATH

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Model Insights", page_icon="🧠", layout="wide")

# ─── Theme CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif !important; }
    .stApp { background-color: #E8EDEB !important; }
    [data-testid="stSidebar"] { background: #2D3436 !important; }
    [data-testid="stSidebar"] * { color: #B2BEC3 !important; }
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
    .section-card { background: #FFFFFF; border-radius: 10px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); border: 1px solid #DFE6E9; padding: 20px; margin-bottom: 16px; }
    .section-card-header { background: #2D3436; color: #FFFFFF; padding: 10px 16px; border-radius: 10px 10px 0 0; margin: -20px -20px 16px -20px; font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; }
    .breadcrumb { color: #636E72; font-size: 0.85rem; margin-bottom: 1rem; }
    .breadcrumb a { color: #E17055; text-decoration: none; }
    hr { border-color: #DFE6E9 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="breadcrumb">🏠 <a href="/">HOME</a> &gt; MODEL INSIGHTS</p>', unsafe_allow_html=True)
st.markdown("# 🧠 Model Insights")
st.markdown("---")

chart_layout = dict(
    plot_bgcolor="#FFFFFF", paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#2D3436", size=12),
    xaxis=dict(gridcolor="#F0F0F0", showline=True, linecolor="#DFE6E9"),
    yaxis=dict(gridcolor="#F0F0F0", showline=True, linecolor="#DFE6E9"),
)



@st.cache_data(ttl=300)
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return None

metrics = load_metrics()
if not metrics:
    st.warning("⚠️ No model metrics found. Run `python src/run_pipeline.py` first.")
    st.stop()

# ─── KPI Cards ────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
with col1: st.metric("Model", metrics.get("model_name", "N/A"))
with col2: st.metric("Accuracy", f"{metrics.get('test_accuracy', 0)*100:.1f}%")
with col3: st.metric("Precision", f"{metrics.get('test_precision', 0)*100:.1f}%")
with col4: st.metric("Recall", f"{metrics.get('test_recall', 0)*100:.1f}%")
with col5: st.metric("F1 Score", f"{metrics.get('test_f1', 0)*100:.1f}%")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Confusion Matrix", "📋 Classification Report", "🌳 Feature Importance", "⚔️ Model Comparison"])

class_labels = metrics.get("class_labels", [])

# ─── Tab 1: Confusion Matrix ─────────────────────────────────────────────
with tab1:
    cm = metrics.get("confusion_matrix", [])
    if cm and class_labels:
        cm_array = np.array(cm)
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = go.Figure(data=go.Heatmap(
                z=cm_array, x=class_labels, y=class_labels,
                colorscale=[[0, "#FFFFFF"], [0.5, "#FAB1A0"], [1, "#E17055"]],
                text=cm_array, texttemplate="%{text}", textfont={"size": 20, "color": "#2D3436"},
                showscale=True, colorbar=dict(title="Count", titlefont=dict(color="#636E72"), tickfont=dict(color="#636E72"))
            ))
            fig.update_layout(**chart_layout, height=450, title="Confusion Matrix",
                              xaxis=dict(title="Predicted Label"), yaxis=dict(title="Actual Label", autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            cm_norm = np.round(cm_array.astype(float) / cm_array.sum(axis=1, keepdims=True) * 100, 1)
            fig = go.Figure(data=go.Heatmap(
                z=cm_norm, x=class_labels, y=class_labels,
                colorscale=[[0, "#FFFFFF"], [0.5, "#DFFEF5"], [1, "#00B894"]],
                text=[[f"{v}%" for v in row] for row in cm_norm], texttemplate="%{text}", textfont={"size": 16, "color": "#2D3436"},
                showscale=False, zmin=0, zmax=100
            ))
            fig.update_layout(**chart_layout, height=450, title="Normalized (%)",
                              xaxis=dict(title="Predicted"), yaxis=dict(title="Actual", autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

        total, correct = cm_array.sum(), np.trace(cm_array)
        st.markdown(f"""
        <div class="section-card">
            <div class="section-card-header">MISCLASSIFICATION ANALYSIS</div>
            <p style="color: #636E72;">
                Total: <b style="color: #2D3436;">{total}</b> ·
                Correct: <b style="color: #00B894;">{correct}</b> ·
                Errors: <b style="color: #E17055;">{total-correct}</b> ·
                Error Rate: <b>{(total-correct)/total*100:.1f}%</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

# ─── Tab 2: Classification Report ────────────────────────────────────────
with tab2:
    report = metrics.get("classification_report", {})
    if report:
        rows = []
        for cls in class_labels:
            if cls in report:
                rows.append({"Class": cls, "Precision": f"{report[cls].get('precision',0)*100:.1f}%",
                    "Recall": f"{report[cls].get('recall',0)*100:.1f}%", "F1-Score": f"{report[cls].get('f1-score',0)*100:.1f}%",
                    "Support": int(report[cls].get('support',0))})
        if "weighted avg" in report:
            rows.append({"Class": "📊 Weighted Avg", "Precision": f"{report['weighted avg'].get('precision',0)*100:.1f}%",
                "Recall": f"{report['weighted avg'].get('recall',0)*100:.1f}%", "F1-Score": f"{report['weighted avg'].get('f1-score',0)*100:.1f}%",
                "Support": int(report['weighted avg'].get('support',0))})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        class_metrics = {cls: report[cls] for cls in class_labels if cls in report}
        if class_metrics:
            fig = go.Figure()
            colors = {"precision": "#E17055", "recall": "#2D3436", "f1-score": "#00B894"}
            for m in ["precision", "recall", "f1-score"]:
                fig.add_trace(go.Bar(name=m.title(), x=list(class_metrics.keys()),
                    y=[class_metrics[c].get(m,0) for c in class_metrics], marker_color=colors[m],
                    text=[f"{class_metrics[c].get(m,0)*100:.1f}%" for c in class_metrics], textposition="outside",
                    textfont=dict(color="#2D3436")))
            fig.update_layout(**chart_layout, barmode="group", height=380, title="Per-Class Metrics",
                              yaxis=dict(range=[0, 1.15], title="Score"))
            st.plotly_chart(fig, use_container_width=True)

    roc_auc = metrics.get("roc_auc")
    if roc_auc:
        st.markdown(f"""
        <div class="section-card">
            <div class="section-card-header">ROC-AUC SCORE</div>
            <p style="font-size: 2.5rem; font-weight: 700; color: #E17055; margin: 8px 0;">{roc_auc*100:.1f}%</p>
            <p style="color: #636E72;">Weighted One-vs-Rest · A score above 90% indicates excellent separation.</p>
        </div>
        """, unsafe_allow_html=True)

# ─── Tab 3: Feature Importance ────────────────────────────────────────────
with tab3:
    importances = metrics.get("feature_importances", {})
    if importances:
        imp_df = pd.DataFrame(list(importances.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
        fig = go.Figure(data=go.Bar(
            x=imp_df["Importance"], y=imp_df["Feature"].apply(lambda x: x.replace("_", " ").title()),
            orientation="h",
            marker=dict(color=imp_df["Importance"], colorscale=[[0, "#FAB1A0"], [1, "#E17055"]], line=dict(width=0)),
            text=[f"{v:.3f}" for v in imp_df["Importance"]], textposition="outside", textfont=dict(color="#2D3436", size=12)
        ))
        fig.update_layout(**chart_layout, height=550, title="Feature Importance", margin=dict(l=180),
                          xaxis=dict(title="Importance"))
        st.plotly_chart(fig, use_container_width=True)

        top5 = imp_df.tail(5).sort_values("Importance", ascending=False)
        st.markdown("#### 🏅 Top 5 Features")
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            st.markdown(f"**{i}. {row['Feature'].replace('_',' ').title()}** — `{row['Importance']*100:.1f}%`")
    else:
        st.info("Feature importance not available for this model type.")

# ─── Tab 4: Model Comparison ─────────────────────────────────────────────
with tab4:
    comparison = metrics.get("model_comparison", {})
    if comparison:
        comp_rows = []
        for name, m in comparison.items():
            row = {"Model": name, "Accuracy": f"{m.get('accuracy',0)*100:.1f}%", "Precision": f"{m.get('precision',0)*100:.1f}%",
                "Recall": f"{m.get('recall',0)*100:.1f}%", "F1": f"{m.get('f1_score',0)*100:.1f}%",
                "CV F1": f"{m.get('cv_f1_mean',0)*100:.1f}%±{m.get('cv_f1_std',0)*100:.1f}%"}
            if "best_params" in m: row["Tuned"] = "✅"
            else: row["Tuned"] = "—"
            comp_rows.append(row)
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

        # Radar chart
        categories = ["Accuracy", "Precision", "Recall", "F1 Score"]
        fig = go.Figure()
        radar_colors = ["#E17055", "#2D3436", "#00B894"]
        for i, (name, m) in enumerate(comparison.items()):
            vals = [m.get("accuracy",0), m.get("precision",0), m.get("recall",0), m.get("f1_score",0)]
            vals.append(vals[0])
            fig.add_trace(go.Scatterpolar(r=vals, theta=categories+[categories[0]], fill="toself", name=name,
                line=dict(color=radar_colors[i % 3]),
                fillcolor=radar_colors[i % 3].replace(")", ",0.1)").replace("rgb", "rgba") if "rgb" in radar_colors[i % 3] else f"rgba({int(radar_colors[i%3][1:3],16)},{int(radar_colors[i%3][3:5],16)},{int(radar_colors[i%3][5:7],16)},0.1)"))
        fig.update_layout(
            polar=dict(bgcolor="#FFFFFF", radialaxis=dict(range=[0,1], gridcolor="#F0F0F0"), angularaxis=dict(gridcolor="#F0F0F0")),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color="#2D3436"),
            legend=dict(font=dict(color="#636E72")), title="Performance Radar", height=450
        )
        st.plotly_chart(fig, use_container_width=True)
