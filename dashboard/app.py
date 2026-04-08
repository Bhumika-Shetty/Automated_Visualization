"""
dashboard/app.py - Streamlit dashboard for AI Visualization Evaluation results.
Run with: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from analysis import load_results, compute_summary, compute_prompt_sensitivity, compute_provider_summary

st.set_page_config(page_title="AI Viz Eval Dashboard", layout="wide")
st.title("🎨 AI Models in Data Visualization — Evaluation Dashboard")
st.caption("1,800 experimental runs | 6 models | 5 datasets | 2 prompt types | 11 metrics")

# Load data
@st.cache_data
def get_data():
    results_dir = Path(__file__).parent.parent / "outputs" / "results"
    df = load_results(str(results_dir))
    return df

df = get_data()

if df.empty:
    st.warning("No results found. Run the pipeline first: `python run_pipeline.py`")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
all_models = sorted(df["model"].unique())
all_datasets = sorted(df["dataset"].unique())
all_prompts = sorted(df["prompt_type"].unique())

sel_models = st.sidebar.multiselect("Models", all_models, default=all_models)
sel_datasets = st.sidebar.multiselect("Datasets", all_datasets, default=all_datasets)
sel_prompts = st.sidebar.multiselect("Prompt Type", all_prompts, default=all_prompts)

filtered = df[
    df["model"].isin(sel_models) &
    df["dataset"].isin(sel_datasets) &
    df["prompt_type"].isin(sel_prompts)
]

st.sidebar.metric("Filtered Runs", len(filtered))

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🔀 Prompt Sensitivity", "💰 Cost vs Quality", "⏱ Latency", "🖼 Results Gallery"
])

# --- Tab 1: Overview ---
with tab1:
    st.subheader("Execution Success Rate by Model")
    summary = compute_summary(filtered)

    fig = px.bar(
        summary, x="model", y="success_rate",
        color="model", text="success_rate",
        labels={"success_rate": "Success Rate (%)", "model": "Model"},
        title="Overall Execution Success Rate (%)"
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(showlegend=False, yaxis_range=[0, 105])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Full Metrics Table")
    st.dataframe(summary, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Fidelity Score")
        fig2 = px.bar(summary, x="model", y="mean_fidelity", color="model",
                      title="Mean Fidelity Score")
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.subheader("Hallucination Rate")
        fig3 = px.bar(summary, x="model", y="mean_hallucination", color="model",
                      title="Mean Hallucination Rate")
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
    with col3:
        st.subheader("Labeling Completeness")
        fig4 = px.bar(summary, x="model", y="mean_labeling", color="model",
                      title="Mean Labeling Completeness (%)")
        fig4.update_layout(showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

# --- Tab 2: Prompt Sensitivity ---
with tab2:
    st.subheader("Rich vs Poor Prompt Success Rate per Model")
    ps = compute_prompt_sensitivity(filtered)

    rich_col = "rich_success_pct" if "rich_success_pct" in ps.columns else None
    poor_col = "poor_success_pct" if "poor_success_pct" in ps.columns else None

    if rich_col and poor_col:
        melted = ps.melt(id_vars=["model"], value_vars=[rich_col, poor_col],
                         var_name="Prompt Type", value_name="Success Rate (%)")
        melted["Prompt Type"] = melted["Prompt Type"].map(
            {rich_col: "Rich", poor_col: "Poor"}
        )
        fig5 = px.bar(melted, x="model", y="Success Rate (%)", color="Prompt Type",
                      barmode="group", title="Rich vs Poor Prompt Success Rate",
                      text="Success Rate (%)")
        fig5.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig5.update_layout(yaxis_range=[0, 115])
        st.plotly_chart(fig5, use_container_width=True)

    st.subheader("Prompt Sensitivity Index")
    st.dataframe(ps, use_container_width=True)

# --- Tab 3: Cost vs Quality ---
with tab3:
    st.subheader("Cost per Run vs Success Rate")
    summary2 = compute_summary(filtered)

    fig6 = px.scatter(
        summary2, x="mean_cost", y="success_rate",
        text="model", size="total_runs",
        title="Cost vs Success Rate (Pareto Frontier)",
        labels={"mean_cost": "Mean Cost per Run ($)", "success_rate": "Success Rate (%)"}
    )
    fig6.update_traces(textposition="top center")
    st.plotly_chart(fig6, use_container_width=True)

    st.subheader("Provider Summary")
    prov = compute_provider_summary(filtered)
    st.dataframe(prov, use_container_width=True)

# --- Tab 4: Latency ---
with tab4:
    st.subheader("Average Latency by Model")
    summary3 = compute_summary(filtered)

    fig7 = px.bar(summary3, x="model", y="mean_latency", color="model",
                  text="mean_latency", title="Mean Generation Latency (seconds)")
    fig7.update_traces(texttemplate="%{text:.1f}s", textposition="outside")
    fig7.update_layout(showlegend=False)
    st.plotly_chart(fig7, use_container_width=True)

    st.subheader("Latency vs Success Rate")
    fig8 = px.scatter(
        summary3, x="mean_latency", y="success_rate", text="model",
        title="Latency vs Success Rate",
        labels={"mean_latency": "Mean Latency (s)", "success_rate": "Success Rate (%)"}
    )
    fig8.update_traces(textposition="top center")
    st.plotly_chart(fig8, use_container_width=True)

# --- Tab 5: Results Gallery ---
with tab5:
    st.subheader("Results Gallery")
    images_dir = Path(__file__).parent.parent / "outputs" / "images"
    images = sorted(images_dir.glob("*.png")) if images_dir.exists() else []

    if not images:
        st.info("No images found. Run the pipeline to generate visualizations.")
    else:
        # Filter images by selected models/datasets/prompts
        filtered_imgs = []
        for img in images:
            name = img.stem
            parts = name.rsplit("_", 1)
            if any(m.replace("/", "_").replace(".", "_") in name for m in sel_models):
                if any(d in name for d in sel_datasets):
                    filtered_imgs.append(img)

        st.write(f"Showing {len(filtered_imgs)} images")
        cols = st.columns(3)
        for i, img_path in enumerate(filtered_imgs[:30]):
            with cols[i % 3]:
                st.image(str(img_path), caption=img_path.stem, use_container_width=True)
