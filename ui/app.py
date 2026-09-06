from __future__ import annotations

from pathlib import Path

import streamlit as st

from ui.dashboard import load_dashboard_data

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="ScaleForge-ML", layout="wide")
st.title("ScaleForge-ML evidence dashboard")
st.caption("Read-only views derived from qualified canonical artifacts")

data = load_dashboard_data(ROOT)
st.info(data["project_status"])

st.subheader("Independent release gates")
st.dataframe(
    data["gates"][["subsystem", "decision", "selected_configuration"]],
    hide_index=True,
    use_container_width=True,
)

left, right = st.columns(2)
with left:
    st.subheader("Serving at frozen concurrency 2")
    st.dataframe(data["selected_serving"], hide_index=True, use_container_width=True)
with right:
    st.subheader("Training qualification replicates")
    chart = data["training"].pivot(index="replicate", columns="config_id", values="tokens_per_s")
    st.bar_chart(chart)

st.warning(
    "The serving selection remains REVIEW because vLLM shutdown was not clean. "
    "Two-GPU numerical qualification is BLOCKED_EXTERNAL."
)
