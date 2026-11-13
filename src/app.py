"""Streamlit interface for EVT Risk Agent."""

from __future__ import annotations

import streamlit as st


st.set_page_config(page_title="EVT Risk Agent", page_icon="📊")
st.title("EVT Risk Agent")

with st.sidebar:
    st.header("Data input")
    st.file_uploader("Upload a historical CSV file", type=["csv"])
    st.slider("Confidence level", 0.90, 0.999, 0.99, 0.01)

# IMPLEMENTAR

