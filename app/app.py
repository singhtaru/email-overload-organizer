import sys
import os

# Fix import path so `src` is visible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.pipeline import EmailAnalyzer

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Email Overload Organizer", layout="wide")

analyzer = EmailAnalyzer()

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
body {
    background-color: #0b0f19;
}
.block-container {
    padding-top: 2rem;
    max-width: 900px;
    margin: auto;
}
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #ffffff;
}
.subtitle {
    text-align: center;
    color: #a0a0a0;
    margin-bottom: 30px;
    font-size: 16px;
}
.card {
    background: #151a28;
    padding: 25px;
    border-radius: 15px;
    margin-top: 20px;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.4);
}
textarea {
    background-color: #1c2233 !important;
    color: white !important;
    border-radius: 10px !important;
}
.stButton>button {
    background: linear-gradient(90deg, #4CAF50, #2ecc71);
    color: white;
    border-radius: 8px;
    height: 45px;
    width: 100%;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="title">📬 Email Overload Organizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Organize your inbox by detecting priority, tasks, and key information</div>', unsafe_allow_html=True)

# ---------- INPUT ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📥 Paste Your Email")

email_text = st.text_area(
    "",
    height=180,
    placeholder="Paste your email here..."
)

analyze = st.button("🚀 Analyze Email")

st.markdown('</div>', unsafe_allow_html=True)

# ---------- RESULTS ----------
if analyze and email_text.strip():

    with st.spinner("Analyzing email..."):
        result = analyzer.analyze(email_text)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Analysis Result")

    col1, col2 = st.columns(2)

    # -------- PRIORITY + SUMMARY --------
    with col1:
        st.markdown("### 📌 Priority")

        classification = result.get("classification", "N/A")

        if classification == "High":
            st.error("🔴 High Priority")
        elif classification == "Medium":
            st.warning("🟡 Medium Priority")
        else:
            st.success("🟢 Low Priority")

        # -------- SUMMARY --------
        st.markdown("### 📝 Summary")
        st.info(result.get("summary", "No summary available"))

    # -------- NER ENTITIES --------
    with col2:
        st.markdown("### 🧠 Key Entities")

        entities = result.get("entities", {})

        if not entities or all(len(v) == 0 for v in entities.values()):
            st.info("No entities found")
        else:
            for label, values in entities.items():
                if values:
                    st.markdown(f"### 🔹 {label}")
                    for val in values:
                        st.markdown(f"- {val}")

    st.markdown('</div>', unsafe_allow_html=True)

    # -------- RAW OUTPUT --------
    with st.expander("🔍 View Technical Output"):
        st.json(result)

elif analyze:
    st.warning("⚠️ Please enter an email first.")