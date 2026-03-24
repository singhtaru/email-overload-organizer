import sys
from pathlib import Path

import streamlit as st

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pipeline import EmailAnalyzer

# @st.cache_resource loads this ONCE when the app first starts,
# then reuses the same object for every subsequent interaction.
# Without this, EmailAnalyzer() — which loads the sentence transformer,
# the stacking model pkl, and spaCy — runs on every single rerun.
@st.cache_resource
def load_analyzer():
    return EmailAnalyzer()

st.title("Email Overload Organizer")

# Show a spinner during the one-time load so the user knows something is happening
with st.spinner("Loading models... (first load only)"):
    analyzer = load_analyzer()

email_input = st.text_area("Enter your email text:")

if st.button("Analyze"):
    if email_input.strip():
        with st.spinner("Analyzing..."):
            result = analyzer.analyze(email_input)

        st.subheader("Result")
        importance = result["importance"]

        # Color-coded importance badge
        if importance == "Important":
            st.success(f"✅ {importance}")
        else:
            st.info(f"📭 {importance}")

        if result["entities"]:
            st.subheader("Extracted Entities")
            st.json(result["entities"])
        elif importance == "Important":
            st.write("No named entities found.")
    else:
        st.warning("Please enter some text.")