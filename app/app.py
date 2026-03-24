import sys
from pathlib import Path

import streamlit as st

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pipeline import EmailAnalyzer

st.title("Email Overload Organizer")

analyzer = EmailAnalyzer()

email_input = st.text_area("Enter your email text:")

if st.button("Analyze"):
    if email_input.strip():
        result = analyzer.analyze(email_input)

        st.subheader("Result")
        st.write("Importance:", result["importance"])

        if result["entities"]:
            st.subheader("Entities")
            st.json(result["entities"])
    else:
        st.warning("Please enter some text.")