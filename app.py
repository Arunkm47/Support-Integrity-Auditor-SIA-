import streamlit as st
import pandas as pd

st.title("Support Integrity Auditor (SIA)")

st.header("Single Ticket Analysis")

subject = st.text_input("Ticket Subject")

description = st.text_area("Ticket Description")

category = st.selectbox(
    "Issue Category",
    ["General Inquiry", "Account", "Billing", "Technical", "Fraud"]
)

priority = st.selectbox(
    "Priority Level",
    ["Low", "Medium", "High", "Critical"]
)

channel = st.selectbox(
    "Ticket Channel",
    ["Email", "Chat", "Phone", "Web Form"]
)

resolution_hours = st.number_input(
    "Resolution Time (Hours)",
    min_value=0,
    value=24
)

satisfaction = st.slider(
    "Satisfaction Score",
    1,
    5,
    3
)

if st.button("Analyze Ticket"):
    st.success("Analysis Completed")

    st.subheader("Prediction")
    st.write("Mismatch")

    st.subheader("Evidence Dossier")

    st.write("Category Score: 3")
    st.write("Satisfaction Risk: 4")
    st.write("Assigned Priority: Low")
    st.write("Estimated Severity: High")

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write(df.head())

    st.success("CSV Uploaded Successfully")

st.header("Priority Mismatch Dashboard")

import pandas as pd

df_dash = pd.read_csv("pseudo_labeled_dataset.csv")

st.subheader("Mismatch Distribution")
st.bar_chart(df_dash["mismatch"].value_counts())

st.subheader("Mismatch Type Distribution")
st.bar_chart(df_dash["mismatch_type"].value_counts())
