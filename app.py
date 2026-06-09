import streamlit as st

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

    st.write("Prediction: Demo Output")

    st.write("Confidence: 95%")
