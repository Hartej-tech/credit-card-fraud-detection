import streamlit as st

st.title("Credit Card Fraud Detection System")

amount = st.number_input("Enter Transaction Amount", min_value=0.0)

if st.button("Check Transaction"):
    if amount > 1000:
        st.error("Fraud Transaction Detected")
    else:
        st.success("Normal Transaction")
        