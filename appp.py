import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title("Spotify Dashboard")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    # Read CSV into DataFrame
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Example plot
    st.subheader("Example Plot")
    if 'ColumnName' in df.columns:  # Replace ColumnName with an actual column in your CSV
        fig, ax = plt.subplots()
        sns.histplot(df['ColumnName'], kde=True, ax=ax)
        st.pyplot(fig)
else:
    st.info("Please upload a CSV file to see the dashboard.")
