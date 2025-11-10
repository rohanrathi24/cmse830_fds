import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.set_page_config(page_title="Spotify Dashboard — CMSE 830 Midterm", layout="wide")

st.title("🎵 Spotify Data Exploration Dashboard — CMSE 830 Midterm")
st.markdown("""
Compare **two features interactively** across histograms, boxplots, correlations, and scatter plots.  
Includes **data cleaning**, **imputation**, **encoding**, **EDA**, and **PCA + clustering**.
""")

# ========================================
# FILE UPLOAD
# ========================================
st.sidebar.header("📂 Upload CSV Files")
uploaded_files = st.sidebar.file_uploader("Upload up to 2 CSV files", accept_multiple_files=True, type="csv")

if not uploaded_files:
    st.warning("Please upload at least one CSV file.")
    st.stop()

dfs = []
for file in uploaded_files[:2]:
    df_temp = pd.read_csv(file)
    df_temp = df_temp.loc[:, ~df_temp.columns.str.contains('^Unnamed')]
    dfs.append(df_temp)

df = pd.concat(dfs, ignore_index=True)

# ========================================
# DATA CLEANING
# ========================================
if 'release_date' in df.columns:
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    most_common_year = int(df['release_date'].dt.year.mode()[0])
    df['release_date'].fillna(pd.Timestamp(f"{most_common_year}-01-01"), inplace=True)
    df['release_year'] = df['release_date'].dt.year

if st.sidebar.checkbox("Drop Duplicates", value=True):
    df = df.drop_duplicates()

drop_thresh = st.sidebar.slider("Drop columns with > x% missing", 50, 100, 95)
col_missing = df.isnull().mean() * 100
df = df.drop(columns=col_missing[col_missing > drop_thresh].index)

# ========================================
# IMPUTATION
# ========================================
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
cat_cols = df.select_dtypes(include=['object', 'string']).columns

before_missing = df.isnull().sum()

if len(num_cols) > 0:
    num_strategy = st.sidebar.selectbox("Numeric imputation strategy", ["mean", "median", "most_frequent"])
    df[num_cols] = SimpleImputer(strategy=num_strategy).fit_transform(df[num_cols])

if len(cat_cols) > 0:
    cat_strategy = st.sidebar.selectbox("Categorical imputation strategy", ["most_frequent", "constant"])
    df[cat_cols] = SimpleImputer(strategy=cat_strategy, fill_value="Unknown" if cat_strategy == "constant" else None).fit_transform(df[cat_cols])

# ========================================
# ENCODING
# ========================================
for col in cat_cols:
    if df[col].nunique() <= 10:
        df = pd.get_dummies(df, columns=[col], prefix=col)
    else:
        freq = df[col].value_counts(normalize=True)
        df[col + "_freq"] = df[col].map(freq)

# ========================================
# FEATURE COMPARISON & CORRELATION
# ========================================
st.header("📊 Feature Comparison & Correlation Explorer")

num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

if len(num_cols) >= 2:
    feature1 = st.selectbox("Select Feature 1", num_cols, index=0)
    feature2 = st.selectbox("Select Feature 2", num_cols, index=1)
    viz_type = st.radio("Choose Visualization Type", ["Histogram Comparison", "Box Plot Comparison", "Correlation Heatmap", "Scatter Plot"])

    if viz_type == "Histogram Comparison":
        fig = px.histogram(
            df, x=feature1, color_discrete_sequence=["#ff7f50"], nbins=40, opacity=0.6, marginal="rug", title=f"Histogram of {feature1}"
        )
        fig2 = px.histogram(
            df, x=feature2, color_discrete_sequence=["#1f77b4"], nbins=40, opacity=0.6, marginal="rug", title=f"Histogram of {feature2}"
        )
        fig.update_traces(opacity=0.5)
        fig2.update_traces(opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)

    elif viz_type == "Box Plot Comparison":
        df_melt = df[[feature1, feature2]].melt(var_name="Feature", value_name="Value")
        fig = px.box(df_melt, x="Feature", y="Value", color="Feature", title="Box Plot Comparison")
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Correlation Heatmap":
        corr = df[[feature1, feature2]].corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title=f"Correlation: {feature1} vs {feature2}")
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Scatter Plot":
        fig = px.scatter(
            df, x=feature1, y=feature2, color_discrete_sequence=["#00cc96"],
            title=f"Scatter Plot: {feature1} vs {feature2}"
        )
        st.plotly_chart(fig, use_container_width=True)

# ========================================
# PCA & CLUSTERING
# ========================================
st.header("📈 PCA + KMeans Clustering")

pca_cols = [c for c in ['valence', 'energy', 'danceability', 'tempo', 'loudness', 'duration_ms', 'popularity'] if c in df.columns]
if len(pca_cols) >= 2:
    X = df[pca_cols].dropna()
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    k = st.slider("Select number of clusters (k)", 2, 8, 3)
    labels = KMeans(n_clusters=k, random_state=42).fit_predict(X_pca)
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = labels

    fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', title="🎨 PCA Clustering Visualization", color_continuous_scale='viridis')
    st.plotly_chart(fig, use_container_width=True)

# ========================================
# EXPORT
# ========================================
df['processed_by'] = "Rohan Rathi — CMSE 830 Midterm"
csv = df.to_csv(index=False)
st.download_button("📥 Download Cleaned CSV", data=csv, file_name="spotify_cleaned.csv", mime="text/csv")

st.success("✅ Enhanced Dashboard Ready — Compare Two Features & Explore Correlations")




