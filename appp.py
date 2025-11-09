import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.set_page_config(page_title="Spotify Dashboard — CMSE 830 Midterm", layout="wide")

st.title("🎵 Spotify Data Exploration Dashboard — CMSE 830 Midterm")
st.markdown("""
Interactive version with **data cleaning**, **imputation**, **encoding**, **EDA**, **PCA + clustering**, and new **interactive filters & visuals**.
""")

st.sidebar.header("📂 Upload CSV Files")
uploaded_files = st.sidebar.file_uploader(
    "Choose up to 2 Spotify CSV files",
    accept_multiple_files=True,
    type="csv"
)

if not uploaded_files:
    st.warning("Please upload at least one CSV file.")
    st.stop()

if len(uploaded_files) > 2:
    uploaded_files = uploaded_files[:2]

dfs = []
for file in uploaded_files:
    try:
        dfs.append(pd.read_csv(file))
    except Exception as e:
        st.error(f"Error reading {file.name}: {e}")

df = pd.concat(dfs, ignore_index=True)

with st.expander("📄 Dataset Preview & Info", expanded=True):
    st.dataframe(df.head())
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    st.write("Missing Values Summary:")
    st.dataframe(df.isnull().sum().sort_values(ascending=False).head(20))

st.sidebar.header("🧹 Data Cleaning")

if 'release_date' in df.columns:
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = df['release_date'].dt.year

if st.sidebar.checkbox("Drop duplicates", value=True):
    df = df.drop_duplicates()

drop_thresh = st.sidebar.slider("Drop columns with > x% missing", 50, 100, 95)
col_missing = df.isnull().mean() * 100
to_drop = col_missing[col_missing > drop_thresh].index
if len(to_drop) > 0:
    df = df.drop(columns=to_drop)

st.sidebar.header("🩺 Missing Value Imputation")

num_cols = df.select_dtypes(include=['float64', 'int64']).columns
cat_cols = df.select_dtypes(include=['object', 'string']).columns

num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

if len(num_cols) > 0:
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

if len(cat_cols) > 0:
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

for col in cat_cols:
    if df[col].nunique() <= 10:
        df = pd.get_dummies(df, columns=[col], prefix=col)
    else:
        freq = df[col].value_counts(normalize=True)
        df[col + "_freq"] = df[col].map(freq)

with st.expander("📊 Cleaned Data Summary", expanded=True):
    st.write("Shape after cleaning:", df.shape)
    st.dataframe(df.describe().T)
    st.dataframe(df.isnull().sum().sort_values(ascending=False).head(10))

st.sidebar.header("🎚️ Interactive Filters")

if 'release_year' in df.columns:
    years = sorted(df['release_year'].dropna().unique())
    year_range = st.sidebar.slider("Select Year Range", int(min(years)), int(max(years)), (int(min(years)), int(max(years))))
    df = df[(df['release_year'] >= year_range[0]) & (df['release_year'] <= year_range[1])]

if 'popularity' in df.columns:
    pop_min, pop_max = int(df['popularity'].min()), int(df['popularity'].max())
    pop_range = st.sidebar.slider("Popularity Range", pop_min, pop_max, (pop_min, pop_max))
    df = df[(df['popularity'] >= pop_range[0]) & (df['popularity'] <= pop_range[1])]

if 'artists' in df.columns:
    top_artists = df['artists'].value_counts().head(20).index.tolist()
    selected_artist = st.sidebar.selectbox("Filter by Artist (optional)", ["All"] + top_artists)
    if selected_artist != "All":
        df = df[df['artists'] == selected_artist]

st.sidebar.button("🔄 Reset Filters")

st.header("🎨 Interactive Exploratory Data Analysis")

num_data_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
if len(num_data_cols) >= 2:
    st.subheader("📊 Custom Scatter Plot")
    x_axis = st.selectbox("Select X-axis", num_data_cols, index=0)
    y_axis = st.selectbox("Select Y-axis", num_data_cols, index=1)
    color_col = st.selectbox("Color By (optional)", ["None"] + num_data_cols)
    plt.figure(figsize=(8, 5))
    if color_col != "None":
        sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=color_col, palette='viridis', s=60)
    else:
        sns.scatterplot(data=df, x=x_axis, y=y_axis, color='royalblue', s=60)
    plt.title(f"{y_axis} vs {x_axis}")
    plt.grid(True)
    st.pyplot(plt)
    plt.clf()

if 'artists' in df.columns:
    top_n = st.slider("Number of Top Artists to Display", 5, 20, 10)
    st.subheader(f"🎤 Top {top_n} Artists")
    top_artists_df = df['artists'].value_counts().head(top_n)
    st.bar_chart(top_artists_df)

st.header("📈 Correlation and PCA Clustering")

num_data = df.select_dtypes(include=['float64', 'int64'])
if len(num_data.columns) > 1:
    with st.expander("Correlation Heatmap"):
        plt.figure(figsize=(10, 7))
        sns.heatmap(num_data.corr(), annot=True, cmap='coolwarm')
        st.pyplot(plt)
        plt.clf()

pca_cols = [c for c in ['valence', 'energy', 'danceability', 'tempo', 'loudness', 'duration_ms', 'popularity']
            if c in df.columns]

if len(pca_cols) >= 2:
    st.subheader("⚙️ PCA + KMeans Clustering")
    X = df[pca_cols].dropna()
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    k = st.slider("Select number of clusters (k)", 2, 8, 3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_pca)
    plt.figure(figsize=(8, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=40)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Projection with KMeans Clusters")
    st.pyplot(plt)
    plt.clf()
    st.write("Explained variance ratio:", pca.explained_variance_ratio_.round(3))

df['processed_by'] = "Rohan Rathi — CMSE 830 Midterm"
csv = df.to_csv(index=False)
st.download_button("📥 Download Cleaned CSV", data=csv, file_name="spotify_cleaned.csv", mime="text/csv")

st.success("✅ Project Completed — CMSE 830 Midterm Interactive Version")
