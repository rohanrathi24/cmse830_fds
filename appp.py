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
An **interactive data exploration dashboard** for analyzing Spotify datasets.  
Includes **cleaning**, **imputation**, **encoding**, **EDA**, **PCA & Clustering**, and **correlation analysis**.
""")

# ========================================
# FILE UPLOAD
# ========================================
st.sidebar.header("📂 Upload CSV Files")
uploaded_files = st.sidebar.file_uploader("Upload up to 2 Spotify CSV files", accept_multiple_files=True, type="csv")

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
# DATASET OVERVIEW
# ========================================
with st.expander("📄 Dataset Overview", expanded=True):
    st.subheader("🧭 Dataset Overview")
    st.metric("Rows", df.shape[0])
    st.metric("Columns", df.shape[1])

    search_term = st.text_input("🔍 Search or filter columns")
    all_columns = df.columns.tolist()
    filtered_cols = [c for c in all_columns if search_term.lower() in c.lower()] if search_term else all_columns
    st.write(f"Showing {len(filtered_cols)} of {len(all_columns)} columns:")

    selected_cols = st.multiselect("Select columns to view sample data", filtered_cols, default=filtered_cols[:5])
    st.dataframe(df[selected_cols].head(10))

    col_info = pd.DataFrame({
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum(),
        "Unique Values": df.nunique()
    })
    st.dataframe(col_info.loc[filtered_cols])

# ========================================
# DATA CLEANING
# ========================================
st.sidebar.header("🧹 Data Cleaning")

if 'release_date' in df.columns:
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    most_common_year = int(df['release_date'].dt.year.mode()[0])
    df['release_date'].fillna(pd.Timestamp(f"{most_common_year}-01-01"), inplace=True)
    df['release_year'] = df['release_date'].dt.year

if st.sidebar.checkbox("Drop duplicates", value=True):
    df = df.drop_duplicates()

drop_thresh = st.sidebar.slider("Drop columns with > x% missing", 50, 100, 95)
col_missing = df.isnull().mean() * 100
df = df.drop(columns=col_missing[col_missing > drop_thresh].index)

st.header("🎤 Top Artists & Songs Exploration")

if "artist" in df.columns or "artists" in df.columns:
    artist_col = "artist" if "artist" in df.columns else "artists"
    
    top_n = st.slider("Select number of top artists", 5, 30, 10)
    top_artists = df[artist_col].value_counts().head(top_n)
    
    fig = px.bar(top_artists, 
                 x=top_artists.values, 
                 y=top_artists.index,
                 orientation='h', 
                 color=top_artists.values,
                 color_continuous_scale="Viridis",
                 title=f"Top {top_n} Most Frequent Artists")
    st.plotly_chart(fig, use_container_width=True)
    
    selected_artist = st.selectbox("Select an artist to view their songs", top_artists.index)
    artist_songs = df[df[artist_col] == selected_artist]
    
    st.write(f"🎵 Showing **{artist_songs.shape[0]}** songs for **{selected_artist}**")
    
    song_cols = [c for c in ["track_name", "song_name", "name", "popularity", "danceability", "energy", "tempo"] if c in df.columns]
    if song_cols:
        st.dataframe(
            artist_songs[song_cols]
            .sort_values(by="popularity" if "popularity" in song_cols else song_cols[0], ascending=False)
            .head(20)
        )
    else:
        st.dataframe(artist_songs.head(20))
else:
    st.warning("No artist column found in dataset. Please ensure your dataset includes an 'artist' or 'artists' column.")


# ========================================
# IMPUTATION
# ========================================
st.sidebar.header("🩺 Missing Value Imputation")

num_cols = df.select_dtypes(include=['float64', 'int64']).columns
cat_cols = df.select_dtypes(include=['object', 'string']).columns

before_missing = df.isnull().sum()

if len(num_cols) > 0:
    numeric_strategy = st.sidebar.selectbox("Numeric imputation strategy", ["mean", "median", "most_frequent"])
    df[num_cols] = SimpleImputer(strategy=numeric_strategy).fit_transform(df[num_cols])

if len(cat_cols) > 0:
    cat_strategy = st.sidebar.selectbox("Categorical imputation strategy", ["most_frequent", "constant"])
    df[cat_cols] = SimpleImputer(strategy=cat_strategy, fill_value="Unknown" if cat_strategy == "constant" else None).fit_transform(df[cat_cols])

after_missing = df.isnull().sum()
comparison = pd.DataFrame({"Before": before_missing, "After": after_missing}).query("Before > 0 or After > 0")

if not comparison.empty:
    fig = px.bar(
        comparison.reset_index().melt(id_vars='index', var_name='Stage', value_name='Missing Values'),
        x='index', y='Missing Values', color='Stage',
        title="📉 Missing Values Before vs After Imputation",
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.success("✅ No missing values remaining!")

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
# EXPLORATORY DATA ANALYSIS (EDA)
# ========================================
st.header("📊 Exploratory Data Analysis (EDA)")

num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

# 📈 Histogram Section
st.subheader("📈 Histogram Visualization")
if len(num_cols) > 0:
    selected_hist = st.selectbox("Select a feature for histogram", num_cols, key="hist_feature")
    fig = px.histogram(df, x=selected_hist, nbins=40, color_discrete_sequence=["#FF7F50"])
    st.plotly_chart(fig, use_container_width=True)

# 📦 Box Plot Section
st.subheader("📦 Box Plot Visualization")
if len(num_cols) > 0:
    selected_box = st.selectbox("Select a feature for box plot", num_cols, key="box_feature")
    fig = px.box(df, y=selected_box, color_discrete_sequence=["#00CC96"])
    st.plotly_chart(fig, use_container_width=True)

# ⚫ Scatter Plot Section
st.subheader("⚫ Scatter Plot Visualization")
if len(num_cols) > 1:
    x_scatter = st.selectbox("Select X-axis feature", num_cols, key="x_feature")
    y_scatter = st.selectbox("Select Y-axis feature", num_cols, index=1, key="y_feature")
    fig = px.scatter(df, x=x_scatter, y=y_scatter, color_discrete_sequence=["#1F77B4"])
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Scatter plot requires at least two numeric features.")
# ========================================
# FEATURE CORRELATION COMPARISON
# ========================================
st.header("🔗 Feature Correlation Comparison")

if len(num_cols) >= 2:
    feature1 = st.selectbox("Feature 1", num_cols, index=0)
    feature2 = st.selectbox("Feature 2", num_cols, index=1)

    if feature1 == feature2:
        st.warning("⚠️ Please select two different features to compare.")
    else:
        corr = df[[feature1, feature2]].corr()
        corr_value = corr.iloc[0, 1]

        fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                        title=f"Correlation Heatmap — {feature1} vs {feature2}")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"📈 **Correlation Coefficient:** `{corr_value:.2f}`")
        if abs(corr_value) > 0.7:
            st.success("✅ Strong correlation between features!")
        elif abs(corr_value) > 0.4:
            st.info("🟨 Moderate correlation.")
        else:
            st.warning("🔹 Weak or no correlation detected.")

# ========================================
# PCA + CLUSTERING
# ========================================
st.header("📉 PCA + KMeans Clustering")

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

    fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', title="🎨 PCA Clustering Visualization")
    st.plotly_chart(fig, use_container_width=True)

# ========================================
# EXPORT
# ========================================
df['processed_by'] = "Rohan Rathi — CMSE 830 Midterm"
csv = df.to_csv(index=False)
st.download_button("📥 Download Cleaned CSV", data=csv, file_name="spotify_cleaned.csv", mime="text/csv")

st.success("✅ Complete Dashboard — Cleaning, EDA, Correlation, and PCA Ready!")
