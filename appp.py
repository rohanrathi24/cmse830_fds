# ========================================
# 🎵 Spotify Data Exploration Dashboard — CMSE 830 Midterm
# ========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore

# --------------------------------
# Streamlit Page Setup
# --------------------------------
st.set_page_config(page_title="Spotify Dashboard — CMSE 830 Midterm", layout="wide")

st.title("🎵 Spotify Data Exploration Dashboard — CMSE 830 Midterm")
st.markdown("""
Interactive dashboard for **data cleaning**, **EDA**, **PCA + clustering**, **feature scaling**, **outlier detection**, 
and **regression analysis** — designed for CMSE 830 Midterm.
""")


# ========================================
# 🎛️ Sidebar Navigation & Upload
# ========================================
with st.sidebar:
    st.markdown("## 🎛️ **Dashboard Controls**")
    st.markdown("---")

    # 📂 File Upload
    st.header("📂 Upload CSV Files")
    uploaded_files = st.file_uploader("Upload up to 2 Spotify CSV files", accept_multiple_files=True, type="csv")

    # 🧹 Cleaning Settings
    st.markdown("---")
    st.header("🧹 Data Cleaning Options")
    st.checkbox("Drop duplicates", True, key="drop_dupes")
    thresh = st.slider("Drop columns with > x% missing", 50, 100, 95, key="missing_thresh")

    # 🧭 Quick Navigation
    st.markdown("---")
    st.header("🧭 Quick Navigation")
    st.markdown("""
    - 🩺 [Missing Value Imputation](#🩺-missing-value-imputation)
    - 🎤 [Top Artists](#🎤-top-artists--songs-exploration)
    - 📊 [EDA](#📊-exploratory-data-analysis-eda)
    - 🔗 [Correlation](#🔗-feature-correlation-comparison)
    - 🎨 [PCA + Clustering](#🎨-pca--kmeans-clustering)
    """)

    # 🌈 Theme toggle (visual only)
    theme_choice = st.radio("🎨 Choose theme (visual)", ["Light Mode", "Dark Mode"], horizontal=True)
    if theme_choice == "Dark Mode":
        st.markdown("🌙 **Dark Mode Activated (Preview)**", unsafe_allow_html=True)
    else:
        st.markdown("☀️ **Light Mode Activated (Preview)**", unsafe_allow_html=True)

    # 👤 About section
    st.markdown("---")
    st.markdown("**👤 Created by:** Rohan Rathi  \n📘 *CMSE 830 Midterm Project*")
    st.caption("Use the controls above to explore and clean Spotify data interactively.")


# ========================================
# 📂 Load and Merge Files
# ========================================
if not uploaded_files:
    st.warning("Please upload at least one CSV file.")
    st.stop()

dfs = []
for file in uploaded_files[:2]:
    try:
        temp_df = pd.read_csv(file)
        temp_df = temp_df.loc[:, ~temp_df.columns.str.contains('^Unnamed')]
        dfs.append(temp_df)
    except Exception as e:
        st.error(f"Error reading {file.name}: {e}")

df = pd.concat(dfs, ignore_index=True)

# ========================================
# 📄 Dataset Overview
# ========================================
with st.expander("📄 Dataset Overview", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])

    search = st.text_input("🔍 Search columns", "")
    columns = [c for c in df.columns if search.lower() in c.lower()] if search else df.columns
    selected = st.multiselect("Select columns to view sample", columns, default=columns[:5])
    st.dataframe(df[selected].head(10))

    meta = pd.DataFrame({
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum(),
        "Unique Values": df.nunique()
    })
    st.write("📊 Column Metadata")
    st.dataframe(meta.loc[columns])

# ========================================
# 🧹 Data Cleaning
# ========================================
if "release_date" in df.columns:
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year
    missing_dates = df["release_date"].isna().sum()
    if missing_dates > 0:
        most_common_year = int(df["release_date"].dt.year.mode()[0])
        df["release_date"] = df["release_date"].fillna(pd.Timestamp(f"{most_common_year}-01-01"))

if st.session_state.get("drop_dupes", True):
    df = df.drop_duplicates()

col_missing = df.isnull().mean() * 100
drop_cols = col_missing[col_missing > st.session_state.get("missing_thresh", 95)].index
if len(drop_cols) > 0:
    df = df.drop(columns=drop_cols)

st.subheader("📊 Dataset Summary")

# Basic info
st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")

# Show describe table
desc_table = df.describe().T
st.dataframe(desc_table, use_container_width=True)

# Missing value summary
missing_summary = df.isnull().sum().to_frame("Missing Values")
st.write("### 🔍 Missing Value Overview")
st.dataframe(missing_summary, use_container_width=True)

# ========================================
# 🩺 Missing Value Imputation
# ========================================

num_cols = df.select_dtypes(include=["float64", "int64"]).columns
cat_cols = df.select_dtypes(include=["object", "string"]).columns

before_missing = df.isnull().sum()

if len(num_cols) > 0:
    st.subheader("⚙️ Numeric Imputation")
    num_strategy = st.selectbox("Strategy for numeric columns", ["mean", "median", "most_frequent"])
    num_imputer = SimpleImputer(strategy=num_strategy)
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

if len(cat_cols) > 0:
    st.subheader("🧩 Categorical Imputation")
    cat_strategy = st.selectbox("Strategy for categorical columns", ["most_frequent", "constant"])
    cat_imputer = SimpleImputer(strategy=cat_strategy, fill_value="Unknown" if cat_strategy == "constant" else None)
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

after_missing = df.isnull().sum()

# 📉 Missing Values Comparison
st.subheader("📉 Missing Values Before vs After Imputation")
comparison_df = pd.DataFrame({"Before": before_missing, "After": after_missing})
comparison_df = comparison_df[(comparison_df["Before"] > 0) | (comparison_df["After"] > 0)]
if not comparison_df.empty:
    st.dataframe(comparison_df.sort_values("Before", ascending=False).head(20))
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.4
    x = np.arange(len(comparison_df))
    ax.bar(x - width/2, comparison_df["Before"], width, color="coral", label="Before")
    ax.bar(x + width/2, comparison_df["After"], width, color="skyblue", label="After")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df.index, rotation=90)
    ax.set_ylabel("Missing Values Count")
    ax.set_title("Missing Values Before vs After Imputation")
    ax.legend()
    st.pyplot(fig)
else:
    st.success("✅ No missing values to compare!")

# ========================================
# 🧠 Encoding
# ========================================
for col in cat_cols:
    if df[col].nunique() <= 10:
        df = pd.get_dummies(df, columns=[col], prefix=col)
    else:
        freq = df[col].value_counts(normalize=True)
        df[col + "_freq"] = df[col].map(freq)

# ========================================
# 🎤 Top Artists & Songs Exploration
# ========================================
st.header("🎤 Top Artists & Songs Exploration")

if "artist" in df.columns or "artists" in df.columns:
    artist_col = "artist" if "artist" in df.columns else "artists"
    
    top_n = st.slider("Select number of top artists", 5, 30, 10)
    top_artists = df[artist_col].value_counts().head(top_n)
    
    fig = px.bar(
        top_artists, 
        x=top_artists.values, 
        y=top_artists.index,
        orientation='h', 
        color=top_artists.values,
        color_continuous_scale="Viridis",
        title=f"Top {top_n} Most Frequent Artists"
    )
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
# 📈 Popularity Over the Years
# ========================================
if "release_year" in df.columns and "popularity" in df.columns:
    st.subheader("📈 Popularity Trend Over the Years")
    year_popularity = df.groupby("release_year")["popularity"].mean().dropna()
    fig = px.line(year_popularity, 
                  x=year_popularity.index, 
                  y=year_popularity.values, 
                  markers=True,
                  title="Average Popularity by Release Year",
                  color_discrete_sequence=["#FF69B4"])
    st.plotly_chart(fig, use_container_width=True)


# ========================================
# 📊 Exploratory Data Analysis (EDA)
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
# 🔗 Correlation Analysis
# ========================================
st.header("🔗 Feature Correlation Comparison")

if len(num_cols) >= 2:
    feature1 = st.selectbox("Feature 1", num_cols, index=0)
    feature2 = st.selectbox("Feature 2", num_cols, index=1)
    corr_value = df[[feature1, feature2]].corr().iloc[0, 1]
    fig = px.imshow(df[[feature1, feature2]].corr(), text_auto=True, color_continuous_scale='RdBu_r')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"📈 **Correlation Coefficient:** `{corr_value:.2f}`")
    if abs(corr_value) > 0.7:
        st.success("✅ Strong correlation between features!")
    elif abs(corr_value) > 0.4:
        st.info("🟨 Moderate correlation.")
    else:
        st.warning("🔹 Weak or no correlation detected.")

# ========================================
# 🚨 Outlier Detection + Z-Score Removal
# ========================================
st.subheader("🚨 Outlier Detection and Z-Score Cleaning")

if len(num_cols) > 0:
    selected_outlier = st.selectbox("Select feature for outlier detection", num_cols)
    q1, q3 = df[selected_outlier].quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers = df[(df[selected_outlier] < q1 - 1.5 * iqr) | (df[selected_outlier] > q3 + 1.5 * iqr)]
    st.write(f"Detected **{outliers.shape[0]}** outliers in **{selected_outlier}**")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x=df[selected_outlier], color="skyblue", ax=ax)
    st.pyplot(fig)

    z_threshold = st.slider("Select Z-score threshold", 1.5, 4.0, 3.0)
    before_rows = df.shape[0]
    df = df[(np.abs(zscore(df[num_cols])) < z_threshold).all(axis=1)]
    after_rows = df.shape[0]
    st.write(f"Removed **{before_rows - after_rows}** rows using Z-score threshold {z_threshold}")

# ========================================
# 📈 Regression
# ========================================
st.header("📉 Simple Linear Regression")

if len(num_cols) >= 2:
    x_feat = st.selectbox("Independent variable (X)", num_cols, key="reg_x")
    y_feat = st.selectbox("Dependent variable (Y)", num_cols, key="reg_y")

    model = LinearRegression()
    X = df[[x_feat]]
    y = df[y_feat]
    model.fit(X, y)
    predictions = model.predict(X)
    r2 = model.score(X, y)

    st.write(f"**R² Score:** {r2:.3f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=x_feat, y=y_feat, data=df, alpha=0.5, ax=ax)
    sns.lineplot(x=df[x_feat], y=predictions, color="red", ax=ax)
    ax.set_title(f"Linear Regression: {y_feat} vs {x_feat}")
    st.pyplot(fig)
else:
    st.info("Need at least two numeric features for regression.")


# ========================================
# 📉 PCA + Clustering
# ========================================
st.header("🎨 PCA + KMeans Clustering")

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

    fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', title="PCA Clustering Visualization")
    st.plotly_chart(fig, use_container_width=True)

    st.write("Explained variance ratio:", pca.explained_variance_ratio_.round(3))
    loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"], index=pca_cols)
    st.dataframe(loadings.round(3))

# ========================================
# 📦 Export
# ========================================
df["processed_by"] = "Rohan Rathi — CMSE 830 Midterm"
csv = df.to_csv(index=False)
st.download_button("📥 Download Cleaned CSV", data=csv, file_name="spotify_cleaned.csv", mime="text/csv")

st.success("✅ Complete Dashboard — Cleaning, EDA, Correlation, PCA, Regression, and Artist Analysis Ready!")

