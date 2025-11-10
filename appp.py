import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore

st.set_page_config(page_title="Spotify Dashboard — CMSE 830 Midterm", layout="wide")

st.title("🎵 Spotify Data Exploration Dashboard — CMSE 830 Midterm")
st.markdown("""
Interactive dashboard with **data cleaning**, **imputation**, **encoding**, **EDA**, **PCA + clustering**, 
and **advanced techniques** like **feature scaling**, **outlier detection**, and **regression analysis**.
""")

# -------------------------------
# File Upload
# -------------------------------
st.sidebar.header("📂 Upload CSV Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload up to 2 Spotify CSV files",
    accept_multiple_files=True,
    type="csv"
)

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

# -------------------------------
# Dataset Overview
# -------------------------------
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

# -------------------------------
# Data Cleaning
# -------------------------------
st.sidebar.header("🧹 Data Cleaning")

if "release_date" in df.columns:
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year
    missing_dates = df["release_date"].isna().sum()
    if missing_dates > 0:
        most_common_year = int(df["release_date"].dt.year.mode()[0])
        df["release_date"] = df["release_date"].fillna(pd.Timestamp(f"{most_common_year}-01-01"))

if st.sidebar.checkbox("Drop duplicates", True):
    df = df.drop_duplicates()

thresh = st.sidebar.slider("Drop columns with > x% missing", 50, 100, 95)
col_missing = df.isnull().mean() * 100
drop_cols = col_missing[col_missing > thresh].index
if len(drop_cols) > 0:
    df = df.drop(columns=drop_cols)

# -------------------------------
# Imputation
# -------------------------------
st.sidebar.header("🩺 Missing Value Imputation")

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

# -------------------------------
# Encoding
# -------------------------------
for col in cat_cols:
    if df[col].nunique() <= 10:
        df = pd.get_dummies(df, columns=[col], prefix=col)
    else:
        freq = df[col].value_counts(normalize=True)
        df[col + "_freq"] = df[col].map(freq)

# -------------------------------
# Statistical Exploration
# -------------------------------
st.header("📊 Statistical & Advanced Exploration")

num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

if len(num_cols) > 0:
    st.subheader("📈 Explore a Numerical Feature")
    selected_num = st.selectbox("Select a numerical column", num_cols)
    stats = df[selected_num].describe()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean", f"{stats['mean']:.2f}")
    with col2:
        st.metric("Median", f"{df[selected_num].median():.2f}")
    with col3:
        st.metric("Std. Dev.", f"{stats['std']:.2f}")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.histplot(df[selected_num], kde=True, color="coral", ax=ax)
    ax.set_title(f"Distribution of {selected_num}")
    st.pyplot(fig)

# -------------------------------
# Outlier Detection
# -------------------------------
st.subheader("🚨 Outlier Detection")
if len(num_cols) > 0:
    selected_outlier = st.selectbox("Select feature for outlier detection", num_cols)
    q1, q3 = df[selected_outlier].quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers = df[(df[selected_outlier] < q1 - 1.5 * iqr) | (df[selected_outlier] > q3 + 1.5 * iqr)]
    st.write(f"Detected **{outliers.shape[0]}** outliers in **{selected_outlier}**")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x=df[selected_outlier], color="skyblue", ax=ax)
    ax.set_title(f"Outliers in {selected_outlier}")
    st.pyplot(fig)

# -------------------------------
# Regression Explorer
# -------------------------------
st.subheader("📉 Linear Relationship Explorer")
if len(num_cols) >= 2:
    x_col = st.selectbox("Select X variable", num_cols)
    y_col = st.selectbox("Select Y variable", num_cols)
    sns.lmplot(x=x_col, y=y_col, data=df, height=5, aspect=1.5, scatter_kws={"alpha": 0.6})
    plt.title(f"Regression Fit: {y_col} vs {x_col}")
    st.pyplot(plt)
    plt.clf()

# -------------------------------
# PCA & Clustering
# -------------------------------
st.header("📈 PCA and Clustering")

num_data = df.select_dtypes(include=["float64", "int64"])
if len(num_data.columns) > 1:
    with st.expander("Correlation Heatmap"):
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(num_data.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

pca_cols = [c for c in ["valence", "energy", "danceability", "tempo", "loudness", "duration_ms", "popularity"] if c in df.columns]

if len(pca_cols) >= 2:
    st.subheader("⚙️ PCA + KMeans Clustering")
    X = df[pca_cols].dropna()
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    k = st.slider("Select number of clusters (k)", 2, 8, 3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_pca)
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", s=40)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Projection with KMeans Clusters")
    st.pyplot(fig)
    st.write("Explained variance ratio:", pca.explained_variance_ratio_.round(3))
    loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"], index=pca_cols)
    st.dataframe(loadings.round(3))

# -------------------------------
# Advanced Data Techniques
# -------------------------------
st.header("🧠 Advanced Data Techniques")

# Z-Score Outlier Removal
st.subheader("1️⃣ Z-Score Based Outlier Removal")
if len(num_cols) > 0:
    z_threshold = st.slider("Select Z-score threshold", 1.5, 4.0, 3.0)
    before_rows = df.shape[0]
    df = df[(np.abs(zscore(df[num_cols])) < z_threshold).all(axis=1)]
    after_rows = df.shape[0]
    st.write(f"Removed **{before_rows - after_rows}** rows using Z-score threshold {z_threshold}")

# Feature Scaling
st.subheader("2️⃣ Feature Scaling Visualization")
if len(num_cols) > 0:
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols)
    st.line_chart(scaled_df.head(50))

# Simple Regression
st.subheader("3️⃣ Simple Regression Analysis")
if len(num_cols) >= 2:
    x_feat = st.selectbox("Independent variable (X)", num_cols)
    y_feat = st.selectbox("Dependent variable (Y)", num_cols)
    model = LinearRegression()
    model.fit(df[[x_feat]], df[y_feat])
    r2 = model.score(df[[x_feat]], df[y_feat])
    st.write(f"**R² Score:** {r2:.3f}")
    sns.regplot(x=x_feat, y=y_feat, data=df, scatter_kws={"alpha": 0.5})
    plt.title(f"Regression: {y_feat} vs {x_feat}")
    st.pyplot(plt)
    plt.clf()

# -------------------------------
# Export
# -------------------------------
df["processed_by"] = "Rohan Rathi — CMSE 830 Midterm"
csv = df.to_csv(index=False)
st.download_button("📥 Download Cleaned CSV", data=csv, file_name="spotify_cleaned.csv", mime="text/csv")

st.success("✅ Project Completed — CMSE 830 Midterm Advanced Version")

