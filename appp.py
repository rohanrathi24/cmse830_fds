import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import zscore

st.set_page_config(page_title="Spotify Dashboard — CMSE 830 Midterm", layout="wide")

st.title("🎵 Spotify Data Exploration Dashboard — CMSE 830 Midterm")
st.markdown("""
Interactive version with **data cleaning**, **imputation**, **encoding**, **EDA**, **PCA + clustering**, 
and advanced features like **Feature Importance, Outlier Detection, and Regression Analysis**.
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
        temp_df = pd.read_csv(file)
        temp_df = temp_df.loc[:, ~temp_df.columns.str.contains('^Unnamed')]
        dfs.append(temp_df)
    except Exception as e:
        st.error(f"Error reading {file.name}: {e}")

df = pd.concat(dfs, ignore_index=True)

with st.expander("📄 Dataset Overview", expanded=True):
    st.subheader("🧭 Dataset Overview")
    st.metric("Rows", df.shape[0])
    st.metric("Columns", df.shape[1])

    all_columns = df.columns.tolist()
    search_term = st.text_input("🔍 Search or filter columns", "")
    filtered_cols = [c for c in all_columns if search_term.lower() in c.lower()] if search_term else all_columns
    st.write(f"Showing {len(filtered_cols)} of {len(all_columns)} columns:")

    selected_cols = st.multiselect("Select columns to view sample data", filtered_cols, default=filtered_cols[:5])
    st.dataframe(df[selected_cols].head(10))

    col_info = pd.DataFrame({
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum(),
        "Unique Values": df.nunique()
    })
    st.write("📊 Column Metadata")
    st.dataframe(col_info.loc[filtered_cols])

    data_type_counts = df.dtypes.value_counts()
    plt.figure(figsize=(5, 5))
    plt.pie(data_type_counts, labels=data_type_counts.index, autopct="%1.1f%%", startangle=140)
    plt.title("Distribution of Data Types")
    st.pyplot(plt)
    plt.clf()

st.sidebar.header("🧹 Data Cleaning")

if 'release_date' in df.columns:
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = df['release_date'].dt.year

    missing_dates = df['release_date'].isna().sum()
    if missing_dates > 0:
        most_common_year = int(df['release_date'].dt.year.mode()[0])
        df['release_date'] = df['release_date'].fillna(pd.Timestamp(f"{most_common_year}-01-01"))

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

with st.expander("🔍 Missing Data Overview", expanded=True):
    missing_counts = df.isnull().sum()
    missing_percent = (missing_counts / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"Missing Values": missing_counts, "Missing (%)": missing_percent})
    st.dataframe(missing_df[missing_df["Missing Values"] > 0].sort_values("Missing Values", ascending=False))
    if missing_df["Missing Values"].sum() == 0:
        st.success("✅ No missing values found!")

if len(num_cols) > 0:
    st.subheader("⚙️ Numeric Imputation")
    numeric_strategy = st.selectbox("Select strategy for numeric columns", ["mean", "median", "most_frequent"])
    num_imputer = SimpleImputer(strategy=numeric_strategy)
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

if len(cat_cols) > 0:
    st.subheader("🧩 Categorical Imputation")
    cat_strategy = st.selectbox("Select strategy for categorical columns", ["most_frequent", "constant"])
    cat_imputer = SimpleImputer(strategy=cat_strategy, fill_value="Unknown" if cat_strategy == "constant" else None)
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

for col in cat_cols:
    if df[col].nunique() <= 10:
        df = pd.get_dummies(df, columns=[col], prefix=col)
    else:
        freq = df[col].value_counts(normalize=True)
        df[col + "_freq"] = df[col].map(freq)

st.header("📊 Statistical & Advanced Exploration")

num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

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
    plt.figure(figsize=(5, 4))
    sns.histplot(df[selected_num], kde=True, color="coral")
    plt.title(f"Distribution of {selected_num}")
    st.pyplot(plt)
    plt.clf()

st.subheader("🚨 Outlier Detection")
if len(num_cols) > 0:
    selected_outlier_feature = st.selectbox("Select feature for outlier detection", num_cols)
    q1 = df[selected_outlier_feature].quantile(0.25)
    q3 = df[selected_outlier_feature].quantile(0.75)
    iqr = q3 - q1
    outliers = df[(df[selected_outlier_feature] < q1 - 1.5 * iqr) | (df[selected_outlier_feature] > q3 + 1.5 * iqr)]
    st.write(f"Found **{outliers.shape[0]}** outliers in **{selected_outlier_feature}**")
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[selected_outlier_feature], color='lightblue')
    plt.title(f"Outliers in {selected_outlier_feature}")
    st.pyplot(plt)
    plt.clf()

st.subheader("📉 Linear Relationship Explorer")
if len(num_cols) >= 2:
    x_col = st.selectbox("Select X feature", num_cols)
    y_col = st.selectbox("Select Y feature", num_cols)
    sns.lmplot(x=x_col, y=y_col, data=df, height=5, aspect=1.5, scatter_kws={'alpha':0.6})
    plt.title(f"Regression Fit: {y_col} vs {x_col}")
    st.pyplot(plt)
    plt.clf()

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

    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=pca_cols)
    st.subheader("🔍 Feature Contributions to PCA Components")
    st.dataframe(loadings.round(3))

df['processed_by'] = "Rohan Rathi — CMSE 830 Midterm"
csv = df.to_csv(index=False)
st.download_button("📥 Download Cleaned CSV", data=csv, file_name="spotify_cleaned.csv", mime="text/csv")

st.success("✅ Project Completed — CMSE 830 Midterm Advanced Version")





