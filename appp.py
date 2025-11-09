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
Interactive version with **data cleaning**, **imputation**, **encoding**, **EDA**, **PCA + clustering**, and new **interactive dataset overview, filters, and visuals**.
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

    before_num_missing = df[num_cols].isnull().sum().sum()
    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    after_num_missing = df[num_cols].isnull().sum().sum()

    st.info(f"Filled {before_num_missing - after_num_missing} missing numeric values using '{numeric_strategy}' strategy.")

if len(cat_cols) > 0:
    st.subheader("🧩 Categorical Imputation")
    cat_strategy = st.selectbox("Select strategy for categorical columns", ["most_frequent", "constant"])
    cat_imputer = SimpleImputer(strategy=cat_strategy, fill_value="Unknown" if cat_strategy == "constant" else None)

    before_cat_missing = df[cat_cols].isnull().sum().sum()
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    after_cat_missing = df[cat_cols].isnull().sum().sum()

    st.info(f"Filled {before_cat_missing - after_cat_missing} missing categorical values using '{cat_strategy}' strategy.")

with st.expander("📉 Missing Values Comparison Before vs After"):
    plt.figure(figsize=(10, 4))
    sns.barplot(x=missing_counts.index, y=missing_counts.values, color="coral", label="Before")
    sns.barplot(x=missing_counts.index, y=df.isnull().sum().values, color="skyblue", label="After")
    plt.xticks(rotation=90)
    plt.title("Missing Values Before vs After Imputation")
    plt.legend()
    st.pyplot(plt)
    plt.clf()

for col in cat_cols:
    if df[col].nunique() <= 10:
        df = pd.get_dummies(df, columns=[col], prefix=col)
    else:
        freq = df[col].value_counts(normalize=True)
        df[col + "_freq"] = df[col].map(freq)

with st.expander("📊 Statistical Exploration", expanded=True):
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()

    if len(num_cols) > 0:
        st.subheader("📈 Explore a Numerical Feature")
        selected_num = st.selectbox("Select a numerical column", num_cols)
        stats = df[selected_num].describe()
        st.metric("Mean", f"{stats['mean']:.2f}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Median", f"{df[selected_num].median():.2f}")
        with col2:
            st.metric("Std. Dev.", f"{stats['std']:.2f}")
        with col3:
            st.metric("Skewness", f"{df[selected_num].skew():.2f}")
        col1, col2 = st.columns(2)
        with col1:
            plt.figure(figsize=(5, 4))
            sns.histplot(df[selected_num], kde=True, color="coral")
            plt.title(f"Distribution of {selected_num}")
            st.pyplot(plt)
            plt.clf()
        with col2:
            plt.figure(figsize=(5, 4))
            sns.boxplot(x=df[selected_num], color="skyblue")
            plt.title(f"Boxplot of {selected_num}")
            st.pyplot(plt)
            plt.clf()

    if len(num_cols) >= 2:
        st.subheader("🔗 Explore Correlations")
        col_x = st.selectbox("Select X feature", num_cols, index=0)
        col_y = st.selectbox("Select Y feature", num_cols, index=1)
        corr_val = df[[col_x, col_y]].corr().iloc[0, 1]
        st.write(f"Correlation between **{col_x}** and **{col_y}**: `{corr_val:.3f}`")
        plt.figure(figsize=(6, 5))
        sns.scatterplot(x=df[col_x], y=df[col_y], color="teal", alpha=0.7)
        plt.title(f"{col_y} vs {col_x}")
        plt.grid(True)
        st.pyplot(plt)
        plt.clf()

    if len(cat_cols) > 0:
        st.subheader("🧩 Categorical Feature Summary")
        selected_cat = st.selectbox("Select a categorical column", cat_cols)
        top_values = df[selected_cat].value_counts().head(10)
        st.bar_chart(top_values)

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



