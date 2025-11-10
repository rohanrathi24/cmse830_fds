import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore

st.set_page_config(page_title="Spotify Dashboard — CMSE 830 Midterm", layout="wide")

st.title("🎵 Spotify Data Exploration Dashboard — CMSE 830 Midterm")
st.markdown("""
An **interactive dashboard** for analyzing Spotify datasets — from **cleaning** and **imputation**
to **EDA**, **correlation exploration**, and **PCA clustering**.
""")


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


with st.expander("📄 Dataset Overview", expanded=True):
    st.subheader("🧭 Overview")
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

    search_term = st.text_input("🔍 Search column names")
    all_cols = df.columns.tolist()
    filtered_cols = [c for c in all_cols if search_term.lower() in c.lower()] if search_term else all_cols

    selected_cols = st.multiselect("Select columns to preview", filtered_cols, default=filtered_cols[:5])
    st.dataframe(df[selected_cols].head(10))

    col_info = pd.DataFrame({
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum(),
        "Unique Values": df.nunique()
    })
    st.dataframe(col_info.loc[filtered_cols])


st.sidebar.header("🧹 Data Cleaning")

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


for col in cat_cols:
    if df[col].nunique() <= 10:
        df = pd.get_dummies(df, columns=[col], prefix=col)
    else:
        freq = df[col].value_counts(normalize=True)
        df[col + "_freq"] = df[col].map(freq)


st.sidebar.header("🎛️ Interactive Filters")

if 'track_genre' in df.columns:
    genre_options = df['track_genre'].dropna().unique().tolist()
    selected_genres = st.sidebar.multiselect("Filter by Genre", genre_options, default=genre_options[:3])
    df = df[df['track_genre'].isin(selected_genres)]

if 'artists' in df.columns:
    artist_options = df['artists'].dropna().unique().tolist()
    selected_artists = st.sidebar.multiselect("Filter by Artist", artist_options[:50])
    if selected_artists:
        df = df[df['artists'].isin(selected_artists)]

num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
if num_cols:
    col_to_filter = st.sidebar.selectbox("Numeric filter column", num_cols)
    min_val, max_val = float(df[col_to_filter].min()), float(df[col_to_filter].max())
    selected_range = st.sidebar.slider(f"Filter {col_to_filter} range", min_val, max_val, (min_val, max_val))
    df = df[(df[col_to_filter] >= selected_range[0]) & (df[col_to_filter] <= selected_range[1])]


st.header("📊 Exploratory Data Analysis (EDA)")

if len(num_cols) > 0:
    selected_feature = st.selectbox("Select a feature for visualization", num_cols)

    tab1, tab2, tab3 = st.tabs(["📈 Histogram", "📦 Box Plot", "⚫ Scatter Plot"])
    with tab1:
        fig = px.histogram(df, x=selected_feature, nbins=40, color_discrete_sequence=["#FF7F50"],
                           title=f"Histogram of {selected_feature}")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = px.box(df, y=selected_feature, color_discrete_sequence=["#00CC96"],
                     title=f"Box Plot of {selected_feature}")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        other_feature = st.selectbox("Select another feature for scatter plot", num_cols, index=1)
        fig = px.scatter(df, x=selected_feature, y=other_feature, color_discrete_sequence=["#1F77B4"],
                         title=f"Scatter Plot: {selected_feature} vs {other_feature}")
        st.plotly_chart(fig, use_container_width=True)


st.header("🔗 Feature Correlation Comparison")

if len(num_cols) >= 2:
    feature1 = st.selectbox("Feature 1", num_cols, index=0)
    feature2 = st.selectbox("Feature 2", num_cols, index=1)

    # Handle identical features gracefully
    if feature1 == feature2:
        st.warning("⚠️ Please select two different features to compare.")
    else:
        corr = df[[feature1, feature2]].corr()
        corr_value = corr.iloc[0, 1]

        fig = px.imshow(
            corr, text_auto=True, color_continuous_scale='RdBu_r',
            title=f"Correlation Heatmap — {feature1} vs {feature2}"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"📈 **Correlation Coefficient:** `{corr_value:.2f}`")

        if abs(corr_value) > 0.7:
            st.success("✅ Strong correlation between features!")
        elif abs(corr_value) > 0.4:
            st.info("🟨 Moderate correlation.")
        else:
            st.warning("🔹 Weak or no correlation detected.")


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


df['processed_by'] = "Rohan Rathi — CMSE 830 Midterm"
csv = df.to_csv(index=False)
st.download_button("📥 Download Cleaned CSV", data=csv, file_name="spotify_cleaned.csv", mime="text/csv")








