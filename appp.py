import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =======================
# Streamlit Page Setup
# =======================
st.set_page_config(
    page_title="Spotify Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🎵 Spotify Data Exploration Dashboard")
st.markdown(
    """
    Explore your Spotify dataset with interactive visualizations.
    Use the sidebar to filter by song.
    """
)

# =======================
# Load Dataset
# =======================
file_path = r"C:\Users\anand\Downloads\spotify-dashboard\data.csv"
df = pd.read_csv(file_path)

# Ensure release_date is datetime
if 'release_date' in df.columns:
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = df['release_date'].dt.year

# =======================
# Sidebar Filter
# =======================
st.sidebar.header("📋 Filter by Song")
if 'name' in df.columns:
    songs = sorted(df['name'].dropna().unique())
    selected_song = st.sidebar.selectbox("Select Song", ["All"] + songs)
    if selected_song != "All":
        df = df[df['name'] == selected_song]

# =======================
# Dataset Preview & Info
# =======================
with st.expander("📄 Dataset Preview & Info", expanded=True):
    st.subheader("Preview")
    st.dataframe(df.head())
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])

    st.subheader("Columns & Data Types")
    st.write("Numerical Columns:", list(df.select_dtypes(include=['int64', 'float64']).columns))
    st.write("Categorical Columns:", list(df.select_dtypes(include=['object', 'string']).columns))

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum())

# =======================
# Top Artists & Songs
# =======================
with st.expander("🎤 Top Artists & Songs"):
    col1, col2 = st.columns(2)
    with col1:
        if 'artists' in df.columns:
            st.subheader("Top 10 Artists")
            top_artists = df['artists'].value_counts().head(10)
            st.bar_chart(top_artists)

    with col2:
        if 'name' in df.columns:
            st.subheader("Top 10 Songs")
            top_songs = df['name'].value_counts().head(10)
            st.bar_chart(top_songs)

# =======================
# Correlation Heatmap
# =======================
with st.expander("📊 Correlation Heatmap"):
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(num_cols) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        st.pyplot(plt)
        plt.clf()

# =======================
# Top Artists by Popularity
# =======================
with st.expander("🏆 Top 10 Artists by Average Popularity"):
    if 'artists' in df.columns and 'popularity' in df.columns:
        top_artists_pop = df.groupby('artists')['popularity'].mean().sort_values(ascending=False).head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_artists_pop.values, y=top_artists_pop.index, palette='magma')
        plt.xlabel("Average Popularity")
        plt.ylabel("Artist")
        st.pyplot(plt)
        plt.clf()

# =======================
# Popularity Over Time
# =======================
with st.expander("📈 Average Popularity Over Years"):
    if 'release_year' in df.columns and 'popularity' in df.columns:
        yearly_pop = df.groupby('release_year')['popularity'].mean()
        plt.figure(figsize=(12, 5))
        sns.lineplot(x=yearly_pop.index, y=yearly_pop.values, marker='o', color='teal')
        plt.xlabel("Year")
        plt.ylabel("Average Popularity")
        plt.grid(True)
        st.pyplot(plt)
        plt.clf()

# =======================
# Boxplots for Numerical Features
# =======================
with st.expander("📦 Boxplots for Numerical Features"):
    num_features = ['valence', 'tempo', 'energy', 'loudness', 'duration_ms']
    existing_num_features = [col for col in num_features if col in df.columns]
    if existing_num_features:
        plt.figure(figsize=(15, 7))
        for i, col in enumerate(existing_num_features, 1):
            plt.subplot(2, 3, i)
            sns.boxplot(x=df[col], color='skyblue')
            plt.title(f'{col} Outliers')
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()

# =======================
# Histograms for Features
# =======================
with st.expander("📊 Feature Distributions"):
    hist_features = ['valence', 'energy', 'tempo', 'loudness', 'danceability']
    existing_hist_features = [col for col in hist_features if col in df.columns]
    if existing_hist_features:
        plt.figure(figsize=(15, 7))
        for i, col in enumerate(existing_hist_features, 1):
            plt.subplot(2, 3, i)
            sns.histplot(df[col], kde=True, color='coral')
            plt.title(f'{col} Distribution')
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()

st.success("Name - Rohan Rathi | CMSE Midterm Project")
