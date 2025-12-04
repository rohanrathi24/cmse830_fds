# streamlit_final_with_glow.py
"""
Spotify Final Project — Single-file Streamlit app with glowing navigation
Upgrades added:
1) RandomForest + model comparison + CV/tuning
2) Expanded feature engineering controls (freq-encode, text-length, polynomial, scaling)
3) Export ZIP including cleaned CSV, engineered CSV, model pickle, README, requirements

Author: Rohan Rathi (upgraded)
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle
import zipfile
import textwrap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import zscore
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Optional XGBoost
try:
    from xgboost import XGBRegressor  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="Spotify Final Project — CMSE 830", layout="wide")

# -----------------------
# Initialize session state defaults
# -----------------------
if "page" not in st.session_state:
    st.session_state["page"] = "dataset_overview"
if "df_raw" not in st.session_state:
    st.session_state["df_raw"] = None
if "df" not in st.session_state:
    st.session_state["df"] = None
if "df_fe" not in st.session_state:
    st.session_state["df_fe"] = None
if "best_model_obj" not in st.session_state:
    st.session_state["best_model_obj"] = None
if "best_model_name" not in st.session_state:
    st.session_state["best_model_name"] = None
if "model_features" not in st.session_state:
    st.session_state["model_features"] = None

# -----------------------
# Top glowing navigation UI (same ultra-cool cards)
# -----------------------
st.markdown("""
<style>
.box-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 18px; margin-top: 14px; margin-bottom: 20px; }
.glow-card {
    background: linear-gradient(135deg, rgba(20,20,30,0.96), rgba(10,10,20,0.96));
    padding: 26px; border-radius: 14px; text-align: center; color: white; font-size: 18px; font-weight: 700; cursor: pointer;
    border: 1px solid rgba(255,255,255,0.06); transition: transform 0.22s ease-in-out, box-shadow 0.22s;
    position: relative; overflow: hidden;
}
.glow-card:before {
    content: ""; position: absolute; top: -2px; left: -2px; right: -2px; bottom: -2px;
    background: linear-gradient(60deg, #ff5f6d, #ffc371, #48c6ef, #8a2be2, #ff5f6d);
    background-size: 300% 300%; z-index: -1; filter: blur(10px); opacity: 0; transition: opacity 0.35s;
}
.glow-card:hover { transform: translateY(-8px) scale(1.02); box-shadow: 0 14px 35px rgba(0,0,0,0.45); }
.glow-card:hover:before { opacity: 1; animation: gradientGlow 4s linear infinite; }
.card-label { display:block; font-size:14px; opacity:0.9; font-weight:600; margin-top:6px; }
@keyframes gradientGlow { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
</style>
""", unsafe_allow_html=True)

st.markdown("<div style='display:flex; align-items:center; justify-content:space-between'>"
            "<h1 style='margin:0;padding:0;'>🎵 Spotify Final Project — CMSE 830</h1>"
            "<div style='color:gray;font-size:14px'>by Rohan Rathi</div></div>", unsafe_allow_html=True)

st.markdown('<div class="box-grid">', unsafe_allow_html=True)

nav_cards = [
    ("📄 Dataset Overview", "dataset_overview", "Preview uploaded data & metadata"),
    ("🩺 Missing Imputation", "missing_imputation", "Impute numeric & categorical columns"),
    ("🎤 Top Artists", "top_artists", "Explore artists and songs"),
    ("📊 EDA", "eda", "Histograms, boxplots, scatter"),
    ("🔗 Correlation", "correlation", "Feature correlation & stats"),
    ("🚨 Outliers", "outliers", "Detect & remove outliers"),
    ("🧪 Feature Engineering", "feature_engineering", "Create new features & transforms"),
    ("🤖 Modeling", "modeling", "LinearReg, RandomForest, CV & tuning"),
    ("🎨 PCA + Clustering", "pca_clustering", "PCA and KMeans segmentation"),
    ("📥 Export & ZIP", "export", "Download cleaned, engineered, model, README")
]

for label, pkey, subtitle in nav_cards:
    if st.button(label, key=pkey):
        st.session_state["page"] = pkey
    st.markdown(f'<div class="glow-card"><div style="font-size:18px">{label}</div>'
                f'<div class="card-label">{subtitle}</div></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")

# -----------------------
# Sidebar controls, uploads
# -----------------------
with st.sidebar:
    st.header("📂 Upload & Settings")
    uploaded_files = st.file_uploader("Upload up to 2 Spotify CSV files", accept_multiple_files=True, type="csv")
    remote_csv = st.text_input("Optional: remote CSV URL (raw)", "")
    st.markdown("---")
    st.header("Cleaning Options")
    st.checkbox("Drop duplicates", True, key="drop_dupes")
    missing_thresh = st.slider("Drop columns with > x% missing", 0, 100, 95)
    st.markdown("---")
    st.caption("Use top cards to jump sections. Upload data in this sidebar.")

# -----------------------
# Data loading helpers & synthetic generation
# -----------------------
@st.cache_data
def read_csv_bytes(buf):
    try:
        return pd.read_csv(buf)
    except Exception:
        return None

def load_uploaded(files):
    dfs = []
    for f in files[:2]:
        df_tmp = read_csv_bytes(f)
        if df_tmp is not None:
            df_tmp = df_tmp.loc[:, ~df_tmp.columns.str.contains('^Unnamed')]
            dfs.append(df_tmp)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True, sort=False)

# Load uploaded files
if uploaded_files:
    df_up = load_uploaded(uploaded_files)
    if df_up is not None:
        st.session_state["df_raw"] = df_up

# Load remote URL
if remote_csv:
    try:
        df_remote = pd.read_csv(remote_csv)
        df_remote = df_remote.loc[:, ~df_remote.columns.str.contains('^Unnamed')]
        if st.session_state["df_raw"] is not None:
            st.session_state["df_raw"] = pd.concat([st.session_state["df_raw"], df_remote], ignore_index=True, sort=False)
        else:
            st.session_state["df_raw"] = df_remote
        st.sidebar.success("Loaded remote CSV")
    except Exception as e:
        st.sidebar.error(f"Failed to load remote CSV: {e}")

# If no raw df: allow synthetic generator
if st.session_state["df_raw"] is None:
    st.info("No dataset loaded - upload CSV(s) or use the synthetic demo below.")
    if st.button("Generate synthetic demo dataset (500 rows)"):
        rng = np.random.default_rng(42)
        artists = [f"Artist_{i}" for i in range(1, 51)]
        synth = pd.DataFrame({
            "track_id": [f"t{i}" for i in range(500)],
            "track_name": [f"Track_{i}" for i in range(500)],
            "artist": rng.choice(artists, size=500),
            "popularity": rng.integers(0,101,size=500),
            "danceability": rng.random(500),
            "energy": rng.random(500),
            "valence": rng.random(500),
            "tempo": np.round(rng.normal(120,30,size=500),2).clip(40,220),
            "loudness": np.round(rng.normal(-8,4,size=500),2),
            "duration_ms": rng.integers(120000,300000,size=500),
            "release_date": pd.to_datetime("2016-01-01") + pd.to_timedelta(rng.integers(0,365*6,size=500), unit='D')
        })
        st.session_state["df_raw"] = synth
        st.experimental_rerun()

# -----------------------
# Prepare & clean dataframe and store in session
# -----------------------
def prepare_df(raw_df):
    df = raw_df.copy()
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # release_date coercion
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        if df["release_date"].isnull().any():
            try:
                mode_year = int(df["release_date"].dropna().dt.year.mode().iat[0])
            except Exception:
                mode_year = 2019
            df["release_date"] = df["release_date"].fillna(pd.Timestamp(f"{mode_year}-01-01"))
        df["release_year"] = df["release_date"].dt.year
    # drop columns exceeding missingness threshold
    col_missing = df.isnull().mean() * 100
    drop_cols = col_missing[col_missing > missing_thresh].index.tolist()
    if drop_cols:
        df = df.drop(columns=drop_cols)
    # duplicates
    if st.session_state.get("drop_dupes", True):
        df = df.drop_duplicates()
    return df

if st.session_state["df_raw"] is not None and st.session_state["df"] is None:
    st.session_state["df"] = prepare_df(st.session_state["df_raw"])

# quick alias
df = st.session_state.get("df")

# -----------------------
# Section functions
# -----------------------
def section_dataset_overview():
    st.header("📄 Dataset Overview")
    if df is None:
        st.info("No dataset loaded.")
        return
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    search = st.text_input("🔍 Search columns", key="search_cols")
    columns = [c for c in df.columns if search.lower() in c.lower()] if search else list(df.columns)
    selected = st.multiselect("Select columns to preview", options=columns, default=columns[:6] if columns else [])
    if selected:
        st.dataframe(df[selected].head(10))
    meta = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isnull().sum(), "unique": df.nunique()})
    st.dataframe(meta.sort_values("missing", ascending=False).head(40))

def section_missing_imputation():
    st.header("🩺 Missing Value Imputation")
    if df is None:
        st.info("No data.")
        return
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    before_missing = df.isnull().sum()
    if num_cols:
        st.subheader("Numeric Imputation")
        num_strategy = st.selectbox("Strategy for numeric", ["mean", "median", "most_frequent"], key="num_strat")
        num_imp = SimpleImputer(strategy=num_strategy)
        try:
            df[num_cols] = num_imp.fit_transform(df[num_cols])
            st.success("Numeric imputation applied")
        except Exception as e:
            st.error("Numeric imputation failed: " + str(e))
    if cat_cols:
        st.subheader("Categorical Imputation")
        cat_strategy = st.selectbox("Strategy for categorical", ["most_frequent", "constant"], key="cat_strat")
        fill_val = "Unknown" if cat_strategy == "constant" else None
        cat_imp = SimpleImputer(strategy=cat_strategy, fill_value=fill_val)
        try:
            df[cat_cols] = cat_imp.fit_transform(df[cat_cols])
            st.success("Categorical imputation applied")
        except Exception as e:
            st.error("Categorical imputation failed: " + str(e))
    after_missing = df.isnull().sum()
    comp = pd.DataFrame({"Before": before_missing, "After": after_missing})
    comp = comp[(comp["Before"] > 0) | (comp["After"] > 0)]
    if not comp.empty:
        st.dataframe(comp.sort_values("Before", ascending=False).head(20))

def section_top_artists():
    st.header("🎤 Top Artists & Songs Exploration")
    if df is None:
        st.info("No data.")
        return
    artist_col = "artist" if "artist" in df.columns else ("artists" if "artists" in df.columns else None)
    if not artist_col:
        st.warning("No 'artist' column found.")
        return
    top_n = st.slider("Top N artists", 5, 50, 10)
    top_art = df[artist_col].value_counts().head(top_n)
    fig = px.bar(x=top_art.values, y=top_art.index, orientation="h", title="Top Artists")
    st.plotly_chart(fig, use_container_width=True)
    selected = st.selectbox("Select artist to inspect", top_art.index, key="artist_select")
    artist_songs = df[df[artist_col] == selected]
    song_cols = [c for c in ["track_name", "name", "popularity", "danceability", "energy", "tempo"] if c in df.columns]
    if song_cols:
        st.dataframe(artist_songs[song_cols].sort_values(by=("popularity" if "popularity" in song_cols else song_cols[0]), ascending=False).head(30))
    else:
        st.dataframe(artist_songs.head(30))

def section_eda():
    st.header("📊 Exploratory Data Analysis (EDA)")
    if df is None:
        st.info("No data.")
        return
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    st.subheader("Descriptive (numeric)")
    if num_cols:
        desc = df[num_cols].describe().T
        desc["skew"] = df[num_cols].skew()
        desc["kurtosis"] = df[num_cols].kurtosis()
        st.dataframe(desc.round(3))
    else:
        st.info("No numeric columns for EDA.")
    st.subheader("Visualizations")
    viz = st.selectbox("Choose visualization", ["Histogram", "Boxplot", "Violin", "Scatter + OLS", "Correlation Heatmap", "Time-series (popularity by year)"])
    if viz == "Histogram" and num_cols:
        col = st.selectbox("Histogram column", num_cols, key="hist_col")
        nbins = st.slider("Bins", 10, 100, 40, key="hist_bins")
        st.plotly_chart(px.histogram(df, x=col, nbins=nbins))
    elif viz == "Boxplot" and num_cols:
        col = st.selectbox("Boxplot column", num_cols, key="box_col")
        st.plotly_chart(px.box(df, y=col))
    elif viz == "Violin" and num_cols:
        col = st.selectbox("Violin column", num_cols, key="violin_col")
        st.plotly_chart(px.violin(df, y=col, box=True))
    elif viz == "Scatter + OLS" and len(num_cols) >= 2:
        x = st.selectbox("X", num_cols, key="scatter_x")
        y = st.selectbox("Y", num_cols, key="scatter_y")
        st.plotly_chart(px.scatter(df, x=x, y=y, trendline="ols"))
    elif viz == "Correlation Heatmap" and len(num_cols) >= 3:
        topk = st.slider("Top-k numeric columns by variance", 3, min(30, len(num_cols)), min(8, len(num_cols)))
        cols_var = df[num_cols].var().sort_values(ascending=False).head(topk).index.tolist()
        st.plotly_chart(px.imshow(df[cols_var].corr(), text_auto=True))
    elif viz == "Time-series (popularity by year)":
        if "release_year" in df.columns and "popularity" in df.columns:
            pop_year = df.groupby("release_year")["popularity"].mean().dropna()
            st.plotly_chart(px.line(x=pop_year.index, y=pop_year.values, markers=True))
        else:
            st.info("Need 'release_year' and 'popularity' columns")

def section_correlation():
    st.header("🔗 Correlation & Quick Stats")
    if df is None:
        st.info("No data.")
        return
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    if len(num_cols) >= 2:
        f1 = st.selectbox("Feature 1", num_cols, key="corr1")
        f2 = st.selectbox("Feature 2", num_cols, key="corr2")
        corr_val = df[[f1, f2]].corr().iloc[0, 1]
        st.write(f"Pearson r = {corr_val:.3f}")
        st.plotly_chart(px.imshow(df[[f1, f2]].corr(), text_auto=True))
    else:
        st.info("Need at least two numeric features")

def section_outliers():
    st.header("🚨 Outlier Detection & Removal")
    if df is None:
        st.info("No data.")
        return
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    if not num_cols:
        st.info("No numeric columns.")
        return
    col = st.selectbox("Select feature for outlier detection", num_cols, key="out_col")
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)]
    st.write(f"Detected {outliers.shape[0]} outliers (IQR method) in {col}")
    fig, ax = plt.subplots()
    sns.boxplot(x=df[col], ax=ax)
    st.pyplot(fig)
    z_thresh = st.slider("Z-score threshold (rows removed if ANY numeric feature abs(z) >= thresh)", 2.0, 5.0, 3.0, key="zthresh")
    if st.button("Preview outlier removal (do not apply)"):
        mask = (np.abs(zscore(df[num_cols].fillna(0))) < z_thresh).all(axis=1)
        st.write("Rows kept:", int(mask.sum()), "Rows removed:", int((~mask).sum()))
    if st.button("Apply outlier removal"):
        mask = (np.abs(zscore(df[num_cols].fillna(0))) < z_thresh).all(axis=1)
        st.session_state["df"] = df.loc[mask].reset_index(drop=True)
        st.success("Outliers removed and dataset updated.")
        st.experimental_rerun()

def section_feature_engineering():
    st.header("🧪 Feature Engineering & Processing")
    if st.session_state.get("df") is None:
        st.info("No data.")
        return
    df_local = st.session_state["df"].copy()

    st.subheader("Quick feature toggles")
    add_time = st.checkbox("Add time features (year, month, dayofweek)", value=True)
    add_textlen = st.checkbox("Add text-length features (track_name, artist)", value=True)
    add_freq = st.checkbox("Add frequency encoding for categorical cols (artist)", value=True)
    add_audio_agg = st.checkbox("Add audio aggregated features (mean, std)", value=True)

    if add_time and "release_date" in df_local.columns:
        df_local["release_month"] = df_local["release_date"].dt.month
        df_local["release_dayofweek"] = df_local["release_date"].dt.dayofweek

    if add_textlen:
        if "track_name" in df_local.columns:
            df_local["track_name_len"] = df_local["track_name"].astype(str).str.len()
            df_local["track_name_words"] = df_local["track_name"].astype(str).str.split().apply(len)
        if "artist" in df_local.columns:
            df_local["artist_len"] = df_local["artist"].astype(str).str.len()

    if add_freq and "artist" in df_local.columns:
        df_local["artist_freq"] = df_local["artist"].map(df_local["artist"].value_counts(normalize=True))

    audio_cols = [c for c in ["danceability", "energy", "valence", "tempo", "loudness"] if c in df_local.columns]
    if add_audio_agg and audio_cols:
        df_local["audio_mean"] = df_local[audio_cols].mean(axis=1)
        df_local["audio_std"] = df_local[audio_cols].std(axis=1)

    st.markdown("### Advanced transforms for modeling")
    numeric_for_model = st.multiselect("Select numeric features to scale / polynomialize", options=df_local.select_dtypes(include=["int64","float64"]).columns.tolist(), default=df_local.select_dtypes(include=["int64","float64"]).columns.tolist()[:8])
    do_scale = st.checkbox("Scale selected numeric features", value=True)
    do_poly = st.checkbox("Add polynomial features (degree 2) for selected numeric features (can be large)", value=False)

    df_fe = df_local.copy()
    scaler = None
    if do_scale and numeric_for_model:
        scaler = StandardScaler()
        df_fe[numeric_for_model] = scaler.fit_transform(df_fe[numeric_for_model].fillna(0))
        st.write("Scaling applied (preview):")
        st.dataframe(df_fe[numeric_for_model].head(3))

    if do_poly and numeric_for_model:
        pf = PolynomialFeatures(degree=2, include_bias=False)
        poly_arr = pf.fit_transform(df_fe[numeric_for_model].fillna(0))
        poly_cols = pf.get_feature_names_out(numeric_for_model)
        poly_df = pd.DataFrame(poly_arr, columns=poly_cols, index=df_fe.index)
        df_fe = pd.concat([df_fe, poly_df], axis=1)
        st.write(f"Added polynomial features: {len(poly_cols)} columns")

    st.session_state["df_fe"] = df_fe
    st.success("Feature engineering complete — engineered dataset saved to session_state['df_fe']")

    st.markdown("Preview of engineered dataset (first 5 rows)")
    st.dataframe(df_fe.head())

def _model_metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred))
    }

def section_modeling():
    st.header("🤖 Modeling — LinearRegression + RandomForest (+ optional XGBoost)")
    # prefer engineered df if exists, else cleaned df
    df_model = st.session_state.get("df_fe") if st.session_state.get("df_fe") is not None else st.session_state.get("df")
    if df_model is None:
        st.info("No data available for modeling. Run tabs 'Dataset Overview' and 'Feature Engineering' first.")
        return

    numeric_cols = df_model.select_dtypes(include=["int64","float64"]).columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns available for modeling.")
        return

    target = st.selectbox("Choose numeric target variable", options=numeric_cols, index=0)
    features_default = [c for c in numeric_cols if c != target][:8]
    features = st.multiselect("Choose feature columns", options=[c for c in numeric_cols if c != target], default=features_default)

    if not features:
        st.info("Choose at least one feature to train models.")
        return

    # Prepare data
    X = df_model[features].fillna(0)
    y = df_model[target].fillna(0)
    test_size = st.slider("Test set fraction", 0.05, 0.4, 0.2, step=0.05)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Model toggles
    run_lr = st.checkbox("Run Linear Regression (baseline)", value=True)
    run_rf = st.checkbox("Run Random Forest (tuned)", value=True)
    run_xgb = st.checkbox("Run XGBoost (optional)", value=False) if HAS_XGB else st.checkbox("XGBoost not installed", value=False, disabled=True)

    results = {}

    if run_lr:
        st.subheader("Linear Regression")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        metrics = _model_metrics(y_test, y_pred)
        cv_scores = cross_val_score(lr, X, y, cv=5, scoring="r2")
        results["LinearRegression"] = {"model": lr, "metrics": metrics, "cv": cv_scores}
        st.write("LR metrics:", metrics)
        st.write("LR CV R2 mean:", float(cv_scores.mean()))

    if run_rf:
        st.subheader("Random Forest with RandomizedSearchCV")
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        param_dist = {
            "n_estimators": [100, 200, 400],
            "max_depth": [None, 6, 12, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
        rsearch = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=3, scoring="r2", n_jobs=-1, random_state=42)
        with st.spinner("Tuning RandomForest (this may take a moment)..."):
            rsearch.fit(X_train, y_train)
        best_rf = rsearch.best_estimator_
        y_pred = best_rf.predict(X_test)
        metrics = _model_metrics(y_test, y_pred)
        cv_scores = cross_val_score(best_rf, X, y, cv=5, scoring="r2")
        results["RandomForest"] = {"model": best_rf, "metrics": metrics, "cv": cv_scores, "best_params": rsearch.best_params_}
        st.write("Best RF params:", rsearch.best_params_)
        st.write("RF metrics:", metrics)
        st.write("RF CV R2 mean:", float(cv_scores.mean()))

    if run_xgb and HAS_XGB:
        st.subheader("XGBoost (tuned)")
        xgb = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
        param_dist_xgb = {"n_estimators": [100, 200], "max_depth": [3,6,10], "learning_rate": [0.01,0.1,0.2]}
        rsearch_xgb = RandomizedSearchCV(xgb, param_distributions=param_dist_xgb, n_iter=6, cv=3, scoring="r2", n_jobs=-1, random_state=42)
        with st.spinner("Tuning XGBoost..."):
            rsearch_xgb.fit(X_train, y_train)
        best_xgb = rsearch_xgb.best_estimator_
        y_pred = best_xgb.predict(X_test)
        metrics = _model_metrics(y_test, y_pred)
        cv_scores = cross_val_score(best_xgb, X, y, cv=5, scoring="r2")
        results["XGBoost"] = {"model": best_xgb, "metrics": metrics, "cv": cv_scores, "best_params": rsearch_xgb.best_params_}
        st.write("XGB best params:", rsearch_xgb.best_params_)
        st.write("XGB metrics:", metrics)
        st.write("XGB CV R2 mean:", float(cv_scores.mean()))

    # Summarize comparisons
    if results:
        st.markdown("### Model comparison")
        comp_rows = []
        for name, res in results.items():
            comp_rows.append({"Model": name, "Test R2": res["metrics"]["r2"], "Test RMSE": res["metrics"]["rmse"], "Test MAE": res["metrics"]["mae"], "CV mean R2": float(np.mean(res["cv"]))})
        comp_df = pd.DataFrame(comp_rows).sort_values("Test R2", ascending=False)
        st.dataframe(comp_df.style.highlight_max(axis=0))

        # plot actual vs predicted for best model
        best_name = comp_df.iloc[0]["Model"]
        best_model = results[best_name]["model"]
        y_pred_best = best_model.predict(X_test)
        fig = px.scatter(x=y_test, y=y_pred_best, labels={"x": "Actual", "y": "Predicted"}, title=f"Actual vs Predicted — {best_name}")
        fig.add_shape(type="line", x0=min(y_test), x1=max(y_test), y0=min(y_test), y1=max(y_test), line=dict(color="red", dash="dash"))
        st.plotly_chart(fig, use_container_width=True)

        # If RF is best, show importances
        if best_name == "RandomForest":
            feat_imp = pd.Series(best_model.feature_importances_, index=features).sort_values(ascending=False)
            st.write("Top feature importances (RandomForest):")
            st.dataframe(feat_imp.head(15).to_frame("importance"))

        # Save best model to session for download/export
        st.session_state["best_model_obj"] = best_model
        st.session_state["best_model_name"] = best_name
        st.session_state["model_features"] = features

        # offer download of model pickle
        if st.button("📦 Download best model pickle"):
            buffer = io.BytesIO()
            pickle.dump({"model": best_model, "features": features, "model_name": best_name}, buffer)
            buffer.seek(0)
            st.download_button("Download model.pkl", data=buffer, file_name=f"best_model_{best_name}.pkl", mime="application/octet-stream")

def section_pca_clustering():
    st.header("🎨 PCA + KMeans Clustering")
    if st.session_state.get("df") is None:
        st.info("No data.")
        return
    pca_candidates = [c for c in ['valence','energy','danceability','tempo','loudness','duration_ms','popularity'] if c in st.session_state["df"].columns]
    if not pca_candidates:
        st.info("No expected audio columns found for PCA.")
        return
    sel = st.multiselect("Select features for PCA", options=pca_candidates, default=pca_candidates[:4])
    if len(sel) < 2:
        st.info("Select at least 2 features for PCA")
        return
    X = st.session_state["df"][sel].dropna()
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(Xs)
    pca_df = pd.DataFrame(pcs, columns=["PC1","PC2"])
    k = st.slider("Number of clusters (K)", 2, 8, 3)
    labels = KMeans(n_clusters=k, random_state=42).fit_predict(pcs)
    pca_df["cluster"] = labels.astype(str)
    st.plotly_chart(px.scatter(pca_df, x="PC1", y="PC2", color="cluster"), use_container_width=True)
    st.write("Explained variance ratio:", pca.explained_variance_ratio_.round(3))

def section_export():
    st.header("📥 Export: Cleaned, Engineered, Model, README, requirements (ZIP)")
    # build files in memory
    files = {}
    if st.session_state.get("df") is not None:
        csv_clean = st.session_state["df"].to_csv(index=False).encode("utf-8")
        files["spotify_cleaned.csv"] = csv_clean
        st.write("Cleaned CSV ready.")
    if st.session_state.get("df_fe") is not None:
        csv_fe = st.session_state["df_fe"].to_csv(index=False).encode("utf-8")
        files["spotify_engineered.csv"] = csv_fe
        st.write("Engineered CSV ready.")
    if st.session_state.get("best_model_obj") is not None:
        buf = io.BytesIO()
        pickle.dump({"model": st.session_state["best_model_obj"], "features": st.session_state.get("model_features"), "name": st.session_state.get("best_model_name")}, buf)
        buf.seek(0)
        files[f"best_model_{st.session_state.get('best_model_name','model')}.pkl"] = buf.read()
        st.write("Model pickle ready.")
    # README
    readme = textwrap.dedent(f"""
    # Spotify Final Project — CMSE 830

    Author: Rohan Rathi

    This archive contains:
    - spotify_cleaned.csv (cleaned dataset)
    - spotify_engineered.csv (feature-engineered dataset; optional)
    - best_model_*.pkl (pickled best model)
    - requirements.txt

    How to run the app:
    1. pip install -r requirements.txt
    2. streamlit run streamlit_final_with_glow.py
    """).strip().encode("utf-8")
    files["README.md"] = readme

    # requirements
    requirements = "\n".join([
        "streamlit",
        "pandas",
        "numpy",
        "scikit-learn",
        "plotly",
        "seaborn",
        "matplotlib",
        "scipy",
        "xgboost" if HAS_XGB else ""
    ]).strip().encode("utf-8")
    files["requirements.txt"] = requirements

    if not files:
        st.info("No artifacts available yet. Run earlier tabs to generate cleaned data, engineered data, and models.")
        return

    # create ZIP in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, data in files.items():
            zf.writestr(fname, data)
    zip_buffer.seek(0)
    st.download_button("📦 Download project ZIP (cleaned + engineered + model + README)", data=zip_buffer, file_name="spotify_final_project_artifacts.zip", mime="application/zip")
    st.success("ZIP ready — download and push to GitHub for submission.")

# -----------------------
# Router
# -----------------------
page = st.session_state["page"]
if page == "dataset_overview":
    section_dataset_overview()
elif page == "missing_imputation":
    section_missing_imputation()
elif page == "top_artists":
    section_top_artists()
elif page == "eda":
    section_eda()
elif page == "correlation":
    section_correlation()
elif page == "outliers":
    section_outliers()
elif page == "feature_engineering":
    section_feature_engineering()
elif page == "modeling":
    section_modeling()
elif page == "pca_clustering":
    section_pca_clustering()
elif page == "export":
    section_export()
else:
    st.info("Choose a top card to begin.")

# Footer: short checklist
st.markdown("---")
with st.expander("Rubric checklist (quick)"):
    st.write("""
    - Data Collection & Preparation: upload(s), remote CSV, synthetic available
    - Advanced cleaning & preprocessing: imputation, missing-column drop, date fixes
    - EDA & Visualization: histogram, box, violin, scatter+OLS, correlation heatmap, PCA scatter
    - Feature Engineering: text-length, freq-encode, audio aggregation, scaling, polynomial
    - Modeling: LinearRegression, RandomForest with RandomizedSearchCV tuning, CV and comparison
    - Streamlit App: top glowing navigation, caching, session_state, download buttons
    - Export: ZIP ready for GitHub submission (cleaned, engineered, model, README, requirements)
    """)
st.caption("If you'd like, I can now: (A) narrow models to a training pipeline file; (B) add unit tests; (C) produce a README mapping rubric bullets to specific code lines. Tell me which.")
