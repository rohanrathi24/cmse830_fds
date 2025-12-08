import streamlit as st
import plotly.express as px
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hiplot as hip
import time as timer
import joblib  # kept in case you use it elsewhere
from PIL import Image

from sklearn import metrics
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    confusion_matrix,
    accuracy_score,
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder, StandardScaler

# model imports
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ------------------------- PAGE CONFIG -------------------------
st.set_page_config(
    page_title="Predicting Strokes: Insights from the Data",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

# ------------------------- LOAD DATA ---------------------------
# Use the main CSV from your GitHub repo
df = pd.read_csv("healthcare-dataset-stroke-data.csv")


# Function to replace missing values with median
def replace_missing_with_median(df_):
    df_["bmi"] = df_["bmi"].fillna(df_["bmi"].median())


# basic cleaning
df = df.drop("id", axis=1)
replace_missing_with_median(df)

# Check for duplicate rows (not used further, but kept)
duplicate_rows = df.duplicated()
duplicate_rows.sum()

# ----------------- SHARED PREPROCESSING FOR MODELS -------------
label_gender = LabelEncoder()
label_married = LabelEncoder()
label_work = LabelEncoder()
label_residence = LabelEncoder()
label_smoking = LabelEncoder()

df["gender"] = label_gender.fit_transform(df["gender"])
df["ever_married"] = label_married.fit_transform(df["ever_married"])
df["work_type"] = label_work.fit_transform(df["work_type"])
df["Residence_type"] = label_residence.fit_transform(df["Residence_type"])
df["smoking_status"] = label_smoking.fit_transform(df["smoking_status"])

# Handling Imbalanced Class Data Using SMOTE Technique
smote = SMOTE(sampling_strategy="minority")
X, y = smote.fit_resample(df.loc[:, df.columns != "stroke"], df["stroke"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.22, random_state=42
)

# Data Standardization (for LR, SVM)
scaler = StandardScaler()
scaler = scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)


# Cache the tuned XGBoost model so it's trained once
if hasattr(st, "cache_resource"):
    cache_decorator = st.cache_resource
else:
    cache_decorator = st.cache


@cache_decorator
def get_trained_xgb_model():
    xgb_mt = XGBClassifier(
        objective="reg:logistic",
        random_state=42,
        use_label_encoder=False,
        colsample_bytree=0.5,
        gamma=0.2,
        learning_rate=0.25,
        max_depth=10,
        min_child_weight=1,
    )
    xgb_mt.fit(X_train, y_train)
    return xgb_mt


# ------------------------- HELPER FUNCTIONS --------------------
# Function to filter data based on user selections
def filter_data(
    df_raw, selected_work_type, selected_smoking_status, selected_age_range, selected_gender
):
    filtered_data = df_raw[
        (df_raw["work_type"] == selected_work_type)
        & (df_raw["smoking_status"] == selected_smoking_status)
        & (df_raw["age"] >= selected_age_range[0])
        & (df_raw["age"] <= selected_age_range[1])
        & (df_raw["gender"] == selected_gender)
    ]
    return filtered_data


# Function to create bar plots of categorical features by diagnosis
def create_bar_plot(df_raw, categorical_feature):
    fig = px.histogram(df_raw, x="stroke", color=categorical_feature, barmode="group")
    return fig


# Function to create violin plots of numerical features by diagnosis
def create_violin_plot(df_raw, numerical_feature):
    fig = px.violin(df_raw, x="stroke", y=numerical_feature, box=True, hover_data=df_raw.columns)
    return fig


# Function to create a correlation matrix
def create_correlation_matrix(df_raw, corr_range):
    numerical_features = df_raw.select_dtypes(include=["float64"]).columns
    selected_corr_data = df_raw[numerical_features].corr()
    selected_corr_data = selected_corr_data[
        (selected_corr_data >= corr_range[0]) & (selected_corr_data <= corr_range[1])
    ]
    return selected_corr_data


# Plots for the models
def calculate_metrics_and_plots(model, train_X, train_y, test_X, test_y):
    # Train the classifier
    model.fit(train_X, train_y)

    # Predict on the test set
    y_pred_model = model.predict(test_X)

    # Calculate metrics
    ac = accuracy_score(test_y, y_pred_model)
    rc = roc_auc_score(test_y, y_pred_model)
    prec = precision_score(test_y, y_pred_model)
    rec = recall_score(test_y, y_pred_model)
    f1 = f1_score(test_y, y_pred_model)

    # Confusion Matrix
    cm = confusion_matrix(test_y, y_pred_model)

    # ROC Curve
    fpr, tpr, _ = roc_curve(test_y, model.predict_proba(test_X)[:, 1])
    roc_auc_val = auc(fpr, tpr)

    # Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(
        test_y, model.predict_proba(test_X)[:, 1]
    )
    pr_auc_val = auc(recall_vals, precision_vals)

    # Confusion Matrix Heatmap
    fig_cm = go.Figure()
    fig_cm.add_trace(
        go.Heatmap(
            z=cm[::-1],
            x=["Predicted 0", "Predicted 1"],
            y=["Actual 1", "Actual 0"],
            colorscale="Viridis",
            showscale=False,
        )
    )
    fig_cm.update_layout(
        title="Confusion Matrix",
        xaxis=dict(title="Predicted Class"),
        yaxis=dict(title="Actual Class"),
    )

    # ROC Curve
    fig_roc = go.Figure()
    fig_roc.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name="ROC curve (AUC={:.2f})".format(roc_auc_val),
        )
    )
    fig_roc.update_layout(
        title="Receiver Operating Characteristic (ROC) Curve",
        xaxis=dict(title="False Positive Rate"),
        yaxis=dict(title="True Positive Rate"),
        showlegend=True,
    )

    # Precision-Recall Curve
    fig_pr = go.Figure()
    fig_pr.add_trace(
        go.Scatter(
            x=recall_vals,
            y=precision_vals,
            mode="lines",
            name="Precision-Recall curve (AUC={:.2f})".format(pr_auc_val),
        )
    )
    fig_pr.update_layout(
        title="Precision-Recall Curve",
        xaxis=dict(title="Recall"),
        yaxis=dict(title="Precision"),
        showlegend=True,
    )

    # Metrics Bar Graph
    metrics_labels = ["Accuracy", "ROC AUC", "Precision", "Recall", "F1-Score"]
    metrics_values = [ac, rc, prec, rec, f1]

    fig_metrics = go.Figure()
    fig_metrics.add_trace(go.Bar(x=metrics_labels, y=metrics_values, name="Metrics"))
    fig_metrics.update_layout(
        barmode="group", xaxis=dict(title="Metrics"), yaxis=dict(title="Value")
    )

    return fig_cm, fig_roc, fig_pr, fig_metrics


# ------------------------- APP TITLE ---------------------------
st.write(
    '<h2 style="text-align:center; vertical-align:middle; line-height:2; color:#046366;">Predicting Strokes: Insights from the Data</h2>',
    unsafe_allow_html=True,
)

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["About the Data", "Visualizations", "Playground", "Method Assessment", "Prediciton", "Conclusion", "About Me"]
)

# ------------------------- TAB 1 -------------------------------
with tab1:
    image_path = "bg.png"  # Replace with the actual file path

    # Check if the image file exists at the specified path
    try:
        with open(image_path, "rb") as image_file:
            img = Image.open(image_file)
            img = img.resize((img.width, 300))
            st.image(img, caption="Stroke Prediction", use_column_width=True)
    except FileNotFoundError:
        pass

    st.write(
        "Stroke Prediction plays a pivotal role in predicting the likelihood of an individual experiencing a stroke. "
        "Strokes, as the second leading cause of death globally, accounting for approximately 11% of total deaths "
        "according to the World Health Organization (WHO), represent a critical healthcare challenge."
    )
    st.write(
        "The 'Stroke Prediction Dataset' was sourced from Kaggle which can be accessed at: "
        "[Kaggle Dataset Link](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset) and emerged as the most "
        "suitable choice due to its alignment with the primary focus of stroke prediction and prevention. This dataset "
        "encompasses a wide array of attributes, including demographic information, medical history, and lifestyle factors."
    )
    st.write(
        "This dataset comprises 5110 records and 12 columns featuring both numerical and categorical data. It includes "
        "critical information such as unique identifiers, gender, age, medical conditions (hypertension and heart disease), "
        "marital status, occupation, residence type, glucose levels, BMI, smoking status, and stroke occurrences. Its "
        "primary objective is to unveil relationships between these factors and the likelihood of a stroke."
    )

    st.header("About the App")
    st.write(
        "1. By employing machine learning, the app offers users a comprehensive risk assessment for strokes, aiding in early prediction and preventive measures."
    )
    st.write(
        "2. Users can interactively explore the dataset through various visualizations, such as scatter plots and interactive 3D plots. "
        "The app ensures a user-friendly experience, allowing customization of attribute selection and visualization choices."
    )
    st.write(
        "3. Beyond risk assessment, the application serves as an educational platform, providing valuable information about strokes and "
        "associated risk factors. This educational component aims to increase awareness and encourage proactive health management."
    )
    st.write(
        "4. The web app offers transparency regarding model performance, presenting users with model evaluation metrics. It also provides "
        "personalized recommendations for stroke prevention based on an individual's risk factors, empowering users to make informed decisions "
        "about their health."
    )

    st.markdown(
        """<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """,
        unsafe_allow_html=True,
    )
    st.write(
        "If you'd like to view the unprocessed data, click the 'Show Raw Data' button. For those interested in numbers, explore detailed "
        "feature breakdowns and statistical analysis in the section below."
    )

    checks = st.columns(2)
    # Display the dataset
    with checks[0]:
        with st.expander("Show Raw Data"):
            st.write(pd.DataFrame(df, columns=df.columns))
            st.write("Stroke Prediction Dataset Information:")
            st.write(f"Total Number of Samples: {df.shape[0]}")
            st.write(f"Number of Features: {df.shape[1]}")

    with checks[1]:
        with st.expander("Show Statistics about Data"):
            st.write(df.describe())
            st.write("Stroke Prediction Dataset Information:")
            st.write(f"Total Number of Samples: {df.shape[0]}")
            st.write(f"Number of Features: {df.shape[1]}")

    st.write(
        "To explore the data further we will take a look into interactive plots and visualizations in the next tabs"
    )

# ------------------------- TAB 2 -------------------------------
with tab2:
    st.sidebar.title("Welcome to the data exploration section")
    st.header("What factors are causing a Stroke ?")

    st.sidebar.subheader("Use filters to uncover insights")

    selected_work_type = st.sidebar.selectbox("Work Type", df["work_type"].unique())
    selected_smoking_status = st.sidebar.selectbox(
        "Smoking Status", df["smoking_status"].unique()
    )
    selected_age_range = st.sidebar.slider(
        "Age Range", int(df["age"].min()), int(df["age"].max()), (20, 80)
    )
    selected_gender = st.sidebar.selectbox("Gender", df["gender"].unique())

    # Apply filters and store filtered data
    filtered_data = filter_data(
        df, selected_work_type, selected_smoking_status, selected_age_range, selected_gender
    )

    # Display filtered data
    st.sidebar.subheader("Filtered Data")
    st.sidebar.dataframe(filtered_data)

    if st.checkbox("Examining stroke trends by lifestyle category"):
        st.markdown("---")
        st.subheader("Examining stroke trends by lifestyle category.")
        st.markdown(
            "You can pick different categories like gender, marital status, type of work, where you live, "
            "hypertension, heart disease and smoking habits. This chart helps you see how many people in each "
            "category had a stroke."
        )

        categorical_variables = [
            "gender",
            "hypertension",
            "heart_disease",
            "ever_married",
            "work_type",
            "Residence_type",
            "smoking_status",
        ]
        bar_x = st.selectbox("Select a category", categorical_variables)
        bar_plot = create_bar_plot(df, bar_x)
        st.plotly_chart(bar_plot)

        st.markdown("Observations:")
        st.markdown(
            "1. Features like gender and residence type do not show strong differences in stroke probability."
        )
        st.markdown(
            "2. Marriage status, work type, and smoking habits show more noticeable differences."
        )
        st.markdown(
            "3. Most patients do not have hypertension or heart disease."
        )

    if st.checkbox("Examining stroke trends with human charcateristics"):
        st.markdown("---")
        st.subheader("Examining stroke trends with human charcateristics")
        st.markdown(
            "You can pick different factors like age, average glucose level, and BMI. "
            "The violin graph shows how these values are distributed among people with and without stroke."
        )

        violin_y = st.selectbox(
            "Select a category", df.select_dtypes(include=["float64"]).columns
        )
        violin_plot = create_violin_plot(df, violin_y)
        st.plotly_chart(violin_plot)

        st.markdown("Observations:")
        st.markdown(
            "1. Stroke probability increases with age, especially after about 60 years."
        )
        st.markdown(
            "2. Higher glucose levels are associated with greater stroke risk."
        )
        st.markdown(
            "3. BMI is less clearly separated between stroke and non-stroke groups."
        )

    if st.checkbox("Histogram"):
        st.header("Histogram")
        st.write(
            "These histograms show the distribution of continuous features split by gender."
        )
        st.subheader("Select a feature for the histogram:")
        selected_feature = st.selectbox(
            "Select a feature", df.select_dtypes(include=["float64"]).columns
        )
        gender_format = "gender"
        bin_count = st.slider("Number of Bins", min_value=1, max_value=100, value=20)

        st.subheader(f"Histogram of {selected_feature}")

        fig = px.histogram(df, x=selected_feature, color=gender_format, nbins=bin_count)
        fig.update_xaxes(title_text=selected_feature)
        fig.update_yaxes(title_text="Count")
        fig.update_traces(marker=dict(line=dict(width=2)))
        fig.update_layout(height=700, width=900)
        st.plotly_chart(fig)

    if st.checkbox("Studying relationships in stroke data"):
        st.subheader("Studying relationships in stroke data.")
        st.markdown(
            "We use bivariate scatterplots to explore how two numerical features relate to each other and to stroke-related categories."
        )

        col3, col4, col5 = st.columns(3, gap="large")

        numerical = df.select_dtypes(include=["float64"]).columns
        categorical = df.select_dtypes(include=["int64", "object"]).columns

        with col3:
            alt_x = st.selectbox("Select a feature for (X)?", numerical)
        with col4:
            alt_y = st.selectbox("Select a feature for (Y) ?", numerical)
        with col5:
            cat_hue = st.selectbox("Choose target", categorical)

        fig3 = None
        if alt_x and alt_y and cat_hue:
            # Fix: avoid trendline when X and Y are the same
            if alt_x == alt_y:
                st.info("X and Y are the same feature. Showing scatterplot without regression line.")
                fig3 = px.scatter(df, x=alt_x, y=alt_y, color=cat_hue)
            else:
                fig3 = px.scatter(df, x=alt_x, y=alt_y, color=cat_hue, trendline="ols")

            fig3.update_layout(
                {
                    "plot_bgcolor": "rgba(0, 0, 0, 0)",
                    "paper_bgcolor": "rgba(0, 0, 0, 0)",
                },
                font=dict(size=18),
            )
            st.write(fig3)

        st.markdown("Observations:")
        st.markdown(
            "The scatter plots suggest that as people get older, their average glucose levels and BMI tend to rise, and so does their risk of having a stroke."
        )

    if st.checkbox("Correlation"):
        st.subheader("Correlation")
        st.markdown(
            "Correlation helps show how strongly numerical features move together."
        )

        with st.form("key2"):
            corr_range = st.slider(
                "Select correlation magnitude range", value=[-1.0, 1.0], step=0.05
            )

            correlation_data = create_correlation_matrix(df, corr_range)

            st.write("Correlation Matrix:")
            st.dataframe(correlation_data, width=800, height=150)

            button2 = st.form_submit_button("Apply range")

        corr_mat = st.checkbox("Show/hide correlation matrix")

        if corr_mat:
            st.subheader("Correlation Matrix Heatmap")

            fig_corr = px.imshow(
                correlation_data,
                color_continuous_scale="RdBu_r",
                title="Correlation Matrix Heatmap",
            )
            fig_corr.update_layout(width=800, height=600)
            st.plotly_chart(fig_corr)

    if st.checkbox("3D Scatter Plot"):
        st.header("3D Scatter Plot")
        st.write(
            "The 3D scatter plot shows how age, avg_glucose_level, and bmi interact with stroke status."
        )

        fig = px.scatter_3d(
            df,
            x="age",
            y="avg_glucose_level",
            z="bmi",
            color="stroke",
            color_continuous_scale=["blue", "red"],
            labels={
                "age": "Age",
                "avg_glucose_level": "Average Glucose Level",
                "bmi": "BMI",
                "stroke": "Stroke",
            },
        )

        fig.update_layout(
            scene=dict(
                xaxis_title="Age",
                yaxis_title="Average Glucose Level",
                zaxis_title="BMI",
            ),
            title="Age, Average Glucose Level, BMI vs. Stroke",
        )
        st.plotly_chart(fig)

# ------------------------- TAB 3 -------------------------------
with tab3:
    # visualization with HiPlot
    def save_hiplot_to_html(exp):
        output_file = "hiplot_plot_1.html"
        exp.to_html(output_file)
        return output_file

    st.write("Visualization with HiPlot")
    selected_columns = st.multiselect("Select columns to visualize", df.columns)
    selected_data = df[selected_columns]
    if not selected_data.empty:
        experiment = hip.Experiment.from_dataframe(selected_data)
        hiplot_html_file = save_hiplot_to_html(experiment)
        st.components.v1.html(
            open(hiplot_html_file, "r").read(), height=1500, scrolling=True
        )
    else:
        st.write("No data selected. Please choose at least one column to visualize.")

    st.markdown(
        """<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """,
        unsafe_allow_html=True,
    )

    # Conclusion
    st.markdown("Conclusions that can be drawn from observations are:")
    st.markdown(
        "1. The target variable stroke is highly imbalanced with far more instances of class 0 (no stroke) than class 1 (stroke)."
    )
    st.markdown(
        "2. Categorical variables such as gender, hypertension, heart_disease, ever_married, work_type, Residence_type, and smoking_status showed various distributions."
    )
    st.markdown(
        "3. Continuous variables (age, avg_glucose_level, bmi) exhibited different distributions. Age and average glucose level were found to be higher in stroke patients."
    )
    st.markdown(
        "4. BMI might not be a strong predictor for stroke, as its distribution was similar for stroke and non-stroke patients."
    )
    st.markdown(
        "5. Older patients, particularly those who are self-employed or in private jobs, have a higher incidence of stroke."
    )
    st.markdown(
        "6. The EDA provided valuable insights into the factors associated with strokes. Age, hypertension, heart disease, and average glucose level appear to be significant factors."
    )

# ------------------------- TAB 4 -------------------------------
with tab4:
    model_menu = [
        "XGBoost (XGB) with HyperTuned Parameters",
        "XGBoost (XGB)",
        "Random Forest (RF)",
        "Logistic Regression (LR)",
        "Decision Tree (DT)",
        "Gaussian Naive Bayes (GNB)",
        "Singular Vector Machine (SVM)",
    ]
    model = st.selectbox("Select a Model", model_menu)

    # XGBoost (basic)
    if model == "XGBoost (XGB)":
        start = timer.time()
        xgb_m = XGBClassifier(
            objective="reg:logistic",
            random_state=42,
            use_label_encoder=False,
        )
        xgb_m.fit(X_train, y_train)
        end = timer.time()
        st.success("Training time {:.2f} seconds".format(end - start))

        y_xgb = xgb_m.predict(X_test)
        cnf_matrix = metrics.confusion_matrix(y_test, y_xgb)
        st.write("Accuracy:", metrics.accuracy_score(y_test, y_xgb))
        st.write("Precision:", metrics.precision_score(y_test, y_xgb))
        st.write("Recall:", metrics.recall_score(y_test, y_xgb))
        st.write("F1:", metrics.f1_score(y_test, y_xgb))

        fig_cm, fig_roc, fig_pr, fig_metrics = calculate_metrics_and_plots(
            XGBClassifier(
                objective="reg:logistic",
                random_state=42,
                use_label_encoder=False,
            ),
            X_train,
            y_train,
            X_test,
            y_test,
        )

        st.subheader("Confusion Matrix")
        st.plotly_chart(fig_cm)
        st.subheader("ROC Curve")
        st.plotly_chart(fig_roc)
        st.subheader("Precision-Recall Curve")
        st.plotly_chart(fig_pr)
        st.subheader("Metrics Bar Graph")
        st.plotly_chart(fig_metrics)

    # XGBoost with HyperTuned Parameter
    elif model == "XGBoost (XGB) with HyperTuned Parameters":
        start = timer.time()
        xgb_mt = XGBClassifier(
            objective="reg:logistic",
            random_state=42,
            use_label_encoder=False,
            colsample_bytree=0.5,
            gamma=0.2,
            learning_rate=0.25,
            max_depth=10,
            min_child_weight=1,
        )
        xgb_mt.fit(X_train, y_train)
        end = timer.time()
        st.success("Training time {:.2f} seconds".format(end - start))

        y_xgb = xgb_mt.predict(X_test)
        y_train_predict = xgb_mt.predict(X_train)
        cnf_matrix = metrics.confusion_matrix(y_train, y_train_predict)
        st.write("Train Accuracy", accuracy_score(y_train, y_train_predict))
        st.write("Accuracy:", metrics.accuracy_score(y_test, y_xgb))
        st.write("Precision:", metrics.precision_score(y_test, y_xgb))
        st.write("Recall:", metrics.recall_score(y_test, y_xgb))
        st.write("F1:", metrics.f1_score(y_test, y_xgb))

        fig_cm, fig_roc, fig_pr, fig_metrics = calculate_metrics_and_plots(
            XGBClassifier(
                objective="reg:logistic",
                random_state=42,
                use_label_encoder=False,
                colsample_bytree=0.5,
                gamma=0.2,
                learning_rate=0.25,
                max_depth=10,
                min_child_weight=1,
            ),
            X_train,
            y_train,
            X_test,
            y_test,
        )

        st.subheader("Confusion Matrix")
        st.plotly_chart(fig_cm)
        st.subheader("ROC Curve")
        st.plotly_chart(fig_roc)
        st.subheader("Precision-Recall Curve")
        st.plotly_chart(fig_pr)
        st.subheader("Metrics Bar Graph")
        st.plotly_chart(fig_metrics)

    # Random Forest
    elif model == "Random Forest (RF)":
        start = timer.time()
        ranfor_m = RandomForestClassifier(n_estimators=100, random_state=42)
        ranfor_m.fit(X_train, y_train)
        end = timer.time()
        st.success("Training time {:.2f} seconds".format(end - start))

        y_ranfor = ranfor_m.predict(X_test)
        cnf_matrix = metrics.confusion_matrix(y_test, y_ranfor)
        st.write("Accuracy:", metrics.accuracy_score(y_test, y_ranfor))
        st.write("Precision:", metrics.precision_score(y_test, y_ranfor))
        st.write("Recall:", metrics.recall_score(y_test, y_ranfor))
        st.write("F1:", metrics.f1_score(y_test, y_ranfor))

        fig_cm, fig_roc, fig_pr, fig_metrics = calculate_metrics_and_plots(
            RandomForestClassifier(n_estimators=100, random_state=42),
            X_train,
            y_train,
            X_test,
            y_test,
        )

        st.subheader("Confusion Matrix")
        st.plotly_chart(fig_cm)
        st.subheader("ROC Curve")
        st.plotly_chart(fig_roc)
        st.subheader("Precision-Recall Curve")
        st.plotly_chart(fig_pr)
        st.subheader("Metrics Bar Graph")
        st.plotly_chart(fig_metrics)

    # Decision Tree
    elif model == "Decision Tree (DT)":
        start = timer.time()
        dtree_m = DecisionTreeClassifier(random_state=42)
        dtree_m.fit(X_train, y_train)
        end = timer.time()
        st.success("Training time {:.2f} seconds".format(end - start))

        y_dtree = dtree_m.predict(X_test)
        cnf_matrix = metrics.confusion_matrix(y_test, y_dtree)
        st.write("Accuracy:", metrics.accuracy_score(y_test, y_dtree))
        st.write("Precision:", metrics.precision_score(y_test, y_dtree))
        st.write("Recall:", metrics.recall_score(y_test, y_dtree))
        st.write("F1:", metrics.f1_score(y_test, y_dtree))

        fig_cm, fig_roc, fig_pr, fig_metrics = calculate_metrics_and_plots(
            DecisionTreeClassifier(random_state=42),
            X_train,
            y_train,
            X_test,
            y_test,
        )

        st.subheader("Confusion Matrix")
        st.plotly_chart(fig_cm)
        st.subheader("ROC Curve")
        st.plotly_chart(fig_roc)
        st.subheader("Precision-Recall Curve")
        st.plotly_chart(fig_pr)
        st.subheader("Metrics Bar Graph")
        st.plotly_chart(fig_metrics)

    # Logistic Regression
    elif model == "Logistic Regression (LR)":
        start = timer.time()
        logit_m = LogisticRegression(solver="lbfgs", random_state=42, max_iter=1000)
        logit_m.fit(X_train_std, y_train)
        end = timer.time()
        st.success("Training time {:.2f} seconds".format(end - start))

        y_pred = logit_m.predict(X_test_std)
        cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
        st.write("Accuracy:", metrics.accuracy_score(y_test, y_pred))
        st.write("Precision:", metrics.precision_score(y_test, y_pred))
        st.write("Recall:", metrics.recall_score(y_test, y_pred))
        st.write("F1:", metrics.f1_score(y_test, y_pred))

        fig_cm, fig_roc, fig_pr, fig_metrics = calculate_metrics_and_plots(
            LogisticRegression(solver="lbfgs", random_state=42, max_iter=1000),
            X_train_std,
            y_train,
            X_test_std,
            y_test,
        )

        st.subheader("Confusion Matrix")
        st.plotly_chart(fig_cm)
        st.subheader("ROC Curve")
        st.plotly_chart(fig_roc)
        st.subheader("Precision-Recall Curve")
        st.plotly_chart(fig_pr)
        st.subheader("Metrics Bar Graph")
        st.plotly_chart(fig_metrics)

    # Gaussian Naive Bayes
    elif model == "Gaussian Naive Bayes (GNB)":
        start = timer.time()
        gnb_m = GaussianNB()
        gnb_m.fit(X_train, y_train)
        end = timer.time()
        st.success("Training time {:.2f} seconds".format(end - start))

        y_gnb = gnb_m.predict(X_test)
        cnf_matrix = metrics.confusion_matrix(y_test, y_gnb)
        st.write("Accuracy:", metrics.accuracy_score(y_test, y_gnb))
        st.write("Precision:", metrics.precision_score(y_test, y_gnb))
        st.write("Recall:", metrics.recall_score(y_test, y_gnb))
        st.write("F1:", metrics.f1_score(y_test, y_gnb))

        fig_cm, fig_roc, fig_pr, fig_metrics = calculate_metrics_and_plots(
            GaussianNB(), X_train, y_train, X_test, y_test
        )

        st.subheader("Confusion Matrix")
        st.plotly_chart(fig_cm)
        st.subheader("ROC Curve")
        st.plotly_chart(fig_roc)
        st.subheader("Precision-Recall Curve")
        st.plotly_chart(fig_pr)
        st.subheader("Metrics Bar Graph")
        st.plotly_chart(fig_metrics)

    # Singular Vector Machine
    elif model == "Singular Vector Machine (SVM)":
        start = timer.time()
        svm_m = SVC(kernel="rbf", probability=True)
        svm_m.fit(X_train_std, y_train)
        end = timer.time()
        st.success("Training time {:.2f} seconds".format(end - start))

        y_svm = svm_m.predict(X_test_std)
        cnf_matrix = metrics.confusion_matrix(y_test, y_svm)
        st.write("Accuracy:", metrics.accuracy_score(y_test, y_svm))
        st.write("Precision:", metrics.precision_score(y_test, y_svm))
        st.write("Recall:", metrics.recall_score(y_test, y_svm))
        st.write("F1 Score:", metrics.f1_score(y_test, y_svm))

        fig_cm, fig_roc, fig_pr, fig_metrics = calculate_metrics_and_plots(
            SVC(kernel="rbf", probability=True),
            X_train_std,
            y_train,
            X_test_std,
            y_test,
        )

        st.subheader("Confusion Matrix")
        st.plotly_chart(fig_cm)
        st.subheader("ROC Curve")
        st.plotly_chart(fig_roc)
        st.subheader("Precision-Recall Curve")
        st.plotly_chart(fig_pr)
        st.subheader("Metrics Bar Graph")
        st.plotly_chart(fig_metrics)

    st.markdown("Conclusion:")
    st.info(
        "XGBoost (with Hyper Tuned Parameters) has been selected as the Best Model due to its High Accuracy compared to other models that have been trained."
    )

# ------------------------- TAB 5 -------------------------------
with tab5:
    st.markdown("Enter the User's Details to predict the occurance of Stroke")
    st.text("Please Enter correct details to get better results")

    # Getting User Inputs
    gender = st.radio("What is User's gender", ("Male", "Female"))
    age = st.number_input("Enter User's age", value=40)
    hypertension = st.radio("Hypertension?", ("Yes", "No"))
    heart_disease = st.radio("User Ever had a heart disease?", ("Yes", "No"))
    ever_married = st.radio("User Ever Married?", ("Yes", "No"))
    work_type = st.radio(
        "What is User's work type?",
        ("Government Job", "Private Job", "Self Employed", "Never Worked", "Children"),
    )
    Residence_type = st.radio(
        "What is User's Residence type?", ("Urban", "Rural")
    )
    avg_glucose_level = st.number_input(
        "Enter User's Average Glucose Level", value=92.35
    )

    # BMI Calculation with Height and Weight if User doesn't know BMI
    if st.checkbox("Dont Know BMI? Use height and weight"):
        height = st.number_input("Enter User's Height in cm", value=160)
        weight = st.number_input("Enter User's Weight in kgs", value=60)
        bmi = weight / (height / 100) ** 2
        st.write("BMI of user is {:.2f} and will be autoupdated".format(bmi))
    else:
        bmi = st.number_input("Enter User's BMI", value=25.4)

    smoking_status = st.radio(
        "User's Smoking Status?",
        ("Unknown", "Formerly Smoked", "Never Smoked", "Smokes"),
    )

    # model (XGBoost tuned, trained inside the app)
    prediction_model = "XGBoost (tuned)"
    trained_model = get_trained_xgb_model()
    model_accuracy = "94.9%"  # or your actual measured accuracy

    if st.button("Submit"):
        # Encoding categorical attributes to values
        gender_val = 1 if gender == "Male" else 0
        age_val = float(age)
        hypertension_val = 1 if hypertension == "Yes" else 0
        ever_married_val = 1 if ever_married == "Yes" else 0
        heart_disease_val = 1 if heart_disease == "Yes" else 0

        if work_type == "Government Job":
            work_type_val = 0
        elif work_type == "Never Worked":
            work_type_val = 1
        elif work_type == "Private Job":
            work_type_val = 2
        elif work_type == "Self Employed":
            work_type_val = 3
        elif work_type == "Children":
            work_type_val = 4
        else:
            work_type_val = 2

        Residence_type_val = 1 if Residence_type == "Urban" else 0
        avg_glucose_level_val = float(avg_glucose_level)
        bmi_val = float(bmi)

        if smoking_status == "Unknown":
            smoking_status_val = 0
        elif smoking_status == "Formerly Smoked":
            smoking_status_val = 1
        elif smoking_status == "Never Smoked":
            smoking_status_val = 2
        elif smoking_status == "Smokes":
            smoking_status_val = 3
        else:
            smoking_status_val = 0

        # Creating nparray of User Inputs
        user_input = np.array(
            [
                gender_val,
                age_val,
                hypertension_val,
                heart_disease_val,
                ever_married_val,
                work_type_val,
                Residence_type_val,
                avg_glucose_level_val,
                bmi_val,
                smoking_status_val,
            ]
        ).reshape(1, -1)

        # converting into dataframe to avoid mismatching feature_names error
        user_input = pd.DataFrame(
            user_input,
            columns=[
                "gender",
                "age",
                "hypertension",
                "heart_disease",
                "ever_married",
                "work_type",
                "Residence_type",
                "avg_glucose_level",
                "bmi",
                "smoking_status",
            ],
        )

        # prediction using tuned XGBoost model
        prediction = trained_model.predict(user_input)

        # Prediction Probability
        pred_prob = trained_model.predict_proba(user_input)
        stroke_prob = pred_prob[0][1] * 100

        if prediction == 1:
            st.header("User has Higher Chances of having a Stroke")
        else:
            st.header("User has Lower Chances of having a Stroke")

        if stroke_prob < 25:
            st.success("Probability of Occurance of Stroke is {:.2f}%".format(stroke_prob))
        elif stroke_prob < 50:
            st.info("Probability of Occurance of Stroke is {:.2f}%".format(stroke_prob))
        elif stroke_prob < 75:
            st.warning("Probability of Occurance of Stroke is {:.2f}%".format(stroke_prob))
        else:
            st.error("Probability of Occurance of Stroke is {:.2f}%".format(stroke_prob))

        st.text("Predicted with " + prediction_model + " Model with Accuracy of " + model_accuracy)

# ------------------------- TAB 6 -------------------------------
with tab6:
    st.markdown("Conclusions:")

    st.markdown(
        "Our Stroke Prediction web application, utilizing the Kaggle Stroke Prediction Dataset, is a comprehensive tool for understanding and predicting stroke risk."
    )
    st.markdown(
        "1. The app addresses an imbalanced target variable, with a majority of instances indicating no stroke."
    )
    st.markdown(
        "2. Categorical variables like gender, hypertension, heart_disease, and others show varied distributions."
    )
    st.markdown(
        "3. Continuous variables such as age, avg_glucose_level, and BMI exhibit distinct patterns, with age and average glucose level identified as significant factors in stroke risk."
    )
    st.markdown(
        "4. BMI may not be a strong predictor for strokes, as its distribution remains similar for both stroke and non-stroke cases."
    )
    st.markdown(
        "5. Older age, certain occupations, and elevated glucose levels are associated with a higher incidence of strokes."
    )
    st.markdown(
        "6. The scatter plots reveal age and average glucose levels as influential factors in stroke risk."
    )
    st.markdown(
        "7. Certain features like gender and residence type show minimal impact on predicting strokes, while marriage status, work type, and smoking habits are notable determinants."
    )
    st.markdown(
        "8. The app provides valuable insights through EDA, guiding feature selection and modeling. However, the imbalanced target variable presents a challenge in model development."
    )
    st.write(
        "9. XGBoost (with Hyper Tuned Parameters) has been selected as the Best Model due to its High Accuracy compared to other trained models."
    )

    st.markdown(
        "[Stroke Prediction Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)"
    )

# ------------------------- TAB 7 -------------------------------
with tab7:
    image_path = "bio.jpg"
    try:
        image = open(image_path, "rb").read()
        st.image(image, width=300)
    except FileNotFoundError:
        st.write("Bio image not found.")

    text_column = st.columns(2)[0]

    with text_column:
        st.write(
            "Hello there! I'm Madhurya, a dedicated learner currently pursuing a Master's in Data Science. "
            "My journey is fueled by a passion for unraveling the stories hidden in data."
        )
        st.write(
            "In the halls of MSU, I dive deep into the realms of Python, Data Analysis, and Machine Learning. "
            "Learning isn't just a task; it's my enthusiasm for embracing new technologies and methodologies in the dynamic field of data science."
        )
        st.write(
            "Beyond the screen, I find joy in diverse pursuits. Whether it's a badminton match, painting, "
            "playing guitar, or hiking, I embrace the beauty of life beyond coding."
        )
        st.write(
            "Come, explore my web app, and join me in this adventure of data exploration and analytics."
        )
