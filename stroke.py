import streamlit as st
import plotly.express as px
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hiplot as hip
import time as timer
import joblib

from sklearn import metrics
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix, accuracy_score,make_scorer
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.metrics import auc,roc_auc_score,roc_curve,precision_score,recall_score,f1_score


#model imports
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance

#charts
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Keep this to avoid unwanted warning on the wen app
st.set_option('deprecation.showPyplotGlobalUse', False)

# Loading dataset
df = pd.read_csv('https://raw.githubusercontent.com/madhuryashankar/CMSE/main/healthcare-dataset-stroke-data.csv')


# Function to replace missing values with median
def replace_missing_with_median(df):
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())

df = df.drop('id',axis=1)

# Check for duplicate rows
duplicate_rows = df.duplicated()    

# Count the number of duplicate rows
duplicate_rows.sum()
    
# Function to filter data based on user selections
def filter_data(df, selected_work_type, selected_smoking_status, selected_age_range, selected_gender):
    filtered_data = df[(df['work_type'] == selected_work_type) &
                       (df['smoking_status'] == selected_smoking_status) &
                       (df['age'] >= selected_age_range[0]) & (df['age'] <= selected_age_range[1]) &
                       (df['gender'] == selected_gender)]
    return filtered_data

# Function to create bar plots of categorical features by diagnosis
def create_bar_plot(df, categorical_feature):
    fig = px.histogram(df, x="stroke", color=categorical_feature, barmode='group')
    return fig

# Function to create violin plots of numerical features by diagnosis
def create_violin_plot(df, numerical_feature):
    fig = px.violin(df, x="stroke", y=numerical_feature, box=True, hover_data=df.columns)
    return fig

# Function to create scatter plots with correlation analysis
def create_scatterplot_with_correlation(df, x_feature, y_feature, hue_feature):
    fig = px.scatter(df, x=x_feature, y=y_feature, color=hue_feature, trendline="ols")
    return fig

# Function to create a correlation matrix
def create_correlation_matrix(df, corr_range):
    numerical_features = df.select_dtypes(include=['float64']).columns
    selected_corr_data = df[numerical_features].corr()
    selected_corr_data = selected_corr_data[(selected_corr_data >= corr_range[0]) & (selected_corr_data <= corr_range[1])]
    return selected_corr_data

# Plots for the models
def calculate_metrics_and_plots(model,train_X, train_y, test_X, test_y):
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
    roc_auc = auc(fpr, tpr)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(test_y, model.predict_proba(test_X)[:, 1])
    pr_auc = auc(recall, precision)

    # Create Plots
    # Confusion Matrix Heatmap
    fig_cm = go.Figure()
    fig_cm.add_trace(go.Heatmap(z=cm[::-1], x=['Predicted 0', 'Predicted 1'], y=['Actual 1', 'Actual 0'],
                                colorscale='Viridis', showscale=False))
    fig_cm.update_layout(title='Confusion Matrix', xaxis=dict(title='Predicted Class'), yaxis=dict(title='Actual Class'))

    # ROC Curve
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve (AUC={:.2f})'.format(roc_auc)))
    fig_roc.update_layout(title='Receiver Operating Characteristic (ROC) Curve',
                          xaxis=dict(title='False Positive Rate'),
                          yaxis=dict(title='True Positive Rate'),
                          showlegend=True)

    # Precision-Recall Curve
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='Precision-Recall curve (AUC={:.2f})'.format(pr_auc)))
    fig_pr.update_layout(title='Precision-Recall Curve',
                         xaxis=dict(title='Recall'),
                         yaxis=dict(title='Precision'),
                         showlegend=True)

    # Metrics Bar Graph
    metrics_labels = ['Accuracy', 'ROC AUC', 'Precision', 'Recall', 'F1-Score']
    metrics_values = [ac, rc, prec, rec, f1]

    fig_metrics = go.Figure()
    fig_metrics.add_trace(go.Bar(x=metrics_labels, y=metrics_values, name='Metrics'))
    fig_metrics.update_layout(barmode='group', xaxis=dict(title='Metrics'), yaxis=dict(title='Value'))

    return fig_cm, fig_roc, fig_pr, fig_metrics

# Apply styling
st.set_page_config(
    page_title="Predicting Strokes: Insights from the Data",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

# Set the title and description of the app
st.write('<h2 style="text-align:center; vertical-align:middle; line-height:2; color:#046366;">Predicting Strokes: Insights from the Data</h2>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7= st.tabs(["About the Data", "Visualizations", "Playground", "Method Assessment", "Prediciton", "Conclusion", "About Me"])

with tab1 :
    image_path = "bg.png"  # Replace with the actual file path

    # Check if the image file exists at the specified path
    try:
        with open(image_path, "rb") as image_file:
            img = Image.open(image_file)
            img = img.resize((img.width, 300))
            st.image(img, caption="Stroke Prediction", use_column_width=True)
    except FileNotFoundError:
        pass
  
    st.write("Stroke Prediction plays a pivotal role in predicting the likelihood of an individual experiencing a stroke. Strokes, as the second leading cause of death globally, accounting for approximately 11% of total deaths according to the World Health Organization (WHO), represent a critical healthcare challenge.")
    st.write("The 'Stroke Prediction Dataset' was sourced from Kaggle which can be accessed at: [Kaggle Dataset Link](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset) and emerged as the most suitable choice due to its alignment with the primary focus of stroke prediction and prevention. This dataset encompasses a wide array of attributes, including demographic information, medical history, and lifestyle factors.")
    st.write("This dataset comprises 5110 records and 12 columns featuring both numerical and categorical data. It includes critical information such as unique identifiers, gender, age, medical conditions (hypertension and heart disease), marital status, occupation, residence type, glucose levels, BMI, smoking status, and stroke occurrences. Its primary objective is to unveil relationships between these factors and the likelihood of a stroke.")
    
    st.header("About the App")
    st.write("1. By employing machine learning, the app offers users a comprehensive risk assessment for strokes, aiding in early prediction and preventive measures.")

    st.write("2. Users can interactively explore the dataset through various visualizations, such as scatter plots and interactive 3D plots. The app ensures a user-friendly experience, allowing customization of attribute selection and visualization choices.")

    st.write("3. Beyond risk assessment, the application serves as an educational platform, providing valuable information about strokes and associated risk factors. This educational component aims to increase awareness and encourage proactive health management.")

    st.write("4. The web app offers transparency regarding model performance, presenting users with model evaluation metrics. It also provides personalized recommendations for stroke prevention based on an individual's risk factors, empowering users to make informed decisions about their health.")

    st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    st.write("If you'd like to view the unprocessed data, click the 'Show Raw Data' button. For those interested in numbers, explore detailed feature breakdowns and statistical analysis in the section below.")
    
    checks = st.columns(2)
    # Display the dataset
    with checks[0]:
        with st.expander("Show Raw Data"):
            # if st.checkbox('Show Raw Data'):
            st.write(pd.DataFrame(df, columns=df.columns))
            st.write('Stroke Prediction Dataset Information:')
            st.write(f'Total Number of Samples: {df.shape[0]}')
            st.write(f'Number of Features: {df.shape[1]}')

    with checks[1]:
        with st.expander("Show Statistics about Data"):
            st.write(df.describe())
            st.write('Stroke Prediction Dataset Information:')
            st.write(f'Total Number of Samples: {df.shape[0]}')
            st.write(f'Number of Features: {df.shape[1]}')

    st.write('To explore the data further we will take a look into interactive plots and visualizations in the next tabs')
        


with tab2 :
    st.sidebar.title('Welcome to the data exploration section')
    st.header("What factors are causing a Stroke ?")

    replace_missing_with_median(df)

    # Sidebar inputs
    st.sidebar.subheader('Use filters to uncover insights')
    selected_work_type = st.sidebar.selectbox('Work Type', df['work_type'].unique())
    selected_smoking_status = st.sidebar.selectbox('Smoking Status', df['smoking_status'].unique())
    selected_age_range = st.sidebar.slider('Age Range', int(df['age'].min()), int(df['age'].max()), (20, 80))
    selected_gender = st.sidebar.selectbox('Gender', df['gender'].unique())

    # Apply filters and store filtered data
    filtered_data = filter_data(df, selected_work_type, selected_smoking_status, selected_age_range, selected_gender)

    # Display filtered data
    st.sidebar.subheader('Filtered Data')
    st.sidebar.dataframe(filtered_data)

    if st.checkbox('Examining stroke trends by lifestyle category'):
        # Partition and description for bar graph
        st.markdown("---")
        st.subheader('Examining stroke trends by lifestyle category.')
        st.markdown("You can pick different categories like gender, marital status, type of work, where you live, hypertension, heart disease and smoking habits. It's a bit like a colorful bar chart you might see in a magazine. This chart helps you see how many people in each category had a stroke. By looking at this chart, you can figure out if these things, like being married or where you live, can make a stroke more likely to happen. It's like a detective tool to study strokes and learn what might cause them.")

        # Create bar plot
        categorical_variables = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        bar_x = st.selectbox('Select a category', categorical_variables)
        bar_plot = create_bar_plot(df, bar_x)
        st.plotly_chart(bar_plot)

        # Description for bar graph
        if bar_x == 'smoking_status':
            st.markdown("**Smoking Status:** By looking at the bar graph for the 'smoking status' category, we can learn about the relationship between smoking and strokes. The graph shows different groups, like people who smoke, used to smoke, or never smoked. The height of each bar represents how many people in each group had a stroke. So, if you see a really tall bar for the 'smokers' group, it means a lot of smokers had strokes. If the 'never smoked' bar is short, it suggests that fewer non-smokers had strokes.")
        elif bar_x == 'work_type':
            st.markdown("**Work Type:** From the bar graph of the 'Work Type' category, we can see how different types of jobs are related to strokes. The graph uses different colors to show the number of people in each job category who had a stroke. By looking at this graph, we can understand if certain jobs have a higher or lower risk of strokes. ")
        elif bar_x == 'gender':
            st.markdown("**Gender:**  In this graph, you'll see two bars, one for 'Male' and one for 'Female'. The height of these bars shows how many men and women in the dataset had a stroke. If one bar is taller than the other, it means more people of that gender had a stroke. This helps us understand if gender has any connection to strokes. For example, if the 'Female' bar is taller, it might mean that women in this dataset had more strokes. ")
        elif bar_x == 'Residence_type':
            st.markdown("**Residence Type:**  From the bar graph of the 'Residence Type' category, you can see two colorful bars representing two different types of residence: 'Urban' and 'Rural.' Each bar shows the number of people who had a stroke in these two types of areas. This helps us understand whether living in an urban or rural place might be connected to having a stroke. It's like looking at the numbers to see if the place where someone lives is related to their risk of having a stroke. This information can be valuable for understanding and preventing strokes.")
        elif bar_x == 'ever_married':
            st.markdown("**Ever Married:**   The graph will show two bars, one for people who have been married and another for those who haven't. You'll see how many people in each group had a stroke. This helps us understand if being married or not being married might affect the chances of having a stroke. It's like looking at data to find clues about how different life experiences might impact our health.")
        elif bar_x == 'hypertension':
            st.markdown("**Hypertension:** In this bar graph, we analyze the relationship between hypertension and strokes. The bars represent different groups, such as individuals with and without hypertension. The height of each bar indicates the number of people in each group who had a stroke. If you observe a tall bar for the 'hypertension' group, it means a significant number of people with hypertension had strokes. Conversely, if the 'no hypertension' bar is shorter, it suggests that fewer individuals without hypertension had strokes. This information can help us understand the impact of hypertension on stroke risk.")
        elif bar_x == 'heart_disease':
            st.markdown("**Heart Disease:** This bar graph illustrates the connection between heart disease and strokes. The bars represent different categories, including individuals with and without heart disease. The height of each bar represents the number of people in each category who had a stroke. If you see a tall bar for the 'heart disease' group, it indicates that a significant number of individuals with heart disease had strokes. Conversely, a shorter bar for the 'no heart disease' category suggests that fewer individuals without heart disease had strokes. This analysis allows us to explore the relationship between heart disease and the risk of strokes.")

        st.markdown("Observations:")
        st.markdown("1. When we look at features like gender and residence type, they don't seem to make a big difference in predicting whether a person will have a stroke or not. The chances of having a stroke for different groups within these features are quite similar to the overall dataset.")
        st.markdown("2. As for features like marriage status, work type, and smoking habits, the chances of having a stroke for specific groups are noticeably different from the overall dataset. This indicates that these features are also important in determining whether a person is at risk of having a stroke.")
        st.markdown("3. Regarding hypertension and heart disease, The majority of patients do not have hypertension or heart disease.")

    if st.checkbox('Examining stroke trends with human charcateristics'): 
        # Partition and description for violin graph
        st.markdown("---")
        st.subheader('Examining stroke trends with human charcateristics')
        st.markdown("You can pick different factors like age, average glucose level, and BMI. The violin graph here shows a picture of how these numbers are spread out among people who had a stroke and people who didn't. It helps us compare and see the differences in these factors between the two groups. It's like a visual tool to understand how these numbers affect the chances of having a stroke.")

        # Create violin plot
        violin_y = st.selectbox('Select a category', df.select_dtypes(include=['float64']).columns)
        violin_plot = create_violin_plot(df, violin_y)
        st.plotly_chart(violin_plot)

        # Description for violin graph
        if violin_y == 'bmi':
            st.markdown("**BMI (Body Mass Index):** By examining this graph, we can draw some valuable conclusions. If you notice that there's a significant difference in BMI between people who had a stroke and those who didn't, it could suggest that BMI plays a role in stroke risk. In simpler terms, it helps us see if being underweight, overweight, or having a healthy BMI might affect the likelihood of experiencing a stroke. So, if you see a big difference in the heights of the bars in the graph, it might indicate that BMI is an important factor when it comes to stroke prediction.")
        elif violin_y == 'avg_glucose_level':
            st.markdown("**Average Glucose Level:** The graph displays how average glucose levels are distributed among individuals in the dataset who either had a stroke or did not. You can see two violins side by side, one for those who had a stroke and one for those who didn't. By looking at the shape and position of these violins, you can make inferences. For example, if the 'stroke' violin is noticeably wider at higher glucose levels, it might suggest that elevated glucose levels could be associated with a higher risk of stroke.")
        elif violin_y == 'age':
            st.markdown("**Age:** The width of the plot shows us how ages are distributed among people who either had a stroke or did not. If the plot is wider at a certain age range, it means there are more people in that group. For example, if the plot is wider in the middle, it means more people in that age group had a stroke. By looking at the plot, we can see if there's a particular age group where strokes are more common. This helps us understand how age is related to the risk of having a stroke.")

        st.markdown("Observations:")
        st.markdown("1. For younger patients, especially those aged 0-30, the chances of having a stroke are very low (less than 0.01), but there's a significant increase in the likelihood for patients aged 60 and older.")
        st.markdown("2. Patients with BMI in the range of 20-50 have stroke probabilities similar to the overall dataset. However, there's a noticeable drop in stroke probability for patients with BMIs outside this range. Please note that some groups may have small sample sizes.")
        st.markdown("3. Patients with lower average glucose levels (below 170) have stroke probabilities similar to the overall dataset. In contrast, patients with higher average glucose levels have a significantly higher stroke probability.")

    if st.checkbox('Histogram'):
        st.header('Histogram')
        st.write(" Imagine the bars as groups, each with a different color. These bars show how often something happened, like a stroke, and how it's related to something else, like a age. The taller the bars, the more it happened, and you can point your mouse at them to see the exact numbers. When the bars overlap, like they're close together, it means these things are connected. If one group's bars are mostly on one side and another group's bars are on the other side, it means they're different in some way. It's like they're sharing a secret!")
        col1,col2=st.columns(2,gap='small')
        st.subheader('Select a feature for the histogram:')
        selected_feature = st.selectbox('Select a feature', df.select_dtypes(include=['float64']).columns)
        gender_format = 'gender'
        bin_count = st.slider('Number of Bins', min_value=1, max_value=100, value=20)

        st.subheader(f'Histogram of {selected_feature}')

        # Create an interactive histogram using Plotly Express
        fig = px.histogram(df, x=selected_feature, color=gender_format, nbins=bin_count)
        fig.update_xaxes(title_text=selected_feature)
        fig.update_yaxes(title_text='Count')
        fig.update_traces(marker=dict(line=dict(width=2)))
        fig.update_layout(height=700, width=900)
        # Display the interactive plot
        st.plotly_chart(fig)

        st.markdown("Observations:")
        st.markdown("The patterns for average glucose levels and BMI look a bit like a bell curve, with a slight tilt to the right. This means there are a few higher values on the right side. Similar behavior is seen for age, where there's also a bit of spread. There isn't much distinction between genders. However, we have very few samples from other gender groups, so we won't consider them for now.")

    if st.checkbox('Studying relationships in stroke data'):  
        st.subheader("Studying relationships in stroke data.")
        st.markdown("Let's explore a new concept that can help us understand the factors related to strokes. We're going to look at something called a 'Bivariate scatterplot by diagnosis (Stroke).' This is like a special picture that shows two things at once and helps us see if there's a connection between those two things and whether someone had a stroke.")
        st.markdown("In simpler terms, it's like using a visual chart to see how two different factors might be linked to strokes. For instance, we can use this chart to investigate whether age and having high blood pressure are connected to having a stroke. The chart uses dots to show this relationship, making it much easier for us to see and understand the connections. So, let's dive into this visual tool and uncover valuable insights about strokes and the factors involved.")
        
        l1, m1, r1 = st.columns((2,5,1))

        col3, col4, col5 = st.columns(3,gap='large')

        numerical = df.select_dtypes(include=['float64']).columns;
        categorical = df.select_dtypes(include=['object']).columns;
        with col3:
            alt_x = st.selectbox("Select a feature for (X)?", numerical)
        with col4:
            alt_y = st.selectbox("Select a feature for (Y) ?", numerical)
        with col5:
            cat_hue = st.selectbox("Choose target", categorical)

        if alt_x and alt_y and cat_hue:
            fig3 = px.scatter(df, alt_x, alt_y, color=cat_hue, trendline="ols")
            fig3.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            },font=dict(
                size=18
                )   
            )

        st.write(fig3)

        st.markdown('When you study these scatter plots, you can make some interesting discoveries:')
        st.markdown("1. Lines: When you notice a line going up or down, it's like seeing a connection between the two factors you chose. If it's going up, it means when one thing increases, the other often increases too. If it's heading down, when one thing goes up, the other tends to go down.")
        st.markdown("2. Dots: Think of each dot as representing one person's information. If most of these dots are close to the line, it tells us that these two factors are closely linked. On the other hand, if the dots are spread out all over the place, it suggests that they aren't strongly connected.")
        st.markdown("So, these plots act like visual puzzles. If you spot a clear pattern in the dots, it helps you understand how these factors affect each other. It's like uncovering clues in a colorful puzzle!")
    
        st.markdown("Observations:")
        st.markdown("The scatter plots suggest that as people get older, their average glucose levels and body mass index (BMI) tend to rise, and so does their risk of having a stroke. Additionally, it appears that age and average glucose levels have a more significant impact on the risk of a stroke.")

    if st.checkbox('Correlation'): 

        l3, m3, r3 = st.columns((4, 5, 1))
        st.subheader("Correlation")

        st.markdown("Correlation means looking at how two things are connected or related to each other. For example, imagine we want to understand if age and the risk of having a stroke are connected. If we find that, as people get older, the chances of having a stroke also increase, we say there's a positive correlation between age and stroke risk. On the other hand, if we see that as the number of fruits people eat increases, the risk of a stroke decreases, that would be a negative correlation.")

        st.markdown("Correlation helps us figure out if one thing might be influencing another, like age influencing stroke risk, or if there's no connection between them at all.")    

        with st.form("key2"):
            corr_range = st.slider("Select correlation magnitude range", value=[-1.0, 1.0], step=0.05)

            correlation_data = create_correlation_matrix(df, corr_range)

            st.write("Correlation Matrix:")
            st.dataframe(correlation_data, width=800, height=150)

            button2 = st.form_submit_button("Apply range")

        corr_mat = st.checkbox('Show/hide correlation matrix')

        if corr_mat:
            st.subheader("Correlation Matrix Heatmap")
            st.markdown("The correlation matrix heatmap provides an overview of the relationships between numerical features. Strong correlations are shown in warmer (reddish) or cooler (bluish) colors.")

            fig_corr = px.imshow(correlation_data,
                                color_continuous_scale="RdBu_r",
                                title="Correlation Matrix Heatmap")
            fig_corr.update_layout(width=800, height=600)
            st.plotly_chart(fig_corr)

            st.markdown("Observations:")
            st.markdown("The colorful heatmap reveals that there are only faint links between average glucose levels and age, as well as average glucose levels and BMI (Body Mass Index). However, there's a somewhat stronger connection between BMI and age.")    

    if st.checkbox('3D Scatter Plot'):
        st.header('3D Scatter Plot')
        st.write(" The 3D scatter plot provides a three-dimensional view of how age, avg_glucose_level, and bmi interact with each other with respect to the stroke status. The points are colored based on whether a patient had a stroke (red) or not (blue).")
       
        # Create a color map for the 'stroke' variable
        colors = df['stroke'].map({0: 'blue', 1: 'red'})

        fig = px.scatter_3d(df, x='age', y='avg_glucose_level', z='bmi', color='stroke',color_continuous_scale=["blue", "red"],labels={'age': 'Age', 'avg_glucose_level': 'Average Glucose Level', 'bmi': 'BMI', 'stroke': 'Stroke'})

        fig.update_layout(scene=dict(xaxis_title='Age', yaxis_title='Average Glucose Level', zaxis_title='BMI'),title='Age, Average Glucose Level, BMI vs. Stroke')
        # Display the interactive 3D scatter plot
        st.plotly_chart(fig)

        st.markdown("Observations:")
        st.markdown("1. Stroke patients (red points) generally tend to be older and have higher glucose levels, which is consistent with our earlier findings.")
        st.markdown("2. BMI does not appear to differentiate stroke patients from non-stroke patients as there is significant overlap in the BMI values of both groups.")


with tab3 :
    #visualization with HiPlot
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
        st.components.v1.html(open(hiplot_html_file, 'r').read(), height=1500, scrolling=True)
    else:
        st.write("No data selected. Please choose at least one column to visualize.")

    st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    
    #Conclusion
    st.markdown("Conclusions that can be drawn from observations are:")
    st.markdown("1. The target variable stroke is highly imbalanced with far more instances of class 0 (no stroke) than class 1 (stroke). This is an important observation as it will affect the choice of machine learning model and evaluation metric.")
    st.markdown("2. Categorical variables such as gender, hypertension, heart_disease, ever_married, work_type, Residence_type, and smoking_status showed various distributions. Notably, hypertension and heart disease were found more frequently in patients who had a stroke.")
    st.markdown("3. Continuous variables (age, avg_glucose_level, bmi) exhibited different distributions. Age and average glucose level were found to be higher in stroke patients, but no significant difference in BMI was observed between stroke and non-stroke patients.")
    st.markdown("4. The analysis also indicated that bmi might not be a strong predictor for stroke, as the distribution of BMI was similar for stroke and non-stroke patients.")
    st.markdown("5. It was observed that older patients, particularly those who are self-employed or in private jobs, have a higher incidence of stroke. Also, stroke patients generally have higher glucose levels regardless of their work type and gender.")
    st.markdown("6. The EDA provided valuable insights into the factors associated with strokes. Age, hypertension, heart disease, and average glucose level appear to be significant factors, while BMI might not be a significant predictor. This information can guide the feature selection and modeling process. However, the imbalance in the target variable could present a challenge in building a predictive model.")

with tab4 :
        label_gender = LabelEncoder()
        label_married = LabelEncoder()
        label_work = LabelEncoder()
        label_residence = LabelEncoder()
        label_smoking = LabelEncoder()
        df['gender'] = label_gender.fit_transform(df['gender'])
        df['ever_married'] = label_married.fit_transform(df['ever_married'])
        df['work_type']= label_work.fit_transform(df['work_type'])
        df['Residence_type']= label_residence.fit_transform(df['Residence_type'])
        df['smoking_status']= label_smoking.fit_transform(df['smoking_status'])
         #Handling Imbalanced Class Data Using SMOTE Technique
        smote = SMOTE(sampling_strategy='minority')
        X, y= smote.fit_resample(df.loc[:,df.columns!='stroke'], df['stroke'])
        #Building Data Model and Training
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.22,random_state=42)

        #Data Standardization
        scaler = StandardScaler()
        scaler = scaler.fit(X_train)
        X_train_std = scaler.transform(X_train)
        X_test_std = scaler.transform(X_test)

        #ML Model Training and Evaluation
        model_menu = ["XGBoost (XGB) with HyperTuned Parameters","XGBoost (XGB)","Random Forest (RF)","Logistic Regression (LR)","Decision Tree (DT)","Gaussian Naive Bayes (GNB)","Singular Vector Machine (SVM)"]
        model = st.selectbox("Select a Model",model_menu)
        #XGBoost
        if model == "XGBoost (XGB)":
            start = timer.time()
            xgb_m = XGBClassifier(objective="reg:logistic", random_state=42,use_label_encoder = False)
            xgb_m.fit(X_train, y_train)
            end = timer.time()
            st.success("Training time {:.2f} seconds".format(end-start))
            # Predicting test data
            y_xgb = xgb_m.predict(X_test)
            cnf_matrix = metrics.confusion_matrix(y_test, y_xgb)
            st.write("Accuracy:",metrics.accuracy_score(y_test, y_xgb))
            st.write("Precision:",metrics.precision_score(y_test, y_xgb))
            st.write("Recall:",metrics.recall_score(y_test, y_xgb))
            st.write("F1:",metrics.f1_score(y_test, y_xgb))


            # Calculate metrics and create plots
            fig_cm, fig_roc, fig_pr, fig_metrics = calculate_metrics_and_plots(XGBClassifier(), X_train, y_train, X_test, y_test)

            # Display Plots
            st.subheader("Confusion Matrix")
            st.plotly_chart(fig_cm)

            st.subheader("ROC Curve")
            st.plotly_chart(fig_roc)

            st.subheader("Precision-Recall Curve")
            st.plotly_chart(fig_pr)

            st.subheader("Metrics Bar Graph")
            st.plotly_chart(fig_metrics)


        #XGBoost with HyperTuned Parameter
        elif model == "XGBoost (XGB) with HyperTuned Parameters":
            start = timer.time()
            xgb_mt = XGBClassifier(objective="reg:logistic", random_state=42,
                                use_label_encoder = False, colsample_bytree= 0.5, 
                                gamma= 0.2, learning_rate= 0.25,
                                max_depth= 10, min_child_weight= 1,)
            xgb_mt.fit(X_train, y_train)
            end = timer.time()
            st.success("Training time {:.2f} seconds".format(end-start))
            # Predicting test data
            y_xgb = xgb_mt.predict(X_test)
            y_train_predict = xgb_mt.predict(X_train)
            cnf_matrix = metrics.confusion_matrix(y_train , y_train_predict)
            st.write('Train Accuracy',accuracy_score(y_train , y_train_predict))
            st.write("Accuracy:",metrics.accuracy_score(y_test, y_xgb))
            st.write("Precision:",metrics.precision_score(y_test, y_xgb))
            st.write("Recall:",metrics.recall_score(y_test, y_xgb))
            st.write("F1:",metrics.f1_score(y_test, y_xgb))

            # Calculate metrics and create plots
            fig_cm, fig_roc, fig_pr, fig_metrics = calculate_metrics_and_plots(XGBClassifier(), X_train, y_train, X_test, y_test)

            # Display Plots
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
            st.success("Training time {:.2f} seconds".format(end-start))
            # Predicting test data
            y_ranfor = ranfor_m.predict(X_test)
            cnf_matrix = metrics.confusion_matrix(y_test, y_ranfor)
            st.write("Accuracy:",metrics.accuracy_score(y_test, y_ranfor))
            st.write("Precision:",metrics.precision_score(y_test, y_ranfor))
            st.write("Recall:",metrics.recall_score(y_test, y_ranfor))
            st.write("F1:",metrics.f1_score(y_test, y_ranfor))

            # Calculate metrics and create plots
            fig_cm, fig_roc, fig_pr, fig_metrics = calculate_metrics_and_plots(RandomForestClassifier(), X_train, y_train, X_test, y_test)

            # Display Plots
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
            st.success("Training time {:.2f} seconds".format(end-start))
            # Predicting test data
            y_dtree = dtree_m.predict(X_test)
            cnf_matrix = metrics.confusion_matrix(y_test, y_dtree)
            st.write("Accuracy:",metrics.accuracy_score(y_test, y_dtree))
            st.write("Precision:",metrics.precision_score(y_test, y_dtree))
            st.write("Recall:",metrics.recall_score(y_test, y_dtree))
            st.write("F1:",metrics.f1_score(y_test, y_dtree))
            # Calculate metrics and create plots
            fig_cm, fig_roc, fig_pr, fig_metrics = calculate_metrics_and_plots(DecisionTreeClassifier(), X_train, y_train, X_test, y_test)

            # Display Plots
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
            logit_m = LogisticRegression(solver='lbfgs', random_state=42)
            logit_m.fit(X_train_std,y_train)
            end = timer.time()
            st.success("Training time {:.2f} seconds".format(end-start))
            # Predicting test data
            y_pred = logit_m.predict(X_test_std)
            cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
            st.write("Accuracy:",metrics.accuracy_score(y_test, y_pred))
            st.write("Precision:",metrics.precision_score(y_test, y_pred))
            st.write("Recall:",metrics.recall_score(y_test, y_pred))
            st.write("F1:",metrics.f1_score(y_test, y_pred))
            # Calculate metrics and create plots
            fig_cm, fig_roc, fig_pr, fig_metrics = calculate_metrics_and_plots(LogisticRegression(), X_train, y_train, X_test, y_test)

            # Display Plots
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
            st.success("Training time {:.2f} seconds".format(end-start))
            # Predicting test data
            y_gnb = gnb_m.predict(X_test)
            cnf_matrix = metrics.confusion_matrix(y_test, y_gnb)
            st.write("Accuracy:",metrics.accuracy_score(y_test, y_gnb))
            st.write("Precision:",metrics.precision_score(y_test, y_gnb))
            st.write("Recall:",metrics.recall_score(y_test, y_gnb))
            st.write("F1:",metrics.f1_score(y_test, y_gnb))
            # Calculate metrics and create plots
            fig_cm, fig_roc, fig_pr, fig_metrics = calculate_metrics_and_plots(GaussianNB(), X_train, y_train, X_test, y_test)

            # Display Plots
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
            svm_m = SVC(kernel='rbf',probability=True)
            svm_m.fit(X_train_std, y_train)
            end = timer.time()
            st.success("Training time {:.2f} seconds".format(end-start))
            # Predicting test data
            y_svm = svm_m.predict(X_test_std)
            cnf_matrix = metrics.confusion_matrix(y_test, y_svm)
            st.write("Accuracy:",metrics.accuracy_score(y_test, y_svm))
            st.write("Precision:",metrics.precision_score(y_test, y_svm))
            st.write("Recall:",metrics.recall_score(y_test, y_svm))
            st.write("F1 Score:",metrics.f1_score(y_test, y_svm))
            # Calculate metrics and create plots
            fig_cm, fig_roc, fig_pr, fig_metrics = calculate_metrics_and_plots(SVC(kernel='linear'), X_train, y_train, X_test, y_test)

            # Display Plots
            st.subheader("Confusion Matrix")
            st.plotly_chart(fig_cm)

            st.subheader("ROC Curve")
            st.plotly_chart(fig_roc)

            st.subheader("Precision-Recall Curve")
            st.plotly_chart(fig_pr)

            st.subheader("Metrics Bar Graph")
            st.plotly_chart(fig_metrics)
        
        st.markdown("Conclusion:")
        st.info("XGBoost (with Hyper Tuned Parameters) has been selected as the Best Model due to its High Accuracy compared to other models that has been Trained")

with tab5 :

        st.markdown("Enter the User's Details to predict the occurance of Stroke")
        st.text("Please Enter correct details to get better results")
        
        #Getting User Inputs
        gender = st.radio("What is User's gender",("Male","Female"))
        age = st.number_input("Enter User's age",value=40)
        hypertension = st.radio("Hypertension?",("Yes","No"))
        heart_disease = st.radio("User Ever had a heart disease?",("Yes","No"))
        ever_married = st.radio("User Ever Married?",("Yes","No"))
        work_type = st.radio("What is User's work type?",("Government Job","Private Job","Self Employed","Never Worked","Children"))
        Residence_type = st.radio("What is User's Residence type?",("Urban","Rural"))
        avg_glucose_level = st.number_input("Enter User's Average Glucose Level",value=92.35)
        
        #BMI Calculation with Height and Weight is User doesn't know BMI
        if st.checkbox("Dont Know BMI? Use height and weight"):
            height = st.number_input("Enter User's Height in cm",value=160)
            weight = st.number_input("Enter User's Weight in kgs",value=60)
            bmi = weight / (height/100)**2
            st.write("BMI of user is {:.2f} and will be autoupdated".format(bmi))
        else:
            bmi = st.number_input("Enter User's BMI",value=25.4)

        smoking_status = st.radio("User's Smoking Status?",("Unknown","Formerly Smoked","Never Smoked","Smokes"))
        
        #model (XGBoost)
        prediction_model = 'XGBoost'
        trained_model = joblib.load('XGBoostTunedModel.pkl')
        model_accuracy = "94.9%"

        if st.button("Submit"):
            #Encoding categorical attributes to values
            gender = 1 if gender == 'Male' else 0
            age = float(age)
            hypertension = 1 if hypertension == 'Yes' else 0
            ever_married = 1 if ever_married == 'Yes' else 0
            heart_disease = 1 if heart_disease == 'Yes' else 0
            if work_type == 'Government Job':
                work_type = 0 
            elif work_type == 'Never Worked':
                work_type = 1 
            elif work_type == 'Private Job':
                work_type = 2 
            elif work_type == 'Self Employed':
                work_type = 3
            elif work_type == 'Children':
                work_type = 4 
            Residence_type = 1 if Residence_type == 'Urban' else 0
            avg_glucose_level = float(avg_glucose_level)
            bmi = float(bmi)
            if smoking_status == 'Unknown':
                smoking_status = 0 
            elif smoking_status == 'Formerly Smoked':
                smoking_status = 1 
            elif smoking_status == 'Never Smoked':
                smoking_status = 2
            elif smoking_status == 'Smokes':
                smoking_status = 3 

            #Creating nparray of User Inputs
            user_input = np.array([gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status]).reshape(1,-1)
            
            #converting into dataframe to avoid mismatching feature_names error
            user_input = pd.DataFrame(user_input, columns = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'])
            
            #prediction using selected model
            prediction = trained_model.predict(user_input)

            #Prediction Probability
            pred_prob = trained_model.predict_proba(user_input)
            stroke_prob = pred_prob[0][1]*100

            #Printing Predicted results
            if prediction == 1:
                st.header("User has Higher Chances of having a Stroke")
            else:
                st.header("User has Lower Chances of having a Stroke")
            
            #printing prediction probability 
            if stroke_prob < 25:
                st.success("Probability of Occurance of Stroke is {:.2f}%".format(stroke_prob))
            elif stroke_prob < 50:
                st.info("Probability of Occurance of Stroke is {:.2f}%".format(stroke_prob))
            elif stroke_prob < 75:
                st.warning("Probability of Occurance of Stroke is {:.2f}%".format(stroke_prob))
            else:
                st.error("Probability of Occurance of Stroke is {:.2f}%".format(stroke_prob))
            st.text("Predicted with "+prediction_model+" Model with Accuracy of " +model_accuracy)

with tab6:
    #Conclusion
    st.markdown("Conclusions:")

    st.markdown("Our Stroke Prediction web application, utilizing the Kaggle Stroke Prediction Dataset, is a comprehensive tool for understanding and predicting stroke risk.")

    st.markdown("1. The app addresses an imbalanced target variable, with a majority of instances indicating no stroke. This imbalance influences model choice and evaluation metrics.")
    st.markdown("2. Categorical variables like gender, hypertension, heart_disease, and others show varied distributions, emphasizing the importance of certain conditions in stroke occurrence.")
    st.markdown("3. Continuous variables such as age, avg_glucose_level, and BMI exhibit distinct patterns, with age and average glucose level identified as significant factors in stroke risk.")
    st.markdown("4. BMI may not be a strong predictor for strokes, as its distribution remains similar for both stroke and non-stroke cases.")
    st.markdown("5. Older age, certain occupations, and elevated glucose levels are associated with a higher incidence of strokes.")
    st.markdown("6. The scatter plots reveal age and average glucose levels as influential factors in stroke risk.")
    st.markdown("7. Certain features like gender and residence type show minimal impact on predicting strokes, while marriage status, work type, and smoking habits are notable determinants.")
    st.markdown("8. The app provides valuable insights through EDA, guiding feature selection and modeling. However, the imbalanced target variable presents a challenge in model development.")
    st.write("9. XGBoost (with Hyper Tuned Parameters) has been selected as the Best Model due to its High Accuracy compared to other trained models.")

    st.markdown("[Stroke Prediction Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)")

with tab7 :

    image_path = "bio.jpg" 
    image = open(image_path, "rb").read()
    st.image(image, width=300)
    text_column = st.columns(2)[0]

    with text_column:
        st.write("Hello there! I'm Madhurya, a dedicated learner currently pursuing a Master's in Data Science. My journey is fueled by a passion for unraveling the stories hidden in data.")
        st.write("In the halls of MSU, I dive deep into the realms of Python, Data Analysis, and Machine Learning. Learning isn't just a task; it's my enthusiasm for embracing new technologies and methodologies in the dynamic field of data science.") 
        st.write("Beyond the screen, I find joy in diverse pursuits. Whether it's a fierce badminton match, the strokes of a paintbrush, the soothing chords of a guitar, or the tranquility of a hiking trail, I embrace the beauty of life beyond coding.")
        st.write("Come, explore my web app, and join me in this exciting adventure of data exploration and analytics. Let's make technology not just a skill but a thrilling journey!")
