import streamlit as st
import pandas as pd
import seaborn as sns

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('data.csv')  # Adjust the file name if needed
    return df

df = load_data()




st.title('Breast Cancer Wisconsin (Diagnostic) Data Set Explorer')
df


# Remove 2nd and 33th columns
cancer = df.drop(df.columns[[1, 32]], axis=1)
# User input to select a column for plotting
selected_column = st.selectbox('Select a column for plotting', cancer.columns)

# Plot using Seaborn
st.subheader(f'KDE Plot for {selected_column}')
plot = sns.kdeplot(data = cancer, x = df[selected_column])
st.pyplot(fig=plot.get_figure())

# Add more plots 

attribute1="radius_mean"
attribute2="radius_worst"
plot = sns.jointplot(data=cancer, x=attribute1, y=attribute2, hue="diagnosis")
# Display the plot in Streamlit
st.pyplot(plot)

