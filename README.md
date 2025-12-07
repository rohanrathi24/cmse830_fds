# Predicting Strokes: Insights from the Data

This project analyzes the Stroke Prediction Dataset and builds machine learning models to estimate the likelihood of an individual experiencing a stroke. The work includes exploratory data analysis, data preprocessing, model development, model evaluation, and a Streamlit-based web application for interactive predictions.

## Overview

The goal of this project is to understand patterns related to stroke occurrence and create an effective prediction system that uses demographic, medical, and lifestyle factors. The project combines data analysis and machine learning to support early identification of individuals at higher risk of stroke.

### Exploratory Data Analysis
  - Key observations from the EDA include:
  - Stroke occurrence increases notably with age.
  - Higher average glucose levels are associated with greater stroke risk.
  - Hypertension and heart disease show meaningful relationships with stroke likelihood.
  - BMI does not strongly differentiate stroke and non-stroke groups.
  - Gender and residence type have minimal effect on prediction.
  - Visualizations created include bar charts, violin plots, histograms, scatterplots, correlation matrices, and 3D scatterplots.

### Modeling Approach

Multiple models were trained and compared:
  - XGBoost (baseline and hyperparameter-tuned)
  - Random Forest
  - Decision Tree
  - Logistic Regression
  - Gaussian Naive Bayes
  - Support Vector Machine
  - Models were evaluated using accuracy, precision, recall, F1 score, ROC curves, PR curves, and confusion matrices.
  - The best-performing model was the hyperparameter-tuned XGBoost classifier.

### Streamlit Application
A web application was developed to allow users to:
  - Explore the dataset interactively
  - Visualize relationships between features
  - Compare machine learning models
  - Enter custom user inputs to obtain stroke predictions and probability scores
  - The application uses the tuned XGBoost model for predictions.

### Conclusion

The analysis shows that age, glucose level, hypertension, and heart disease are the most influential factors in predicting stroke. After testing several models, the tuned XGBoost classifier provided the strongest performance. The final Streamlit application integrates data analysis and predictive modeling into a single interactive system for stroke risk assessment.
