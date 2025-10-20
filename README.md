========================================
🎵 Spotify Data Exploration Dashboard
========================================

Overview:
---------
This is an interactive Streamlit dashboard for exploring a Spotify dataset. 
It provides visualizations, summary statistics, and feature distributions. 
The dashboard includes a sidebar filter to select songs and updates charts dynamically.

Features:
---------
1. Dataset preview (first 5 rows).
2. Dataset info: number of rows, columns, numerical/categorical columns, missing values.
3. Top 10 Artists and Top 10 Songs by count.
4. Correlation heatmap for numerical features.
5. Top 10 Artists by average popularity.
6. Average popularity over years.
7. Boxplots for numerical features (detect outliers).
8. Histograms for key features ('valence', 'energy', 'tempo', 'loudness', 'danceability').
9. Collapsible sections using expanders for a clean layout.
10. Sidebar filter for selecting specific songs.

Requirements:
-------------
- Python 3.9+
- See 'requirements.txt' for necessary Python packages.

Setup Instructions:
-------------------
1. Clone or download the project folder.
2. Ensure your dataset is saved at:
   C:\Users\anand\Downloads\data.csv
3. Install dependencies:
   - (Optional) Create a virtual environment:
     python -m venv venv
   - Activate the environment:
     Windows: venv\Scripts\activate
     Mac/Linux: source venv/bin/activate
   - Install packages:
     pip install -r requirements.txt
4. Run the Streamlit app:
   streamlit run spotify_dashboard_expander.py
5. The dashboard will open automatically in your default web browser.

Usage:
------
- Use the sidebar to select a specific song.
- Expand/collapse sections to view the visualizations and statistics.
- Scroll down to explore all charts.

Notes:
------
- The app automatically handles missing values for visualization.
- The 'release_date' column is converted to datetime for year-based analysis.
- The dashboard uses Seaborn and Matplotlib for all visualizations.

Author:
-------
Rohan Rathi

Date:
-----
2025-10-19
