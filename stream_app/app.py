import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

import sys
import os
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../assets')))


# ==================================================Page configuration===========================================
st.set_page_config(
    page_title="Solar Farm Data Dashboard",
    page_icon="☎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define base path relative to current file's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../assets/')

full_path = os.path.join(DATA_DIR, 'satisfaction_aggregated_data.csv')
if not os.path.exists(full_path):
    raise FileNotFoundError(f"The file at {full_path} does not exist.")
data = pd.read_csv(full_path)

engage = os.path.join(DATA_DIR, 'engagement_aggregated_data.csv')
if not os.path.exists(engage):
    raise FileNotFoundError(f"The file at {engage} does not exist.")
df = pd.read_csv(engage)

processed_path = os.path.join(DATA_DIR, 'processed_data.csv')
if not os.path.exists(processed_path):
    raise FileNotFoundError(f"The file at {processed_path} does not exist.")
processed_data = pd.read_csv(processed_path)

st.title("☎ Telecom User Analysis")
# ========================================Page: User Overview Analysis============================================================
def user_overview_analysis():
    st.title("User Overview Analysis")
    st.write('''
             ### Overview
             This project focuses on analyzing telecom user behavior, engagement, and experience through a series of tasks aimed at understanding user satisfaction, identifying key patterns, and providing actionable insights for telecom providers. The project applies clustering techniques and machine learning models to identify patterns in user engagement, network experience, and overall customer satisfaction.
             ''')
    # Display top handsets and manufacturers
    col1, col2 = st.columns(2)

    # Top 10 Handsets
    with col1:
        st.subheader("Top 10 Handsets")
        top_handsets = processed_data['Handset Type'].value_counts().head(10)

        # Plot horizontal bar chart for each manufacturer
        fig, ax = plt.subplots(figsize=(10,7))
        top_handsets.plot(kind='barh', ax=ax, color=['orange', 'teal', 'blue', 'red', 'green', 'purple', 'brown', 'pink', 'gray', 'black'])
        ax.set_xlabel('Count')
        ax.set_ylabel('Handset Manufacturer')
        plt.tight_layout()
        st.pyplot(fig)

    # Top 3 Handset Manufacturers
    with col2:
        st.subheader("Top 3 Manufacturers")
        top_manufacturers = processed_data['Handset Manufacturer'].value_counts().head(3)

        # Plot horizontal bar chart for each manufacturer
        fig, ax = plt.subplots(figsize=(10, 7))
        top_manufacturers.plot(kind='barh', ax=ax, color=['orange', 'teal', 'blue'])
        ax.set_xlabel('Count')
        ax.set_ylabel('Handset Type')
        plt.tight_layout()
        st.pyplot(fig)

    # Top 5 Handsets per Top 3 Manufacturers
    st.subheader("Top 5 Handsets per Manufacturer")

    # Create a container for the horizontal layout
    top_5_handsets_per_manufacturer = {}

    # Use columns to display the results horizontally
    columns = st.columns(len(top_manufacturers))

    # Iterate through top 3 manufacturers
    for col, manufacturer in zip(columns, top_manufacturers.index):
        handsets = processed_data[processed_data['Handset Manufacturer'] == manufacturer]['Handset Type'].value_counts().head(5)
        top_5_handsets_per_manufacturer[manufacturer] = handsets
        
        # Display each manufacturer and their top 5 handsets in a separate column
        with col:
            st.write(f"**{manufacturer}**")
            st.write(handsets)        

# =======================================Page: User Engagement Analysis===========================================================
def user_engagement_analysis():
    st.title("User Engagement Analysis")

    # Top 10 customers by session count
    top_session_count = data[['MSISDN/Number', 'session_count']].sort_values(by='session_count', ascending=False).head(10)

    # Top 10 customers by total download
    top_total_download = data[['MSISDN/Number', 'total_download']].sort_values(by='total_download', ascending=False).head(10)

    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 2 columns of plots

    # Bar plot for session frequency
    ax[0].bar(top_session_count['MSISDN/Number'].astype(str), top_session_count['session_count'], color='blue')
    ax[0].set_title("Top 10 Customers by Session Frequency")
    ax[0].set_xlabel("MSISDN/Number")
    ax[0].set_ylabel("Session Frequency")
    ax[0].set_xticklabels(top_session_count['MSISDN/Number'].astype(str), rotation=45, ha='right')

    # Bar plot for total downloads
    ax[1].bar(top_total_download['MSISDN/Number'].astype(str), top_total_download['total_download'], color='green')
    ax[1].set_title("Top 10 Customers by Total Download")
    ax[1].set_xlabel("MSISDN/Number")
    ax[1].set_ylabel("Total Download [bytes]")
    ax[1].set_xticklabels(top_total_download['MSISDN/Number'].astype(str), rotation=45, ha='right')

    # Adjust layout to prevent overlap
    fig.tight_layout()

    # Show plot with Streamlit
    st.pyplot(fig)

    # Plot: Engagement Clusters
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # First scatter plot: session_count vs total_download
    sns.scatterplot(x='session_count', y='total_download', hue='engagement_cluster', data=data, ax=ax[0])
    ax[0].set_title('User Engagement Clusters (Session Count vs Total Download)')
    ax[0].set_xlabel('Session Count')
    ax[0].set_ylabel('Total Download')

    # Second scatter plot: session_count vs total_duration_ms
    sns.scatterplot(x='session_count', y='total_duration_ms', hue='engagement_cluster', data=data, ax=ax[1])
    ax[1].set_title('User Engagement Clusters (Session Count vs Total Duration)')
    ax[1].set_xlabel('Session Count')
    ax[1].set_ylabel('Total Duration (ms)')

    # Adjust layout and display the plot in Streamlit
    plt.tight_layout()
    st.pyplot(fig)

    # Calculate total data usage for each application
    app_usage = {
        'Social Media': df['Social Media_total_data'].sum(),
        'Google': df['Google_total_data'].sum(),
        'Email': df['Email_total_data'].sum(),
        'Youtube': df['Youtube_total_data'].sum(),
        'Netflix': df['Netflix_total_data'].sum(),
        'Gaming': df['Gaming_total_data'].sum(),
        'Other': df['Other_total_data'].sum(),
    }

    # Sort applications by total data usage and select top 3
    sorted_apps = sorted(app_usage.items(), key=lambda x: x[1], reverse=True)[:3]

    # Extract application names and usage values
    apps = [app[0] for app in sorted_apps]
    usage = [app[1] for app in sorted_apps]

    # Plot a bar chart
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(apps, usage, color=['blue', 'orange', 'green'])

    # Add title and labels
    ax.set_title("Top 3 Most Used Applications by Total Data Usage")
    ax.set_xlabel("Applications")
    ax.set_ylabel("Total Data Usage (Bytes)")

    # Show plot in Streamlit
    st.pyplot(fig)
# =================================================Page: Experience Analysis===========================================
def experience_analysis():
    st.title("Experience Analysis")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='Avg RTT DL (ms)', y='Avg Bearer TP DL (kbps)', hue='experience_cluster', data=data, ax=ax)
    st.pyplot(fig)

    st.write('''
             This scatter plot visualizes the relationship between average throughput in kilobits per second and average round-trip time in milliseconds, color-coded by the "experience_cluster" (with clusters labeled from 0 to 6).

    Key observations:
    * Darker points (clusters 5 and 6) are concentrated in the upper-left quadrant, indicating better performance (lower RTT and higher throughput).
    * Lighter points (clusters 0 to 2) dominate the lower-left quadrant and spread further along the x-axis, indicating poorer performance with higher RTT and lower throughput.''')
    

# ===============================================Page: Satisfaction Analysis============================================
def satisfaction_analysis():
    st.title("Satisfaction Analysis")

    top_10_satisfied = data.sort_values(by='satisfaction_score', ascending=False).head(10)

    # Create a bar plot for the top 10 customers
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(top_10_satisfied['MSISDN/Number'].astype(str), top_10_satisfied['satisfaction_score'], color='orange')

    # Add title and labels
    ax.set_title("Top 10 Customers per Satisfaction Score")
    ax.set_xlabel("MSISDN/Number")
    ax.set_ylabel("Satisfaction Score")

    # Rotate x-axis labels for better readability
    ax.set_xticklabels(top_10_satisfied['MSISDN/Number'].astype(str), ha='right', rotation=45)

    # Adjust layout to prevent overlap
    fig.tight_layout()

    # Show plot with Streamlit
    st.pyplot(fig)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    data['kmeans_cluster'] = kmeans.fit_predict(data[['engagement_score', 'experience_score']])

    st.write('''
             This K-means clustering analysis below shows two distinct groups based on engagement and experience scores. 
        ''')
    # Plot the K-means clustering result
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='engagement_score', y='experience_score', hue='kmeans_cluster', data=data, ax=ax)
    ax.set_title('K-means Clustering of Engagement & Experience Scores')

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Calculate cluster aggregates
    cluster_aggregates = data.groupby('kmeans_cluster').agg({
        'engagement_score': 'mean',
        'experience_score': 'mean',
        'satisfaction_score': 'mean'
    })

    # Display cluster aggregates in Streamlit
    st.subheader("Cluster Aggregates")
    st.write(cluster_aggregates)
    st.write('''             
             Cluster 0 (blue) consists of users with low engagement (avg: 1.12) and low experience scores (avg: 1.75), indicating a group with minimal interaction and likely dissatisfaction (avg satisfaction score: 1.43). 
             
             On the other hand, Cluster 1 (orange) includes users with significantly higher engagement (avg: 4.56) and experience scores (avg: 4.80), suggesting a more satisfied and active group (avg satisfaction score: 4.68). 
             
             The plot visually highlights the spread, with Cluster 1 showing more variability in engagement, while Cluster 0 is more concentrated at lower scores. This segmentation can guide targeted strategies to improve the experience of Cluster 0 users while maintaining engagement for Cluster 1.
             ''')
    




# Main function to run the dashboard
def main():
    st.sidebar.title("Dashboard Navigation")
    page = st.sidebar.radio("Go to", ["User Overview Analysis", "User Engagement Analysis", "Experience Analysis", "Satisfaction Analysis"])
    if page == "User Overview Analysis":
        user_overview_analysis()
    elif page == "User Engagement Analysis":
        user_engagement_analysis()
    elif page == "Experience Analysis":
        experience_analysis()
    elif page == "Satisfaction Analysis":
        satisfaction_analysis()

if __name__ == "__main__":
    main()