# Telecom User Analytics Project
## Overview
This project focuses on analyzing telecom user behavior, engagement, and experience through a series of tasks aimed at understanding user satisfaction, identifying key patterns, and providing actionable insights for telecom providers. The project applies clustering techniques and machine learning models to identify patterns in user engagement, network experience, and overall customer satisfaction.

## Tasks Breakdown
### Task 1: User Overview Analysis
**Objective**: Analyze user device preferences and the top 10 handsets.
* **Methods**: Identified the top handsets used by users and manufacturers.
* **Outcome**: Provided insights on which handsets and manufacturers dominate the user base, offering valuable input for marketing and business decisions.
### Task 2: User Engagement Analysis
* **Objective**: Analyze user engagement metrics, such as session count, session duration, and data usage.
* **Methods**:
P* reprocessed the user engagement data.
* Applied K-Means clustering to classify users into different engagement levels (low, medium, high).
* Aggregated metrics like session frequency, session duration, and total data traffic.
* **Outcome**: Identified key patterns of user engagement and provided recommendations for improving user retention.
### Task 3: User Experience Analytics
**Objective**: Analyze key network parameters such as RTT (Round Trip Time), throughput, and TCP retransmissions to understand user experience.
* **Methods**:
* Preprocessed network experience data.
* Applied K-Means clustering to classify users into experience clusters (poor, average, good experience).
* Visualized experience clusters based on throughput and RTT.
* **Outcome**: Provided insights into network performance and its impact on user satisfaction.
### Task 4: Customer Satisfaction Analysis
**Objective**: Analyze customer satisfaction based on user engagement and experience.
* **Methods**:
* **Engagement Score**: Calculated as the Euclidean distance between a user's data point and the least engaged cluster centroid.
* **Experience Score**: Calculated as the Euclidean distance between a user's data point and the worst experience cluster centroid.
* **Satisfaction Score**: Averaged between the engagement and experience scores.
* **Modeling**: Built a regression model to predict satisfaction scores and applied K-Means clustering to segment users based on satisfaction levels.
* **Data Export**: Exported the final results to a MySQL database for further analysis and reporting (PostgreSQL was used for loading the SQL file, and MySQL was used for the final export).
* **Outcome**: Ranked users by satisfaction and provided insights for improving customer satisfaction.
### Task 5: Dashboard Development
* **Objective**: Design and develop an interactive dashboard to visualize the insights from the analysis tasks.
* **Methods**:
* Used Streamlit to develop a multi-page dashboard with each page dedicated to a specific task.
* Created visualizations for each task using matplotlib, seaborn, and Plotly.
* The dashboard is fully interactive, allowing users to filter and explore insights in real-time.
* Deployed the dashboard, making it accessible via a public URL.
* **Outcome**: Delivered a functional dashboard that visualizes key insights from the analysis, improving stakeholder engagement with the data.

## Key Technologies
`Python`: Core programming language used for data processing, clustering, and modeling.
`Pandas`: For data manipulation and aggregation.
`Scikit-Learn`: For clustering, regression, and machine learning tasks.
`Seaborn & Matplotlib`: For data visualization.
`SQLAlchemy & PostgreSQL/MySQL`: Used for loading data via PostgreSQL and exporting the final results to a MySQL database.
`Streamlit`: For creating the dashboard.
`Docker`: For containerizing the application.