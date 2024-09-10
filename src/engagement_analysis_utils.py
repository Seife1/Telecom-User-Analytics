from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


def normalize_data(df):
    # Select only the columns you want to normalize
    cols_to_normalize = ['session_count', 'total_duration_ms', 'total_download', 'total_upload']
    scaler = StandardScaler()

    # Apply scaling and return a new DataFrame with normalized columns
    normalized_df = df.copy()
    normalized_df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
    
    return normalized_df

# Elbow method to find the optimal number of clusters
def plot_elbow_method(df, x_features, y_features):
    cols_for_clustering = [x_features, y_features]
    wcss = []

    for i in range(1, 11):
        clustering = KMeans(n_clusters=i, init='k-means++', random_state=42)
        clustering.fit(df[cols_for_clustering])
        wcss.append(clustering.inertia_)

    ks = range(1, 11)
    plt.figure(figsize=(8, 6))
    sns.lineplot(x = ks, y = wcss)
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.title("Elbow Method to find Optimal Clusters")
    plt.show()

# Apply K-means clustering
def apply_kmeans(df, x_features, y_features, num_clusters=3):
    # Select the normalized columns for clustering
    cols_for_clustering = [x_features, y_features]
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, n_init='auto',init='k-means++', random_state=42)
    df['engagement_cluster'] = kmeans.fit_predict(df[cols_for_clustering].values)
    
    return df, kmeans

# Visualize the clusters based on session count and total download data
def visualize_clusters_with_centroids(df, kmeans, x_features, y_features):
    # Extract clustering results
    x = df[[x_features, y_features]].values  # Two dimensions to visualize
    y_pred = df['engagement_cluster']        # Cluster predictions from KMeans
    centers = kmeans.cluster_centers_        # Cluster centroids

    # Scatter plots for each cluster and its centroid
    plt.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1], color='orange', alpha=0.5, label='Cluster 0', marker='.')
    plt.scatter(centers[0, 0], centers[0, 1], color='orange', s=250, marker='*', label='Centroid 0')

    plt.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1], color='purple', alpha=0.5, label='Cluster 1', marker='.')
    plt.scatter(centers[1, 0], centers[1, 1], color='purple', s=250, marker='*', label='Centroid 1')

    if len(centers) > 2:
        plt.scatter(x[y_pred == 2, 0], x[y_pred == 2, 1], color='red', alpha=0.5, label='Cluster 2', marker='.')
        plt.scatter(centers[2, 0], centers[2, 1], color='red', s=250, marker='*', label='Centroid 2')

    # If you have more clusters, repeat for them
    if len(centers) > 3:
        plt.scatter(x[y_pred == 3, 0], x[y_pred == 3, 1], color='blue', alpha=0.5, label='Cluster 3', marker='.')
        plt.scatter(centers[3, 0], centers[3, 1], color='blue', s=250, marker='*', label='Centroid 3')

    if len(centers) > 4:
        plt.scatter(x[y_pred == 4, 0], x[y_pred == 4, 1], color='green', alpha=0.5, label='Cluster 4', marker='.')
        plt.scatter(centers[4, 0], centers[4, 1], color='green', s=250, marker='*', label='Centroid 4')

    # Set labels and title
    plt.xlabel(x_features.replace('_', ' ').title())
    plt.ylabel(y_features.replace('_', ' ').title())
    plt.title(f'{x_features.title()} vs {y_features.title()} with KMeans Clusters')

    # Place the legend outside the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    plt.tight_layout()
    plt.show()


# Deriving Top 10 Most Engaged Users Per Application
def top_10_users_per_application(df):
    applications = ['Social Media_total_data', 'Google_total_data', 'Email_total_data', 
                    'Youtube_total_data', 'Netflix_total_data', 'Gaming_total_data', 
                    'Other_total_data']
    
    top_10_users = {}
    for app in applications:
        top_10_users[app] = df[['MSISDN/Number', app]].sort_values(by=app, ascending=False).head(10)
    
    return top_10_users

# Visualizing the Top 3 Most Used Applications
def plot_top_3_applications(df):
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
    plt.figure(figsize=(8, 6))
    plt.bar(apps, usage, color=['blue', 'orange', 'green'])
    
    # Add title and labels
    plt.title("Top 3 Most Used Applications by Total Data Usage")
    plt.xlabel("Applications")
    plt.ylabel("Total Data Usage (Bytes)")
    
    # Show plot
    plt.tight_layout()
    plt.show()