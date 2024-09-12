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
def plot_elbow_method(df):
    cols_for_clustering = ['session_count', 'total_duration_ms', 'total_download', 'total_upload']
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
def apply_kmeans(df, num_clusters=3):
    # Select the normalized columns for clustering
    cols_for_clustering = ['session_count', 'total_duration_ms', 'total_download', 'total_upload']

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, n_init='auto',init='k-means++', random_state=42)
    df['engagement_cluster'] = kmeans.fit_predict(df[cols_for_clustering].values)
    
    return df, kmeans

# Visualize the clusters based on session count and total download data
def visualize_clusters_with_centroids(df, kmeans, x_features, y_features, num_clusters):
    # Extract clustering results
    x = df[[x_features, y_features]].values  # Two dimensions to visualize
    y_pred = df['engagement_cluster']        # Cluster predictions from KMeans
    centers = kmeans.cluster_centers_        # Cluster centroids

    # Get unique cluster labels to determine the number of clusters dynamically
    unique_clusters = sorted(set(y_pred))

    # Define color map for 10 clusters
    colors = sns.color_palette("Set2", num_clusters)

    if num_clusters > len(colors):
        raise ValueError(f"Not enough colors defined for the number of clusters ({num_clusters}).")

    # Scatter plot for each cluster with different colors
    for i in unique_clusters:
        plt.scatter(x[y_pred == i, 0], x[y_pred == i, 1], color=colors[i], alpha=0.5, label=f'Cluster {i}', marker='.')
        plt.scatter(centers[i, 0], centers[i, 1], color=colors[i], s=250, marker='+', label=f'Centroid {i}')

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