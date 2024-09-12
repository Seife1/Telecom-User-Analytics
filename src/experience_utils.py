from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

# Aggregate average RTT, Throughput, and Retransmission data per user
def aggregate_experience_metrics(df):
    return df.groupby('MSISDN/Number').agg({
        'Avg RTT DL (ms)': 'mean',
        'Avg RTT UL (ms)': 'mean',
        'Avg Bearer TP DL (kbps)': 'mean',
        'Avg Bearer TP UL (kbps)': 'mean',
        'TCP DL Retrans. Vol (Bytes)': 'sum',
        'TCP UL Retrans. Vol (Bytes)': 'sum'
    }).reset_index()


# To ensure that all metrics are on the same scale, we'll normalize the data.
def normalize_experience_data(df):
    # Select columns to normalize
    cols_to_normalize = ['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)',
                         'Avg Bearer TP UL (kbps)', 'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']
    
    scaler = StandardScaler()
    df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
    
    return df

# Elbow method to find the optimal number of clusters
def plot_elbow_method(df):
    cols_for_clustering = ['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)',
                           'Avg Bearer TP UL (kbps)', 'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']
    wcss = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(df[cols_for_clustering])
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    sns.lineplot(x=range(1, 11), y=wcss)
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Method for Optimal Clusters')
    plt.show()

# Apply K-means Clustering Based on Chosen Number of Clusters
def apply_kmeans(df, num_clusters=3):
    cols_for_clustering = ['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)',
                           'Avg Bearer TP UL (kbps)', 'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']
    
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
    df['experience_cluster'] = kmeans.fit_predict(df[cols_for_clustering].values)
    
    return df, kmeans

# Visualization function with centroids for RTT DL and Throughput DL with 7 clusters
def visualize_clusters_with_centroids(df, kmeans, num_clusters=3):
    # Extract the two features (RTT DL and Throughput DL) for plotting
    x = df[['Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)']].values
    y_pred = df['experience_cluster']  # Cluster labels from KMeans
    centers = kmeans.cluster_centers_  # Cluster centroids

    # Verify that x and y_pred have the same number of data points
    if len(x) != len(y_pred):
        raise ValueError(f"Mismatch between the number of data points ({len(x)}) and the number of cluster labels ({len(y_pred)}).")

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
    
    # Set plot labels and title
    plt.xlabel('Avg RTT DL (ms)')
    plt.ylabel('Avg Bearer TP DL (kbps)')
    plt.title('User Experience Clusters with Centroids')

    # Place the legend outside the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    plt.tight_layout()
    plt.show()