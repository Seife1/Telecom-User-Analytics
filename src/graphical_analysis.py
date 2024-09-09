from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Visualize data distributions for quantitative variables using histograms and boxplots."""
def graphical_univariate_analysis(df):
    # Select numeric columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    n_cols = 5
    n_rows = (len(numeric_df.columns) + n_cols - 1) // n_cols  # To ensure enough rows

    # Create subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(16, n_rows * 4))

    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    # Iterate over each column and create a histogram plot for each
    for i, column in enumerate(numeric_df.columns):
        sns.histplot(numeric_df[column], bins=30, kde=True, ax=axes[i], color='skyblue')
        axes[i].set_title(f'Histogram of {column}')
        axes[i].set_xlabel(column)

    # Remove any empty subplots if the number of columns is not a multiple of n_cols
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plots
    plt.show()

# Explore the relationship between application-specific data and total data.
def bivariate_analysis(df):
    apps = ['Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other']

    # Create a new column for total data usage
    df['Total_Data'] = df['Total DL (Bytes)'] + df['Total UL (Bytes)']

    # Create subplots, adjusting for the number of apps
    fig, axes = plt.subplots(nrows=len(apps) // 2 + len(apps) % 2, ncols=2, figsize=(12, 10))

    # Flatten axes for easier iteration
    axes = axes.flatten()

    # Iterate over each app and plot a scatter plot
    for i, app in enumerate(apps):
        # Calculate app-specific total data usage
        df[f'{app}_Total'] = df[f'{app} DL (Bytes)'] + df[f'{app} UL (Bytes)']

        # Plot scatterplot in the corresponding subplot
        sns.scatterplot(x=df[f'{app}_Total'], y=df['Total_Data'], ax=axes[i])
        axes[i].set_title(f'{app} vs. Total Data Usage')
        axes[i].set_xlabel(f'{app} Data (Bytes)')
        axes[i].set_ylabel('Total Data (Bytes)')

    # Remove any empty subplots if the number of apps is odd
    if len(apps) % 2 != 0:
        fig.delaxes(axes[-1])  # Remove the last unused axis

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the figure
    plt.show()
    
# Compute correlation matrix for application-specific data usage.
def correlation_analysis(df):
    app_columns = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 
                   'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']

    correlation_matrix = df[app_columns].corr()

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    cmap = sns.diverging_palette(225, 20, as_cmap=True)
    sns.heatmap(correlation_matrix, annot=True, cmap=cmap, fmt=".4f")
    plt.title('Correlation Matrix for Application Data')
    plt.show()


# Perform PCA on application-specific data to reduce dimensionality.
def perform_pca(df):
    app_columns = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 
                   'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[app_columns])

    # Apply PCA
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    # Variance explained by each component
    explained_variance = pca.explained_variance_ratio_
    
    plt.figure(figsize=(6, 4))
    plt.bar(range(1, 3), explained_variance, tick_label=["PC1", "PC2"])
    plt.title('Explained Variance by Principal Components')
    plt.show()

    return pca_data, explained_variance