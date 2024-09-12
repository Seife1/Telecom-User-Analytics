import pandas as pd
import numpy as np
import missingno as msno
from sklearn.impute import SimpleImputer
from scipy import stats
from scipy.stats import zscore

def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # dtype of missing values
    mis_val_dtype = df.dtypes

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent, mis_val_dtype], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns={0: 'Missing Values', 1: '% of Total Values', 2: 'Dtype'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")
    print()

    # Return the dataframe with missing information
    return mis_val_table_ren_columns

def visualize_missing_values(df):
    return msno.matrix(df)

def replace_missing_values(df):
    # Checking missing values
    missing_values = df.isnull().sum()

    # List columns with missing values only
    missing_columns = missing_values[missing_values > 0].index.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Ensure the column names are used for indexing
    missing_categorical_columns = [col for col in missing_columns if col in categorical_features]
    missing_numerical_columns = [col for col in missing_columns if col in numerical_features]

    # Replace missing values in numerical columns
    if len(missing_numerical_columns) > 0:
        print(f"Replacing {len(missing_numerical_columns)} Numeric columns by mean value ...")
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        df[missing_numerical_columns] = imputer.fit_transform(df[missing_numerical_columns])
        print("Replacing Completed!!")
        print()

    # Replace missing values in categorical columns
    if len(missing_categorical_columns) > 0:
        print(f"Replacing {len(missing_categorical_columns)} Categorical columns by most frequent value ...")
        imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        df[missing_categorical_columns] = imputer.fit_transform(df[missing_categorical_columns])
        print("Replacing Completed!!")
        print()

    # Return the missing values matrix or updated DataFrame
    return msno.matrix(df)


def convert_bytes_to_megabytes(df, bytes_data):
    megabyte = 1 * 10e+5
    df[bytes_data] = df[bytes_data] / megabyte
    return df[bytes_data]

def detect_outliers_zscore(df, column_to_process, z_threshold=3):
    if column_to_process not in df.columns:
        raise ValueError(f"Column '{column_to_process}' does not exist in the DataFrame.")
    
    z_scores = zscore(df[column_to_process])
    outlier_indices = np.where(np.abs(z_scores) > z_threshold)[0]
    
    return outlier_indices.tolist()

def remove_outliers(df, column_to_process, z_threshold=3):
    if column_to_process not in df.columns:
        raise ValueError(f"Column '{column_to_process}' does not exist in the DataFrame.")
    
    # Calculate Z-scores for the specified column
    z_scores = zscore(df[column_to_process])
    
    # Create an outlier column to mark outliers
    outlier_column = column_to_process + '_Outlier'
    df[outlier_column] = (np.abs(z_scores) > z_threshold).astype(int)
    
    # Filter out rows where the outlier column is 1 (indicating an outlier)
    df_cleaned = df[df[outlier_column] == 0]
    
    # Drop the outlier column as it's no longer needed
    df_cleaned = df_cleaned.drop(columns=[outlier_column], errors='ignore')

    return df_cleaned

