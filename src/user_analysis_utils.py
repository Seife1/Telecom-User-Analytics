import pandas as pd

# Identify Top 10 Handsets
def top_10_handsets(df):
    top_handsets = df['Handset Type'].value_counts().head(10)
    return top_handsets

# Identify Top 3 Handset Manufacturers
def top_3_manufacturers(df):
    top_manufacturers = df['Handset Manufacturer'].value_counts().head(3)
    return top_manufacturers

# Identify Top 5 Handsets per Top 3 Manufacturers
def top_5_handsets_per_manufacturer(df, top_manufacturers):
    top_5_handsets_per_manufacturer = {}
    for manufacturer in top_manufacturers.index:
        # Filter handsets for the current manufacturer
        handsets = df[df['Handset Manufacturer'] == manufacturer]['Handset Type']
        # Get the top 5 handsets for this manufacturer
        top_handsets = handsets.value_counts().head(5)
        top_5_handsets_per_manufacturer[manufacturer] = top_handsets
    return top_5_handsets_per_manufacturer

# Aggregate User Data

# Aggregate number of xDR sessions per user (MSISDN/Number).
def aggregate_xdr_sessions(df):
    return df.groupby('MSISDN/Number')['Bearer Id'].count().reset_index(name='session_count')

# Aggregate total session duration per user (MSISDN/Number).
def aggregate_session_duration(df):
    return df.groupby('MSISDN/Number')['Dur. (ms)'].sum().reset_index(name='total_duration_ms')

# Aggregate total download (DL) and upload (UL) data per user.
def aggregate_total_data(df):
    return df.groupby('MSISDN/Number').agg(
        total_download=('Total DL (Bytes)', 'sum'),
        total_upload=('Total UL (Bytes)', 'sum')
    ).reset_index()

# Aggregate application data per user.
def aggregate_application_data(df):
    applications = ['Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other']
    
    agg_dict = {}
    for app in applications:
        # Ensure the columns exist before trying to use them
        dl_col = f'{app} DL (Bytes)'
        ul_col = f'{app} UL (Bytes)'
        if dl_col in df.columns and ul_col in df.columns:
            # Use lambda functions to aggregate per user
            agg_dict[f'{app}_total_data'] = lambda x: (x[dl_col] + x[ul_col]).sum()
        else:
            print(f"Warning: Columns {dl_col} or {ul_col} are missing in the DataFrame.")
    
    # Summing data for each application per user
    return df.groupby('MSISDN/Number').apply(lambda x: pd.Series({k: v(x) for k, v in agg_dict.items()})).reset_index()

# Aggregate customer behavior metrics into one DataFrame.
def aggregate_user_data(df):
    # Aggregate sessions, duration, and data usage
    session_count_df = aggregate_xdr_sessions(df)
    session_duration_df = aggregate_session_duration(df)
    total_data_df = aggregate_total_data(df)
    app_data_df = aggregate_application_data(df)

    # Merge all the results into a single DataFrame
    result = session_count_df.merge(session_duration_df, on='MSISDN/Number')
    result = result.merge(total_data_df, on='MSISDN/Number')
    result = result.merge(app_data_df, on='MSISDN/Number')

    return result