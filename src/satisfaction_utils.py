import numpy as np
from scipy.spatial.distance import euclidean

# Extract the features for a specific user by index or MSISDN/Number
def get_user_data(df, index):
    # Extract relevant columns for the user (engagement and experience metrics)
    user_data = df.loc[index, ['session_count', 'total_duration_ms', 'total_download', 'total_upload',
                               'Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)',
                                'Avg Bearer TP UL (kbps)', 'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']].values
    return user_data

# Calculate Euclidean distance between user data and centroid
def calculate_engagement_score(user_data, engagement_centroid):
    return euclidean(user_data, engagement_centroid)

def calculate_experience_score(user_data, experience_centroid):
    return euclidean(user_data, experience_centroid)

# Calculate satisfaction score
def calculate_satisfaction_score(engagement_score, experience_score):
    return (engagement_score + experience_score) / 2