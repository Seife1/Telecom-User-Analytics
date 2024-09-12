import unittest
import sys
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from satisfaction_utils import get_user_data, calculate_engagement_score, calculate_experience_score, calculate_satisfaction_score

class TestSatisfactionFunctions(unittest.TestCase):
    
    def setUp(self):
        # Set up a DataFrame to mimic user data
        self.df = pd.DataFrame({
            'MSISDN/Number': [123, 456],
            'session_count': [10, 15],
            'total_duration_ms': [5000, 7000],
            'total_download': [200000, 300000],
            'total_upload': [100000, 150000],
            'Avg RTT DL (ms)': [100, 80],
            'Avg RTT UL (ms)': [80, 60],
            'Avg Bearer TP DL (kbps)': [500, 600],
            'Avg Bearer TP UL (kbps)': [300, 400],
            'TCP DL Retrans. Vol (Bytes)': [10000, 8000],
            'TCP UL Retrans. Vol (Bytes)': [8000, 6000]
        })

        # Engagement and experience centroids (dummy values)
        self.engagement_centroid = np.array([10, 5000, 200000, 100000])
        self.experience_centroid = np.array([100, 80, 500, 300, 10000, 8000])

    def test_get_user_data(self):
        user_data = get_user_data(self.df, 0)
        expected_data = np.array([10, 5000, 200000, 100000, 100, 80, 500, 300, 10000, 8000])
        np.testing.assert_array_equal(user_data, expected_data)

    def test_calculate_engagement_score(self):
        user_data = np.array([10, 5000, 200000, 100000])
        engagement_score = calculate_engagement_score(user_data, self.engagement_centroid)
        expected_score = euclidean(user_data, self.engagement_centroid)
        self.assertAlmostEqual(engagement_score, expected_score)

    def test_calculate_experience_score(self):
        user_data = np.array([100, 80, 500, 300, 10000, 8000])
        experience_score = calculate_experience_score(user_data, self.experience_centroid)
        expected_score = euclidean(user_data, self.experience_centroid)
        self.assertAlmostEqual(experience_score, expected_score)

    def test_calculate_satisfaction_score(self):
        engagement_score = 0.1
        experience_score = 0.2
        satisfaction_score = calculate_satisfaction_score(engagement_score, experience_score)
        expected_satisfaction = (0.1 + 0.2) / 2
        self.assertAlmostEqual(satisfaction_score, expected_satisfaction)

if __name__ == '__main__':
    unittest.main()
