import unittest
import pandas as pd
from src.utils import get_config
from src.data_processing import read_file, add_datetime, replace_missing_dates, detect_replace_outliers

class TestDataProcessing(unittest.TestCase):
    def test_read_file(self):
        config_data = get_config("configuration.ini")
        df = read_file(config_data)
        list_of_columns = [
            'Time', 'CellName', 'PRBUsageUL', 'PRBUsageDL', 'meanThr_DL',
            'meanThr_UL', 'maxThr_DL', 'maxThr_UL', 'meanUE_DL', 'meanUE_UL',
            'maxUE_DL', 'maxUE_UL', 'maxUE_UL+DL', 'Unusual'
        ]
        self.assertListEqual(df.columns.tolist(), list_of_columns)
        self.assertGreater(df.shape[0], 10)

    def test_add_datetime(self):
        df_without_datetime = pd.DataFrame({
            "Time": ["06:00", "06:30", "06:45", "07:00"],
            "CellName": ['3BLTE', '3BLTE', '3BLTE', '3BLTE']
        })
        df_with_datetime = add_datetime(df_without_datetime)
        self.assertIn("datetime", df_with_datetime.columns)
        self.assertIsInstance(df_with_datetime["datetime"].iloc[0], pd.Timestamp)
        self.assertListEqual(
            df_without_datetime["CellName"].unique().tolist(),
            df_with_datetime["CellName"].unique().tolist()
        )

    def test_replace_missing_dates(self):
        df_without_datetime = pd.DataFrame({
            "Time": ["06:00", "06:30", "06:45", "07:00"],
            "CellName": ['3BLTE', '3BLTE', '3BLTE', '3BLTE']
        })
        df_with_datetime = add_datetime(df_without_datetime)
        df_with_filled_missing_datetime = replace_missing_dates(df_with_datetime)
        self.assertEqual(df_with_filled_missing_datetime.index.freqstr, "15min")

    def test_detect_replace_outliers(self):
        df_with_outliers = pd.DataFrame({
            "datetime": pd.date_range("2025-01-01 06:00:00", periods=13, freq="15min"),
            "CellName": ['3BLTE'] * 13,
            "PRBUsageDL": [3, 2, 4, 3, 2, 4, 100, 3, 2, 4, 3, 2, 4]
        })
        df_with_outliers.set_index('datetime', inplace=True)
        df_without_outliers_prbdl = detect_replace_outliers(df_with_outliers, "PRBUsageDL", 5, 1.5)
        self.assertEqual(len(df_with_outliers), len(df_without_outliers_prbdl))
        self.assertEqual(df_without_outliers_prbdl.isna().sum().sum(), 0)

if __name__ == '__main__':
    unittest.main()
