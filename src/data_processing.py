import pandas as pd
import numpy as np
from ast import literal_eval

def read_file(config):
    """
    Read a CSV file from the path specified in the configuration.

    Parameters
    ----------
    config : ConfigParser
        Configuration object containing the file path under the "FILES" section.

    Returns
    -------
    pd.DataFrame
        The loaded data as a pandas DataFrame.
    """
    return pd.read_csv(config["FILES"]["DATA_PATH"], sep=";")

def add_datetime(df_one_cell):
    """
    Add a unique datetime column to each row of a single cell's data.

    Parameters
    ----------
    df_one_cell : pd.DataFrame
        DataFrame containing one cell's data with a 'Time' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with a new 'datetime' column representing a unique timestamp.
    """
    df_one_cell["temp_Time"] = pd.to_datetime(df_one_cell["Time"], format="%H:%M")
    df_one_cell = df_one_cell.sort_values("temp_Time")
    df_one_cell["day"] = df_one_cell.groupby("Time").cumcount()
    df_one_cell["datetime"] = (
        pd.to_datetime("2025-01-01") +
        pd.to_timedelta(df_one_cell["day"], unit="D") +
        pd.to_timedelta(df_one_cell["temp_Time"].dt.hour * 60 + df_one_cell["temp_Time"].dt.minute, unit="m")
    )
    df_one_cell.drop(columns=["Time", "temp_Time", "day"], inplace=True)
    return df_one_cell

def replace_missing_dates(df_one_cell):
    """
    Fill missing datetime indices at 15-minute intervals.

    Parameters
    ----------
    df_one_cell : pd.DataFrame
        DataFrame with 'datetime' as a column.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by datetime, with missing timestamps filled.
    """
    df_one_cell = df_one_cell.sort_values("datetime")
    df_one_cell.set_index("datetime", inplace=True, drop=True)
    return df_one_cell.asfreq("15min")

def detect_replace_outliers(df_one_cell, oss_counter, window, threshold):
    """
    Detect and replace outliers in a metric using Z-score method and rolling statistics.

    Parameters
    ----------
    df_one_cell : pd.DataFrame
        DataFrame with time-series data.
    oss_counter : str
        Column name of the OSS counter to clean.
    window : int
        Size of the rolling window in time steps.
    threshold : float
        Z-score threshold to identify outliers.

    Returns
    -------
    pd.Series
        Cleaned series with outliers replaced by time interpolation.
    """
    df_copy = df_one_cell.copy()
    rolling_mean = df_copy[oss_counter].rolling(window=window, min_periods=1).mean()
    rolling_std = df_copy[oss_counter].rolling(window=window, min_periods=1).std()
    z_scores = (df_copy[oss_counter][window:] - rolling_mean[window:]) / rolling_std[window:]
    outliers = z_scores.abs() > threshold
    df_copy.loc[outliers[outliers].index, oss_counter] = np.nan
    df_copy[oss_counter] = df_copy[oss_counter].interpolate(method="time", limit_direction="both")
    return df_copy[oss_counter]

def process_oss_counters(cell_name, df_one_cell, oss_counters):
    """
    Process OSS counters by adding datetime, interpolating missing values, and replacing outliers.

    Parameters
    ----------
    cell_name : str
        Name of the cell.
    df_one_cell : pd.DataFrame
        DataFrame for a single cell.
    oss_counters : list of str
        List of OSS counter column names to process.

    Returns
    -------
    pd.DataFrame
        Cleaned and processed DataFrame with interpolated values.
    """
    df_one_cell = add_datetime(df_one_cell)
    df_one_cell.drop(columns=["Unusual"], inplace=True)
    df_processed = replace_missing_dates(df_one_cell)
    for counter in oss_counters:
        df_processed[counter] = detect_replace_outliers(df_processed, counter, window=96, threshold=4)
    df_processed["CellName"] = df_processed["CellName"].fillna(cell_name)
    return df_processed

def add_target(df):
    """
    Add a binary classification target for 24-hour future 'Unusual' status.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime and 'Unusual' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with a new 'target' column.
    """
    df["datetime_after_24h"] = df["datetime"] + pd.Timedelta(hours=24)
    future = df[["CellName", "datetime", "Unusual"]].copy()
    future.rename(columns={"datetime": "datetime_after_24h", "Unusual": "Unusual_after_24h"}, inplace=True)
    df = df.merge(future, on=["CellName", "datetime_after_24h"], how="left")
    df.rename(columns={"Unusual_after_24h": "target"}, inplace=True)
    df.drop(columns=["datetime_after_24h", "Unusual"], inplace=True)
    df = df[~df["target"].isna()].reset_index(drop=True)
    return df

def add_rolling_features(df, metric, window_size):
    """
    Add rolling mean and max features for a given metric and window size.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing OSS metrics.
    metric : str
        Column name of the metric to roll.
    window_size : int
        Window size in hours (each hour assumed to be 4 data points).

    Returns
    -------
    pd.DataFrame
        DataFrame with new rolling mean and max features.
    """
    window = window_size * 4  # 4 samples per hour
    df[f"{metric}_mean_rolling{window_size}h"] = df.groupby("CellName")[metric].transform(lambda x: x.rolling(window).mean())
    df[f"{metric}_max_rolling{window_size}h"] = df.groupby("CellName")[metric].transform(lambda x: x.rolling(window).max())
    return df

def add_ratio_features(df):
    """
    Add ratio-based features between download and upload metrics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OSS counters.

    Returns
    -------
    pd.DataFrame
        DataFrame with new ratio features.
    """
    df["thr_dl_to_ul"] = df["meanThr_DL"] / (df["meanThr_UL"] + 1e-6 )
    df["ue_dl_to_ul"] = df["meanUE_DL"] / (df["meanUE_UL"] + 1e-6 )
    df["prb_dl_to_ul"] = df["PRBUsageDL"] / (df["PRBUsageUL"] + 1e-6 )
    return df

def add_all_features(df, oss_counters):
    """
    Add rolling and ratio-based features based on configuration.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with datetime and OSS counters.
    config : ConfigParser
        Configuration object containing OSS counter list.

    Returns
    -------
    pd.DataFrame
        DataFrame with added features.
    """
    window_sizes = [1, 5, 7, 15]
    for oss_counter in oss_counters:
        for ws in window_sizes:
            df = add_rolling_features(df, oss_counter, ws)
    df["hour"] = df["datetime"].dt.hour
    df = add_ratio_features(df)
    return df

def handle_missing_features(df):
    """
    Backfill missing feature values and drop identifier columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with missing values and 'CellName', 'datetime'.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with features ready for modeling.
    """
    df = df.groupby("CellName", group_keys=False).apply(lambda x: x.bfill())
    df.drop(columns=["CellName", "datetime"], inplace=True)
    return df

def process_data_for_network_activity_classification(config):
    """
    Full preprocessing pipeline for classification task.

    Parameters
    ----------
    config : ConfigParser
        Configuration object.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame ready for classification modeling.
    """
    df = read_file(config)
    oss_counters = literal_eval(config["MODELS"]["OSS_COUNTERS"])
    df = pd.concat([add_datetime(group) for _, group in df.groupby("CellName")]).reset_index(drop=True)
    df = add_target(df)
    df = add_all_features(df, oss_counters)
    df = handle_missing_features(df)
    return df

def process_data_for_oss_counters_forecasting(config):
    """
    Full preprocessing pipeline for time-series forecasting.

    Parameters
    ----------
    config : ConfigParser
        Configuration object with OSS counter list.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame ready for forecasting modeling.
    """
    df = read_file(config)
    oss_counters = literal_eval(config["MODELS"]["OSS_COUNTERS"])
    df_processed = pd.concat([
        process_oss_counters(cell_name, group, oss_counters)
        for cell_name, group in df.groupby("CellName")
    ]).reset_index()
    return df_processed
