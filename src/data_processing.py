import argparse
import pandas as pd
import numpy as np
import configparser
from ast import literal_eval
import os

def read_file(config):
    df = pd.read_csv(config["FILES"]["DATA_PATH"], sep=";")
    return df

def add_datetime(df_one_cell):
    df_one_cell["temp_Time"] = pd.to_datetime(df_one_cell["Time"],format="%H:%M")
    df_one_cell = df_one_cell.sort_values(["temp_Time"])
    df_one_cell["day"] = df_one_cell.groupby(["Time"]).cumcount()
    df_one_cell["datetime"] = pd.to_datetime("2025-01-01")+ \
                     pd.to_timedelta(df_one_cell["day"],unit="D")+ \
                     pd.to_timedelta(df_one_cell["temp_Time"].dt.hour*60+
                     df_one_cell["temp_Time"].dt.minute,unit="m")
    df_one_cell.drop(columns=["Time","temp_Time","day"], inplace=True)
    return df_one_cell

def replace_missing_dates(df_one_cell):
    df_one_cell = df_one_cell.sort_values(by='datetime')
    df_one_cell.set_index("datetime",inplace=True,drop=True)
    df_with_added_dates = df_one_cell.asfreq("15min")
    return df_with_added_dates

def detect_replace_outliers(df_one_cell, oss_counter,window, threshold):
    df_one_cell_copy = df_one_cell.copy()
    rolling_mean = df_one_cell_copy[oss_counter].rolling(window=96, min_periods=1).mean()
    rolling_std =df_one_cell_copy[oss_counter].rolling(window=96, min_periods=1).std()
    z_scores = (df_one_cell_copy[oss_counter][window:]-rolling_mean[window:])/rolling_std[window:]
    outliers = z_scores.abs()>threshold
    df_one_cell_copy.loc[outliers[outliers == True].index, oss_counter] = np.nan
    df_one_cell_copy[oss_counter] = df_one_cell_copy[oss_counter].interpolate(method="time", limit_direction="both")
    return df_one_cell_copy[oss_counter]

def process_oss_counters(cell_name, df_one_cell, oss_counters):
    df_one_cell = add_datetime(df_one_cell)
    df_one_cell.drop(columns=["Unusual"],inplace=True)
    df_one_cell_with_added_dates = replace_missing_dates(df_one_cell)
    for oss_counter in oss_counters:
        df_one_cell_with_added_dates[oss_counter] = (
            detect_replace_outliers(df_one_cell_with_added_dates, oss_counter, 96,  4))
    df_one_cell_with_added_dates["CellName"] = (
        df_one_cell_with_added_dates["CellName"].fillna(cell_name))
    return df_one_cell_with_added_dates

def add_target(df):
    df["datetime_after_24h"] = df["datetime"] + pd.Timedelta(hours=24)
    df_after_24h = df[["CellName", "datetime", "Unusual"]].copy()
    df_after_24h.rename(columns={"datetime": "datetime_after_24h",
                                "Unusual": "Unusual_after_24h"}, inplace=True)
    df = df.merge(df_after_24h, on=["CellName", "datetime_after_24h"], how="left")
    df.rename(columns={"Unusual_after_24h":"target"},inplace=True)
    df.drop(columns=["datetime_after_24h", "Unusual"], inplace=True)
    df = df[~df["target"].isna()]
    df.reset_index(inplace=True,drop=True)
    return df

def add_rolling_features(df, metric, window_size):
    df["{m}_mean_rolling{ws}h".format(m=metric, ws = str(window_size))] = (
        df.groupby("CellName")[metric]
        .transform(lambda x: x.rolling(window=window_size*4).mean()))
    df["{m}_max_rolling{ws}h".format(m=metric, ws = str(window_size))] = (
        df.groupby("CellName")[metric]
        .transform(lambda x: x.rolling(window=window_size*4).max()))
    return df

def add_ratio_features(df):
    df['thr_dl_to_ul'] = df["meanThr_DL"]/(df["meanThr_UL"]+0.000001)
    df['ue_dl_to_ul'] = df["meanUE_DL"]/(df["meanUE_UL"]+0.000001)
    df['prb_dl_to_ul'] = df["PRBUsageDL"]/(df["PRBUsageUL"]+0.000001)
    return df

def add_all_features(df,config):
    window_sizes = [1,5,7,15]
    for metric in literal_eval(config["MODELS"]["OSS_COUNTERS"]):
        for window_size in window_sizes:
            df = add_rolling_features(df, metric, window_size)
    df["hour"] = df["datetime"].dt.hour
    df = add_ratio_features(df)
    return df

def handle_missing_features(df):
    # backward fill, fill with previous valid value
    df = df.groupby("CellName", group_keys=False).apply(lambda x: x.bfill())
    df.drop(columns=["CellName","datetime"], inplace=True)
    return df


def process_data_for_network_activity_classification(config):
    df = read_file(config)
    df = pd.concat([add_datetime(df_one_cell) for cell_name,
                    df_one_cell in df.groupby("CellName")]).reset_index(drop=True)
    df = add_target(df)
    df = add_all_features(df, config)
    df = handle_missing_features(df)
    return df

def process_data_for_oss_counters_forecasting(config):
    df = read_file(config)
    oss_counters = literal_eval(config["MODELS"]["OSS_COUNTERS"])
    df_processed = pd.concat([process_oss_counters(cell_name, df_one_cell, oss_counters)
                              for cell_name, df_one_cell in df.groupby("CellName")]).reset_index()
    return df_processed

