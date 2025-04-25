#import joblib
from data_processing import process_data_for_oss_counters_forecasting
from utils import get_config, parse_args,save_model
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval


def forecast_oss_counter_per_cell(df, oss_counters, datetime="datetime", cell_col="CellName"):
    list_metrics = []
    list_models = []
    for oss_counter in oss_counters:
        for cell, group in df.groupby(cell_col):
            group = group[[datetime, cell_col, oss_counter]].dropna().sort_values(datetime)
            prophet_df = group.rename(columns={datetime: "ds", oss_counter: "y"})

            train_df = prophet_df[['ds', 'y']].dropna().sort_values('ds').iloc[:-96*4].reset_index(drop=True)
            val_df = prophet_df[['ds', 'y']].dropna().sort_values('ds').iloc[-96*4:].reset_index(drop=True)

            try:
                model = Prophet(daily_seasonality=True)
                model.fit(train_df)
                future_df = pd.DataFrame({'ds': val_df["ds"].values})
                forecast = model.predict(future_df)
                y_pred = forecast["yhat"].values
                y_true = val_df["y"].values

                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mape = np.mean(np.abs((y_true-y_pred)/y_true))*100

                list_metrics.append(
                    {"CellName": cell,
                    "oss_counter": oss_counter,
                    "MAE": mae,
                    "RMSE": rmse,
                    "MAPE": mape})

                model_name = "_".join(("prophet", cell, oss_counter))
                save_model(model, f"./src/oss_counters_forecasting_models/{model_name}.pkl")
                print("model saved")

                list_models.append(model)
            #
            #     # Visualization
            #     plt.figure(figsize=(20, 4))
            #     plt.plot(train_df["ds"], train_df["y"], label="Train")
            #     plt.plot(val_df["ds"], val_df["y"], label="Actual")
            #     plt.plot(val_df["ds"], y_pred, label="Forecast")
            #     plt.title(f" Cell: {cell}, oss_counter: {oss_counter}")
            #     plt.xlabel("Time")
            #     plt.ylabel(oss_counter)
            #     plt.legend()
            #     plt.grid(True)
            #     plt.tight_layout()
            #     plt.show()
            #
            except Exception as e:
                 print(f"Skipping {cell}-{oss_counter}")

    df_metrics = pd.DataFrame(list_metrics)
    df_metrics.replace(to_replace=np.inf, value=np.nan, inplace=True)
    df_metrics = df_metrics.groupby("oss_counter").agg(
        {"RMSE": "mean", "MAE": "mean", "MAPE": "mean"}).reset_index()
    return df_metrics


def main(args):
    config = get_config(args.configuration)
    df_processed = process_data_for_oss_counters_forecasting(config)
    df_selected = df_processed[df_processed["CellName"].isin(['10ALTE', '10BLTE'])]
    df_metrics = forecast_oss_counter_per_cell(df_selected, oss_counters=literal_eval(config["MODELS"]["OSS_COUNTERS"]))
    print(df_metrics)


if __name__ == "__main__":
    args = parse_args()
    main(args)



