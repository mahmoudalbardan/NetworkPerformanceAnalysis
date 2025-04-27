import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
from utils import get_config, parse_args, save_model
from data_processing import process_data_for_oss_counters_forecasting


def forecast_oss_counter_per_cell(df, oss_counters, datetime="datetime", cell_col="CellName"):
    """
    Forecast OSS counters per cell using Prophet, evaluate model (MAE, RMSE, MAPE)
    and save the results.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe containing time series data per cell and OSS counter.
    oss_counters : list of str
        List of OSS counter names to forecast.
    datetime : str, optional
        Datetime column, by default "datetime".
    cell_col : str, optional
        Column name representing the cell name, by default "CellName".
    """
    list_metrics = []
    list_models = []

    for oss_counter in oss_counters:
        for cell, group in df.groupby(cell_col):
            group = group[[datetime, cell_col, oss_counter]].dropna().sort_values(datetime)
            prophet_df = group.rename(columns={datetime: "ds", oss_counter: "y"})

            # Train/Validation split: last 24h = 96*4 steps for 15-min intervals
            train_df = prophet_df[["ds", "y"]].dropna().sort_values("ds").iloc[:-96 * 4].reset_index(drop=True)
            val_df = prophet_df[["ds", "y"]].dropna().sort_values("ds").iloc[-96 * 4:].reset_index(drop=True)

            try:
                model = Prophet(daily_seasonality=True)
                model.fit(train_df)

                future_df = pd.DataFrame({"ds": val_df["ds"].values})
                forecast = model.predict(future_df)

                y_pred = forecast["yhat"].values
                y_true = val_df["y"].values

                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

                list_metrics.append({
                    "CellName": cell,
                    "oss_counter": oss_counter,
                    "MAE": mae,
                    "RMSE": rmse,
                    "MAPE": mape
                })

                save_model(
                    model,
                    f"./models/oss_counters_forecasting_models/{'_'.join(("prophet", cell, oss_counter))}.pkl"
                )

                list_models.append(model)
            except:
                print(f"Skipping {cell}-{oss_counter}")

    df_metrics = pd.DataFrame(list_metrics)
    df_metrics.replace(to_replace=np.inf, value=np.nan, inplace=True)
    df_metrics.to_csv("./models/results/oss_counters_forecasting_metrics.csv", sep=",", index=False)


def main(args):
    """
    main function.
    """
    config = get_config(args.configuration)
    df_processed = process_data_for_oss_counters_forecasting(config)
    forecast_oss_counter_per_cell(df_processed,
        oss_counters=literal_eval(config["MODELS"]["OSS_COUNTERS"]))

if __name__ == "__main__":
    args = parse_args()
    main(args)
