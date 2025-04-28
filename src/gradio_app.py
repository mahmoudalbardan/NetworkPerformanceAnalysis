import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import gradio as gr
from utils import load_model
from data_processing import add_datetime, add_rolling_features, add_ratio_features


cell_names = ["3BLTE", "1BLTE", "9BLTE", "4ALTE", "10BLTE", "9ALTE", "4BLTE",
              "4CLTE", "6CLTE", "5CLTE", "7BLTE", "8CLTE", "7ULTE", "6WLTE",
              "7VLTE", "7WLTE", "5ALTE", "6ALTE", "6ULTE", "3CLTE", "5BLTE",
              "8ALTE", "8BLTE", "6BLTE", "10CLTE", "7CLTE", "3ALTE", "1CLTE",
              "2ALTE", "10ALTE", "1ALTE", "6VLTE", "7ALTE"]

oss_counters = ["PRBUsageUL", "PRBUsageDL", "meanThr_DL", "meanThr_UL",
                "maxThr_DL", "maxThr_UL", "meanUE_DL", "meanUE_UL", "maxUE_DL",
                "maxUE_UL", "maxUE_UL+DL"]

def prepare_data_for_net_activity(file_test):
    df = pd.read_csv(file_test.name,sep=";")
    df = pd.concat([add_datetime(group) for _, group in df.groupby("CellName")]).reset_index(drop=True)
    window_sizes = [1, 5, 7, 15]
    for oss_counter in oss_counters:
        for ws in window_sizes:
            df = add_rolling_features(df, oss_counter, ws)
    df["hour"] = df["datetime"].dt.hour
    df = add_ratio_features(df)
    df.dropna(how="any", inplace=True)
    df.rename(columns={"Unusual": "target"}, inplace=True)
    return df


def predict_network_activity(file_test,cell_name):
    df_metrics = pd.read_csv("./models/results/network_activity_classifier_metrics.csv", sep=",")

    model = load_model("./models/network_activity_classifier.pkl")
    df = prepare_data_for_net_activity(file_test)
    df_to_predict = df.drop(columns=["CellName", "datetime"])
    features = df_to_predict.columns
    feature_importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({"Feature": features, "Importance": feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

    # fig feature importance
    fig_feature_importance = plt.figure(figsize=(10, 4))
    ax_feature_importance = fig_feature_importance.add_subplot(111)
    ax_feature_importance.barh(feature_importance_df["Feature"].iloc[:7], feature_importance_df["Importance"].iloc[:7], color='r')
    ax_feature_importance.set_xlabel("Feature Importance")
    ax_feature_importance.set_title("Feature Importance")


    df["prediction"] = pd.DataFrame(model.predict(df_to_predict))
    y = df.loc[df["CellName"] == cell_name].copy()
    y.sort_values(by="datetime", inplace=True)
    fig_prediction = plt.figure(figsize=(15, 6))
    ax = fig_prediction.add_subplot(111)
    ax.stem(y["datetime"].iloc[:96], y["prediction"].iloc[:96], linefmt='b-', markerfmt='bo', basefmt='r-')
    dates_with_1 = y["datetime"].iloc[:96][y["prediction"].iloc[:96] == 1]
    ax.set_xticks(dates_with_1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Usual", "Unusual"])
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Label")
    ax.set_title(f"Activity of cell {cell_name}")
    plt.xticks(rotation=35)
    img_metrics_path = "./models/results/network_activity_classifier_metrics.png"
    return img_metrics_path , df_metrics.round(3), fig_feature_importance, fig_prediction


def load_model_forecasting(cell_name, oss_counter):
    return load_model(f"models/oss_counters_forecasting_models/{"_".join(("prophet", cell_name, oss_counter))}.pkl")


def oss_counter_forecasting(cell_name, oss_counter, nb_points):
    model = load_model_forecasting(cell_name, oss_counter)
    historical_data = model.history
    future = pd.date_range(start=historical_data["ds"].max(), periods=int(nb_points)*96, freq="15min")
    future_df = pd.DataFrame({"ds": future})
    forecast = model.predict(future_df)
    df_metrics = pd.read_csv("./models/results/oss_counters_forecasting_metrics.csv",sep=",")
    df_metrics_agg = df_metrics.groupby("oss_counter").agg(
       {"RMSE": "mean", "MAE": "mean", "MAPE": "mean"}
    ).reset_index()
    df_metrics_cell = df_metrics[(df_metrics["CellName"] == cell_name)&
                                 (df_metrics["oss_counter"] == oss_counter)][["CellName","oss_counter","MAE","RMSE","MAPE"]]
    df_metrics_cell.columns = ["Cell Name", "OSS Counter", "MAE", "RMSE", "MAPE"]
    df_metrics_oss_counter = df_metrics_agg[["oss_counter","MAE","RMSE","MAPE"]]
    df_metrics_oss_counter.columns = ["OSS Counter", "MAE", "RMSE", "MAPE"]
    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_subplot(111)
    ax.plot(historical_data["ds"], historical_data["y"], label="Historical Data", color="k")
    ax.plot(forecast["ds"], forecast["yhat"], label="Forecast", color="b", linestyle="dashed")

    ax.set_title(f"forecast and historical: {cell_name} - {oss_counter}")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Value")
    ax.legend()

    return fig, df_metrics_cell.round(3), df_metrics_oss_counter.round(3), forecast[["ds", "yhat"]]


oss_counter_forecasting_interface = gr.Interface(
    fn=oss_counter_forecasting,
    inputs=[
        gr.Radio(cell_names, label="Cell Name"),
        gr.Radio(oss_counters, label="Oss Counter"),
        gr.Slider(minimum=1, maximum=10, step=1, label="Days to Forecast")
    ],
    outputs=[
        gr.Plot(label="Forecast plot", format="png"),
        gr.Dataframe(label="Model performance for this specific cell/oss couple"),
        gr.Dataframe(label="Model performance for OSS counters accross all cells"),
        gr.Dataframe(label="Forecast data")
    ],
    title="OSS counter forecasting")

predict_network_activity_interface = gr.Interface(
    fn=predict_network_activity,
    inputs = [gr.File(label="Upload test file"),gr.Radio(cell_names, label="Cell Name") ],
    outputs=[gr.Image(label="Model Performances", format="png"),
             gr.Dataframe(label="Model Performances accross all folds for the best probability prediction threshold"),
             gr.Plot(label="Feature importances", format="png"),
             gr.Plot(label="Model Prediction", format="png")],
    title="Cell activity prediction")


gradio_interface = gr.TabbedInterface([oss_counter_forecasting_interface, predict_network_activity_interface],
                                      tab_names=["OSS counter forecasting", "Cell activity prediction"])


if __name__ == "__main__":
    gradio_interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
