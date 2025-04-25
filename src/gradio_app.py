import pickle
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt

cell_names = [ '3BLTE', '1BLTE', '9BLTE', '4ALTE', '10BLTE', '9ALTE', '4BLTE',
               '4CLTE', '6CLTE', '5CLTE', '7BLTE', '8CLTE', '7ULTE', '6WLTE',
               '7VLTE', '7WLTE', '5ALTE', '6ALTE', '6ULTE', '3CLTE', '5BLTE',
               '8ALTE', '8BLTE', '6BLTE', '10CLTE', '7CLTE', '3ALTE', '1CLTE',
               '2ALTE', '10ALTE', '1ALTE', '6VLTE', '7ALTE']
oss_counters = ['PRBUsageUL', 'PRBUsageDL', 'meanThr_DL', 'meanThr_UL',
               'maxThr_DL', 'maxThr_UL', 'meanUE_DL', 'meanUE_UL', 'maxUE_DL',
               'maxUE_UL', 'maxUE_UL+DL']

def load_model(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def load_model_forecasting(cell_name, oss_counter):
    model_name = "_".join(("prophet", cell_name, oss_counter))
    model = load_model(f"C:/Users/mahmo/Documents/Ooredoo Technical test/src/oss_counters_forecasting_models/{model_name}.pkl")
    return model

def predict(cell_name, oss_counter, nb_points):
    model = load_model_forecasting(cell_name, oss_counter)
    historical_data = model.history
    future = pd.date_range(start=historical_data["ds"].max(), periods=int(nb_points)*96, freq="15min")
    future_df = pd.DataFrame({"ds": future})
    forecast = model.predict(future_df)
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(historical_data["ds"], historical_data["y"], label="Historical Data", color="k")
    ax.plot(forecast["ds"], forecast["yhat"], label="Forecast", color="b",linestyle="dashed")

    ax.set_title(f"forecast and historical: {cell_name} - {oss_counter}")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Value")
    ax.legend()
    return fig, forecast[["ds","yhat"]]


app = gr.Interface(
    fn=predict,
    inputs=[gr.Radio(cell_names, label="Cell Name"),
            gr.Radio(oss_counters, label="Oss Counter"),
            gr.Slider(minimum=1, maximum=240, step=1, label="Days to Forecast")],
    outputs= [gr.Plot(label="Forecast plot", format="png"),gr.Dataframe(label="Forecast data")],
    title="OSS counter forecasting")

if __name__ == "__main__":
    app.launch(share=True)