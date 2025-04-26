FROM python:3.8-slim

RUN mkdir /networkapp
WORKDIR /networkapp

COPY requirements.txt /networkapp
COPY configuration.ini /networkapp
COPY ./src /networkapp
COPY ./models /networkapp
COPY ./experimental /networkapp
COPY ./data /networkapp
COPY Readme.md /networkapp

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "src/oss_counters_forecasting.py", "--configuration", "configuration.ini"]
CMD ["python", "src/network_activity_classification.py", "--configuration", "configuration.ini"]
CMD ["python", "src/gradio_app.py"]