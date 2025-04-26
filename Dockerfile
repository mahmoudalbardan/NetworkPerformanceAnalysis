FROM python:3.12-slim

RUN mkdir /networkapp
WORKDIR /networkapp

COPY requirements.txt /networkapp
COPY configuration.ini /networkapp
COPY ./src /networkapp/src
COPY ./models /networkapp/models
COPY ./experimental /networkapp/experimental
COPY ./data /networkapp/data

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
CMD ["python", "src/gradio_app.py"]