#!/bin/bash
python src/oss_counters_forecasting.py --configuration configuration.ini
python src/network_activity_classification.py --configuration configuration.ini
python src/gradio_app.py