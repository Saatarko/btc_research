FROM apache/airflow:2.8.1-python3.10

USER root
RUN apt-get update && apt-get install -y gcc g++ libgl1-mesa-glx libglib2.0-0

COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip && pip install -r /requirements.txt

COPY dags/ /opt/airflow/dags/
COPY inference.py /opt/airflow/
COPY models/ /opt/airflow/models/
COPY logs/ /opt/airflow/logs/