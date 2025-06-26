from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from inference import get_prediction_report

default_args = {
    'owner': 'btc_rl_bot',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

dag = DAG(
    'btc_transformer_inference',
    default_args=default_args,
    start_date=datetime(2025, 6, 26),
    schedule_interval='16-02/15 * * * *',  # Каждые 15 минут с 1-минутной задержкой
    catchup=False,
)

run_inference = PythonOperator(
    task_id='run_btc_inference',
    python_callable=get_prediction_report,
    dag=dag,
)
