import os
import sys

from dotenv import dotenv_values
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

config = {
    **dotenv_values(".env.shared"),  # load shared development variables
    **dotenv_values(".env.secret"),  # load sensitive variables
    **os.environ,  # override loaded values with environment variables
}

import main
import ForecastUsage

has_new, bike_usage_df = main.prepare_usage_data()


def check_has_new(**kwargs):
    if has_new > 0:
        return 'train_model_task'
    else:
        return 'skip_train_model_task'


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 8, 30),
    'email_on_failure': True,
    'email_on_retry': True,
    'email': config['MY_EMAIL'],
    'retries': 1,
    'retry_delay': timedelta(minutes=10)
}

daily_dag = DAG(
    'station_data_processing_dag',
    default_args=default_args,
    schedule='0 12 * * *',
    catchup=False
)

weeks_dag = DAG(
    'usage_data_processing_dag',
    default_args=default_args,
    schedule='0 19 10,15,28 * *',
    catchup=False
)

update_station_task = PythonOperator(
    task_id='update_station_task',
    python_callable=main.update_station_data,
    dag=daily_dag
)

prepare_usage_task = PythonOperator(
    task_id='prepare_usage_task',
    python_callable=main.prepare_usage_data,
    dag=weeks_dag
)

check_has_new_task = BranchPythonOperator(
    task_id='check_has_new_task',
    python_callable=check_has_new,
    dag=weeks_dag
)

train_model_task = PythonOperator(
    task_id='train_model_task',
    python_callable=ForecastUsage.train_model,
    op_args=[bike_usage_df],
    dag=weeks_dag
)

skip_train_model_task = EmptyOperator(
    task_id='skip_train_model_task',
    dag=weeks_dag
)

update_station_task
prepare_usage_task >> check_has_new_task >> [train_model_task, skip_train_model_task]
