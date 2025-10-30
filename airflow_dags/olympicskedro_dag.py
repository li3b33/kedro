# airflow_dags/olympicskedro_dag.py
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "gonzalo",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="olympicskedro_pipeline",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,  # o '@daily' si lo quieres agendado
    catchup=False,
) as dag:

    run_data_engineering = BashOperator(
        task_id="run_data_engineering",
        bash_command="cd /opt/airflow/project/olympicskedro && kedro run --pipeline=data_engineering",
    )


    run_classification = BashOperator(
        task_id="run_classification",
        bash_command="cd /opt/airflow/project/olympicskedro && kedro run --pipeline=data_engineering",
    )

    run_regression = BashOperator(
        task_id="run_regression",
        bash_command="cd /opt/airflow/project/olympicskedro && kedro run --pipeline=data_engineering",
    )

    run_data_engineering >> run_classification >> run_regression
