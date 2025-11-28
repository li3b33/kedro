# airflow_dags/olympicskedro_dag.py
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "gonzalo",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="olympicskedro_ml_pipeline",
    default_args=default_args,
    description="Pipeline completo de ML: Data Engineering -> Supervisado -> No Supervisado -> Integración",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,  # Ejecución manual
    catchup=False,
    tags=["olympics", "kedro", "ml", "unsupervised"],
) as dag:

    # ============ TASK INICIAL ============
    start_pipeline = DummyOperator(
        task_id="start_pipeline",
        dag=dag,
    )

    # ============ DATA ENGINEERING ============
    run_data_engineering = BashOperator(
        task_id="run_data_engineering",
        bash_command="cd /opt/airflow/project/olympicskedro && kedro run --pipeline=data_engineering",
        dag=dag,
    )

    # ============ PIPELINES SUPERVISADOS (PARALELO) ============
    run_classification = BashOperator(
        task_id="run_classification",
        bash_command="cd /opt/airflow/project/olympicskedro && kedro run --pipeline=classification",
        dag=dag,
    )

    run_regression = BashOperator(
        task_id="run_regression", 
        bash_command="cd /opt/airflow/project/olympicskedro && kedro run --pipeline=regression",
        dag=dag,
    )

    # ============ NO SUPERVISADO ============
    run_unsupervised = BashOperator(
        task_id="run_unsupervised",
        bash_command="cd /opt/airflow/project/olympicskedro && kedro run --pipeline=unsupervised",
        dag=dag,
    )

    # ============ INTEGRACIÓN ============
    run_integration = BashOperator(
        task_id="run_integration",
        bash_command="cd /opt/airflow/project/olympicskedro && kedro run --pipeline=integration", 
        dag=dag,
    )

    # ============ REPORTING ============
    run_reporting = BashOperator(
        task_id="run_reporting",
        bash_command="cd /opt/airflow/project/olympicskedro && kedro run --pipeline=reporting",
        dag=dag,
    )

    run_reporting_unsupervised = BashOperator(
        task_id="run_reporting_unsupervised",
        bash_command="cd /opt/airflow/project/olympicskedro && kedro run --pipeline=reporting_unsupervised",
        dag=dag,
    )

    # ============ TASK FINAL ============
    end_pipeline = DummyOperator(
        task_id="end_pipeline",
        dag=dag,
    )

    # ============ DEPENDENCIAS ============
    start_pipeline >> run_data_engineering

    # Data Engineering alimenta a todos los pipelines
    run_data_engineering >> [run_classification, run_regression, run_unsupervised]

    # Supervisado y No Supervisado en paralelo, luego Integración
    [run_classification, run_regression, run_unsupervised] >> run_integration

    # Integración alimenta el reporting
    run_integration >> [run_reporting, run_reporting_unsupervised]

    # Final del pipeline
    [run_reporting, run_reporting_unsupervised] >> end_pipeline