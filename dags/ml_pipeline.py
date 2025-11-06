from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.models import Variable

import pandas as pd
import numpy as np

from scripts.setup_stores import create_datamart, create_bronze_store, create_silver_store, create_gold_store, create_model_bank
from scripts.data_processing_bronze import data_processing_bronze
from scripts.data_processing_silver import data_processing_silver
from scripts.data_processing_gold import data_processing_gold_label
from scripts.data_processing_gold import data_processing_gold_feature
from scripts.data_processing_gold import data_post_processing
from scripts.model_training import train_encoders, training_preprocessing, train_logistic_regression, train_xgb
from scripts.deployment_inference import create_outbox, deploy_model, inference_preprocessing, infer
from scripts.monitoring import create_monitoring_directory, results_monitoring, drift_monitoring, psi_monitoring

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=1)
}
training_start_date = datetime(2023, 9, 1)
inference_start_date = datetime(2023, 9, 1)
monitoring_start_date = datetime(2023, 10, 1)

# Initialize the DAG
ml_pipeline = DAG(
    dag_id='ml_pipeline',
    default_args=default_args,
    description='End-to-end Machine Learning Pipeline - MLE Assignment 2',
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    schedule='@monthly',  # Run monthly
    catchup=True
)

silver_prefixes = {
    "silver_lms_prefix": "silver_lms_",
    "silver_cs_prefix": "silver_cs_",    
    "silver_att_prefix": "silver_att_",
    "silver_fin_prefix": "silver_fin_"
}

start_data_pipeline = EmptyOperator(task_id='start_data_pipeline', dag=ml_pipeline)
create_datamart = PythonOperator(
    task_id='create_datamart',
    python_callable=create_datamart,
    dag=ml_pipeline
)
create_bronze_store = PythonOperator(
    task_id='create_bronze_store',
    python_callable=create_bronze_store,
    dag=ml_pipeline
)
create_silver_store = PythonOperator(
    task_id='create_silver_store',
    python_callable=create_silver_store,
    dag=ml_pipeline
)
create_gold_store = PythonOperator(
    task_id='create_gold_store',
    python_callable=create_gold_store,
    dag=ml_pipeline
)
data_processing_bronze = PythonOperator(
    task_id='data_processing_bronze',
    python_callable=data_processing_bronze,
    dag=ml_pipeline
)
data_processing_silver = PythonOperator(
    task_id='data_processing_silver',
    python_callable=data_processing_silver,
    depends_on_past=True,
    dag=ml_pipeline
)
data_processing_gold_label = PythonOperator(
    task_id='data_processing_gold_label',
    python_callable=data_processing_gold_label,
    dag=ml_pipeline
)
data_processing_gold_feature = PythonOperator(
    task_id='data_processing_gold_feature',
    python_callable=data_processing_gold_feature,
    dag=ml_pipeline
)
end_data_pipeline = EmptyOperator(task_id='end_data_pipeline', dag=ml_pipeline)

start_training_pipeline = EmptyOperator(task_id='start_training_pipeline', start_date=training_start_date, dag=ml_pipeline)
create_model_bank = PythonOperator(
    task_id='create_model_bank',
    python_callable=create_model_bank,
    start_date=training_start_date,
    dag=ml_pipeline
)
data_post_processing = PythonOperator(
    task_id='data_post_processing',
    python_callable=data_post_processing,
    start_date=training_start_date,
    dag=ml_pipeline
)
train_encoders = PythonOperator(
    task_id='train_encoders',
    python_callable=train_encoders,
    start_date=training_start_date,
    dag=ml_pipeline
)
training_preprocessing = PythonOperator(
    task_id='training_preprocessing',
    python_callable=training_preprocessing,
    start_date=training_start_date,
    dag=ml_pipeline
)
train_logistic_regression = PythonOperator(
    task_id='train_logistic_regression',
    python_callable=train_logistic_regression,
    start_date=training_start_date,
    dag=ml_pipeline
)
train_xgb = PythonOperator(
    task_id='train_xgb',
    python_callable=train_xgb,
    start_date=training_start_date,
    depends_on_past=True, # Not strictly required, but appears to run more quickly when only one XGB is running at any time.
    dag=ml_pipeline
)
end_training_pipeline = EmptyOperator(task_id='end_training_pipeline', start_date=training_start_date, dag=ml_pipeline)

start_inference_pipeline = EmptyOperator(task_id='start_inference_pipeline', start_date = inference_start_date, dag=ml_pipeline)
create_outbox = PythonOperator(
    task_id='create_outbox',
    python_callable=create_outbox,
    start_date=inference_start_date,
    dag=ml_pipeline
)
deploy_model = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    start_date=inference_start_date,
    dag=ml_pipeline
)
inference_preprocessing = PythonOperator(
    task_id='inference_preprocessing',
    python_callable=inference_preprocessing,
    start_date=inference_start_date,
    dag=ml_pipeline
)
infer = PythonOperator(
    task_id='infer',
    python_callable=infer,
    start_date=inference_start_date,
    dag=ml_pipeline
)
end_inference_pipeline = EmptyOperator(task_id='end_inference_pipeline', start_date = inference_start_date, dag=ml_pipeline)

start_monitoring_pipeline = EmptyOperator(task_id='start_monitoring_pipeline', start_date = inference_start_date, dag=ml_pipeline)
create_monitoring_directory = PythonOperator(
    task_id='create_monitoring_directory',
    python_callable=create_monitoring_directory,
    start_date=inference_start_date,
    dag=ml_pipeline
)
results_monitoring = PythonOperator(
    task_id='results_monitoring',
    python_callable=results_monitoring,
    start_date=inference_start_date,
    depends_on_past=True,
    dag=ml_pipeline
)
psi_monitoring = PythonOperator(
    task_id='psi_monitoring',
    python_callable=psi_monitoring,
    start_date=inference_start_date,
    dag=ml_pipeline
)
drift_monitoring = PythonOperator(
    task_id='drift_monitoring',
    python_callable=drift_monitoring,
    start_date=monitoring_start_date,
    depends_on_past=True,
    dag=ml_pipeline
)
end_monitoring_pipeline = EmptyOperator(task_id='end_monitoring_pipeline', start_date = inference_start_date, dag=ml_pipeline)


# Task dependencies
start_data_pipeline >> create_datamart >> create_bronze_store >> data_processing_bronze >> data_processing_silver
create_datamart >> create_silver_store >> data_processing_silver >> data_processing_gold_label
create_datamart >> create_silver_store >> data_processing_silver >> data_processing_gold_feature
create_datamart >> create_gold_store >> data_processing_gold_label >> end_data_pipeline
create_datamart >> create_gold_store >> data_processing_gold_feature >> end_data_pipeline

end_data_pipeline >> start_training_pipeline
start_training_pipeline >> create_model_bank >> data_post_processing >> train_encoders >> training_preprocessing
training_preprocessing >> train_logistic_regression >> end_training_pipeline
training_preprocessing >> train_xgb >> end_training_pipeline

end_training_pipeline >> start_inference_pipeline
start_inference_pipeline >> create_outbox >> deploy_model >> inference_preprocessing >>  infer
infer >> end_inference_pipeline

end_inference_pipeline >> create_monitoring_directory >> start_monitoring_pipeline
start_monitoring_pipeline >> results_monitoring >> end_monitoring_pipeline
start_monitoring_pipeline >> psi_monitoring >> end_monitoring_pipeline
start_monitoring_pipeline >> drift_monitoring >> end_monitoring_pipeline


