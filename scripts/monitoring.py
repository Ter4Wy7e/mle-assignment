import os
import logging
import joblib
import pendulum

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics

from airflow.models import XCom

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T

from scripts.helpers import PSIModelMonitor

# Create Logging Directory
if not os.path.exists("/app/logs"):
    os.makedirs("/app/logs")

# Logger
logger = logging.getLogger('monitoring_pipeline')  # Set the logger name
handler = logging.FileHandler('/app/logs/monitoring_pipeline.log')
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Monitoring directory
monitoring_directory = "/app/monitoring"

def create_monitoring_directory(ti, **context):
    current_date = context['ds']

    if not os.path.exists(monitoring_directory):
        os.makedirs(monitoring_directory)
        logger.info(f"[{ti.task_id} | {current_date}] Created monitoring directory: {monitoring_directory}")
    else:
        logger.info(f"[{ti.task_id} | {current_date}] Monitoring directory already exists: {monitoring_directory}")
    ti.xcom_push(key='monitoring_directory', value=monitoring_directory)


def results_monitoring(ti, **context):

    results_monitoring_records_csv = 'results_monitoring_records.csv'
    results_monitoring_records_joblib = 'results_monitoring_records.joblib'

    monitoring_directory = ti.xcom_pull(task_ids='create_monitoring_directory', key='monitoring_directory')
    current_date = context['ds']

    spark = SparkSession.builder \
        .master("local[*]") \
        .config("spark.executor.memory", "8g") \
        .config("spark.executor.memoryOverhead", "1g") \
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC") \
        .getOrCreate()
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
    spark.sparkContext.setLogLevel("ERROR")
    spark

    # Get current label records
    gold_label_filepath = ti.xcom_pull(task_ids='data_processing_gold_label', key='gold_label_filepath')
    df_labels = spark.read.parquet(gold_label_filepath)
    label_1 = df_labels.filter(F.col("label") == 1).count()
    label_0 = df_labels.filter(F.col("label") == 0).count()
    logger.info(f"[{ti.task_id} | {current_date}] Labelled results: 1 [{label_1}, {label_1/df_labels.count()*100:.2f}%], 0 [{label_0}, {label_0/df_labels.count()*100:.2f}%].")

    # Get results
    results_filepath = ti.xcom_pull(task_ids='infer', key='inference_results_filepath')
    df_results = spark.read.parquet(results_filepath)
    result_1 = df_results.filter(F.col("prediction") == 1).count()
    result_0 = df_results.filter(F.col("prediction") == 0).count()
    logger.info(f"[{ti.task_id} | {current_date}] Inference results: 1 [{result_1}, {result_1/df_results.count()*100:.2f}%], 0 [{result_0}, {result_0/df_results.count()*100:.2f}%].")

    # Alert if variation of >30% of results is observed.
    if abs((label_1/df_labels.count() - result_1/df_results.count())*100) > 40:
        logger.warning(f"[{ti.task_id} | {current_date}] Inference results are different than gold labels by more than 40%.")
    
    # Save file record
    record = {
        'run_date': [current_date],
        'label_count': [df_labels.count()],
        'label_1' : [label_1],
        'label_0': [label_0],
        'result_count': [df_results.count()],
        'result_1': [result_1],
        'result_0': [result_0]
    }

    df_record = pd.DataFrame(record)

    record_csv_filepath = os.path.join(monitoring_directory, results_monitoring_records_csv)
    record_joblib_filepath = os.path.join(monitoring_directory, results_monitoring_records_joblib)

    if os.path.exists(record_joblib_filepath):
        results_monitoring_records = joblib.load(record_joblib_filepath)
        df_record = pd.concat([results_monitoring_records, df_record], axis=0)
    joblib.dump(df_record, record_joblib_filepath)
    df_record.to_csv(record_csv_filepath, index=False)
    logger.info(f"[{ti.task_id} | {current_date}] Results anomaly monitoring records (joblib) updated at: {record_joblib_filepath}.")
    logger.info(f"[{ti.task_id} | {current_date}] Results anomaly monitoring records (CSV) updated at: {record_csv_filepath}.")

    df_record['labels_percent'] = df_record['label_1'] / df_record['label_count']
    df_record['results_percent'] = df_record['result_1'] / df_record['result_count']

    results_png_filepath = os.path.join(monitoring_directory, 'results_monitoring_' + current_date + '.png')
    plt.plot(df_record['run_date'], df_record['results_percent'], label="Result")
    plt.plot(df_record['run_date'], df_record['labels_percent'], label="Label")
    plt.xlabel('Date')
    plt.ylabel("Percentage '1' Results")
    plt.title('Sudden Result Drift Monitoring')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    plt.savefig(results_png_filepath)  # Save the image
    plt.close()



def drift_monitoring(ti, **context):

    drift_monitoring_records_csv = 'drift_monitoring_records.csv'
    drift_monitoring_records_joblib = 'drift_monitoring_records.joblib'

    drift_monitoring_pred_v_labels = 'drift_pred_v_labels_'

    monitoring_directory = ti.xcom_pull(task_ids='create_monitoring_directory', key='monitoring_directory')
    current_date = context['ds']

    spark = SparkSession.builder \
        .master("local[*]") \
        .config("spark.executor.memory", "8g") \
        .config("spark.executor.memoryOverhead", "1g") \
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC") \
        .getOrCreate()
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
    spark.sparkContext.setLogLevel("ERROR")
    spark

    # Get current label records
    gold_label_filepath = ti.xcom_pull(task_ids='data_processing_gold_label', key='gold_label_filepath')
    df_labels = spark.read.parquet(gold_label_filepath)
    label_1 = df_labels.filter(F.col("label") == 1).count()
    label_0 = df_labels.filter(F.col("label") == 0).count()
    logger.info(f"[{ti.task_id} | {current_date}] Labelled results: 1 [{label_1}, {label_1/df_labels.count()*100:.2f}%], 0 [{label_0}, {label_0/df_labels.count()*100:.2f}%].")

    # Get results
    results_prev_filepath = XCom.get_one(execution_date=pendulum.parse(current_date).subtract(months=1), task_id="infer", key="inference_results_filepath")
    if results_prev_filepath:
        df_results = spark.read.parquet(results_prev_filepath)
        logger.info(f"Loaded predictions from previous run with {df_results.count()} rows: {results_prev_filepath}")
    else:
        logger.warning(f"[{ti.task_id} | {current_date}] No previous predictions found. Is this the first run?")

    # Raw results
    df_merged = df_labels.join(df_results, on=["customer_id", "loan_start_date"], how="inner")
    pred_v_label_filename = drift_monitoring_pred_v_labels + current_date + '.csv'
    pred_v_label_filepath = os.path.join(monitoring_directory, pred_v_label_filename)
    df_merged.write.mode('overwrite').csv(pred_v_label_filepath, header=True)
    ti.xcom_push(key='drift_monitoring_pred_v_labels', value=pred_v_label_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Predictions vs Labels CSV saved at: {pred_v_label_filepath}")

    df_merged_pd = df_merged.toPandas()

    recall_score = metrics.recall_score(df_merged_pd['label'].values.ravel(), df_merged_pd['prediction'].values.ravel())
    f1_score = metrics.f1_score(df_merged_pd['label'].values.ravel(), df_merged_pd['prediction'].values.ravel())
    auc_score = metrics.roc_auc_score(df_merged_pd['label'].values.ravel(), df_merged_pd['prediction'].values.ravel())
    gini_score = 2*auc_score-1

    # Retrieve drift thresholds and alert if exceeded.
    recall_threshold = ti.xcom_pull(task_ids='deploy_model', key='active_threshold_p0')
    f1_threshold = ti.xcom_pull(task_ids='deploy_model', key='active_threshold_p1')
    gini_threshold = ti.xcom_pull(task_ids='deploy_model', key='active_threshold_p2')

    if recall_score < recall_threshold:
        logger.warning(f"[{ti.task_id} | {current_date}] Recall {recall_score} is below threshold {recall_threshold}.")
    else:
        logger.info(f"[{ti.task_id} | {current_date}] Recall: {recall_score}. Threshold: {recall_threshold}.")

    if f1_score < f1_threshold:
        logger.warning(f"[{ti.task_id} | {current_date}] Recall {f1_score} is below threshold {f1_threshold}.")
    else:
        logger.info(f"[{ti.task_id} | {current_date}] Recall: {f1_score}. Threshold: {f1_threshold}.")

    if gini_score < gini_threshold:
        logger.warning(f"[{ti.task_id} | {current_date}] Recall {gini_score} is below threshold {gini_threshold}.")
    else:
        logger.info(f"[{ti.task_id} | {current_date}] Recall: {gini_score}. Threshold: {gini_threshold}.")
    
    # Save file record

    active_model = ti.xcom_pull(task_ids='deploy_model', key='active_model')
    active_model_date = ti.xcom_pull(task_ids='deploy_model', key='active_model_date')
    active_model_version = ti.xcom_pull(task_ids='deploy_model', key='active_model_version')
    active_metric_p0 = ti.xcom_pull(task_ids='deploy_model', key='active_metric_p0')
    active_threshold_p0 = ti.xcom_pull(task_ids='deploy_model', key='active_threshold_p0')
    active_metric_p1 = ti.xcom_pull(task_ids='deploy_model', key='active_metric_p1')
    active_threshold_p1 = ti.xcom_pull(task_ids='deploy_model', key='active_threshold_p1')
    active_metric_p2 = ti.xcom_pull(task_ids='deploy_model', key='active_metric_p2')
    active_threshold_p2 = ti.xcom_pull(task_ids='deploy_model', key='active_threshold_p2')

    record = {
        'run_date': [current_date],
        'active_model': [active_model],
        'active_model_date': [active_model_date],
        'active_model_version': [active_model_version],
        'active_metric_p0': [active_metric_p0],
        'active_threshold_p0': [active_threshold_p0],
        'active_metric_p1': [active_metric_p1],
        'active_threshold_p1': [active_threshold_p1],
        'active_metric_p2': [active_metric_p2],
        'active_threshold_p2': [active_threshold_p2],
        'recall_score': [recall_score],
        'f1_score': [f1_score],
        'auc_score': [auc_score],
        'gini_score': [gini_score]
    }

    df_record = pd.DataFrame(record)

    drift_monitoring_csv_filepath = os.path.join(monitoring_directory, drift_monitoring_records_csv)
    drift_monitoring_joblib_filepath = os.path.join(monitoring_directory, drift_monitoring_records_joblib)

    if os.path.exists(drift_monitoring_joblib_filepath):
        results_monitoring_records = joblib.load(drift_monitoring_joblib_filepath)
        df_record = pd.concat([results_monitoring_records, df_record], axis=0)
    joblib.dump(df_record, drift_monitoring_joblib_filepath)

    df_record.to_csv(drift_monitoring_csv_filepath, index=False)
    logger.info(f"[{ti.task_id} | {current_date}] Drift monitoring records (joblib) updated at: {drift_monitoring_joblib_filepath}.")
    logger.info(f"[{ti.task_id} | {current_date}] Drift monitoring records (csv) updated at: {drift_monitoring_csv_filepath}.")

    drift_png_filepath = os.path.join(monitoring_directory, 'drift_monitoring_' + current_date + '.png')

    plt.figure(figsize=(24, 16))

    plt.plot(df_record['run_date'], df_record['recall_score'], label="Recall")
    plt.plot(df_record['run_date'], df_record['f1_score'], label="F1")
    plt.plot(df_record['run_date'], df_record['gini_score'], label="GINI")
    
    plt.xlabel('Date')
    plt.ylabel("Perforamnce Score")
    plt.title('Concept Drift Monitoring')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(drift_png_filepath)  # Save the image
    plt.close()




def psi_monitoring(ti, **context):

    spark = SparkSession.builder \
        .master("local[*]") \
        .config("spark.executor.memory", "8g") \
        .config("spark.executor.memoryOverhead", "1g") \
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC") \
        .getOrCreate()
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
    spark.sparkContext.setLogLevel("ERROR")
    spark

    current_date = context['ds']
    monitoring_directory = ti.xcom_pull(task_ids='create_monitoring_directory', key='monitoring_directory')
    psi_monitoring_records_txt = 'psi_monitoring_records_' + current_date + '.txt'
    psi_monitoring_records_fig = 'psi_monitoring_records_' + current_date + '.png'
    filepath_txt = os.path.join(monitoring_directory, psi_monitoring_records_txt)
    filepath_fig = os.path.join(monitoring_directory, psi_monitoring_records_fig)

    if pendulum.parse(current_date) >= pendulum.datetime(2024, 8, 1):
        # xgb_clf, 2024-08-01, 30dpd_6mob
        training_filepath_X = XCom.get_one(execution_date=pendulum.datetime(2024,8,1), task_id="training_preprocessing", key="gold_training_view_X_filepath")
        validation_filepath_X = XCom.get_one(execution_date=pendulum.datetime(2024,8,1), task_id="training_preprocessing", key="gold_validation_view_X_filepath")
        df_train_X = pd.read_parquet(training_filepath_X)
        df_val_X = pd.read_parquet(validation_filepath_X)
        df_train = pd.concat([df_train_X, df_val_X], axis=0, ignore_index=True)

    elif pendulum.parse(current_date) >= pendulum.datetime(2024, 2, 1):
        # xgb_clf, 2024-02-01, 30dpd_6mob
        training_filepath_X = XCom.get_one(execution_date=pendulum.datetime(2024,2,1), task_id="training_preprocessing", key="gold_training_view_X_filepath")
        validation_filepath_X = XCom.get_one(execution_date=pendulum.datetime(2024,2,1), task_id="training_preprocessing", key="gold_validation_view_X_filepath")
        df_train_X = pd.read_parquet(training_filepath_X)
        df_val_X = pd.read_parquet(validation_filepath_X)
        df_train = pd.concat([df_train_X, df_val_X], axis=0, ignore_index=True)

    elif pendulum.parse(current_date) >= pendulum.datetime(2023, 12, 1):
        # xgb_clf, 2023-12-01, 30dpd_6mob
        training_filepath_X = XCom.get_one(execution_date=pendulum.datetime(2023,12,1), task_id="training_preprocessing", key="gold_training_view_X_filepath")
        validation_filepath_X = XCom.get_one(execution_date=pendulum.datetime(2023,12,1), task_id="training_preprocessing", key="gold_validation_view_X_filepath")
        df_train_X = pd.read_parquet(training_filepath_X)
        df_val_X = pd.read_parquet(validation_filepath_X)
        df_train = pd.concat([df_train_X, df_val_X], axis=0, ignore_index=True)


    elif pendulum.parse(current_date) >= pendulum.datetime(2023, 9, 1):
        # lr_clf, 2023-09-01, 30dpd_6mob
        training_filepath_X = XCom.get_one(execution_date=pendulum.datetime(2023,9,1), task_id="training_preprocessing", key="gold_training_view_X_filepath")
        validation_filepath_X = XCom.get_one(execution_date=pendulum.datetime(2023,9,1), task_id="training_preprocessing", key="gold_validation_view_X_filepath")
        df_train_X = pd.read_parquet(training_filepath_X)
        df_val_X = pd.read_parquet(validation_filepath_X)
        df_train = pd.concat([df_train_X, df_val_X], axis=0, ignore_index=True)
    
    features = df_train.columns

    monitor = PSIModelMonitor(df_train, features)

    dates = [dt for dt in pendulum.period(ti.task.start_date, pendulum.parse(current_date)).range('months')]
    for date in dates:
        # Get current label records
        inference_filepath = XCom.get_one(execution_date=date, task_id='inference_preprocessing', key='gold_inference_view_X_filepath')
        df_recent = pd.read_parquet(inference_filepath)

        monitor.monitor_drift(df_recent, monitoring_date=date)
    monitor.generate_monitoring_report(filepath_txt, filepath_fig, len(dates))
        
    logger.info(f"[{ti.task_id} | {current_date}] Monitoring report created.")
