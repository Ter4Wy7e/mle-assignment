import os
import logging

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T


# Logger
logger = logging.getLogger('data_pipeline')  # Set the logger name
handler = logging.FileHandler('/app/logs/data_pipeline.log')
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# Raw Data Path
raw_data_path = "/app/data"

# Data Filenames
lms_csv = "lms_loan_daily.csv"
clickstream_csv = "feature_clickstream.csv"
attributes_csv = "features_attributes.csv"
financials_csv = "features_financials.csv"

# Bronze file prefixes
bronze_lms_prefix = "bronze_lms_"
bronze_cs_prefix = "bronze_cs_"
bronze_att_prefix = "bronze_att_"
bronze_fin_prefix = "bronze_fin_"


# Task
def data_processing_bronze(ti, **context):

    # Spark Session
    spark = SparkSession.builder \
        .master("local[*]") \
        .config("spark.executor.memory", "8g") \
        .config("spark.executor.memoryOverhead", "1g") \
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC") \
        .getOrCreate()
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
    spark.sparkContext.setLogLevel("ERROR")
    spark

    # Load dataframes
    df_lms = spark.read.csv(os.path.join(raw_data_path, lms_csv), header=True, inferSchema=True)
    logger.info(f"Loaded Raw LMS CSV from : {os.path.join(raw_data_path, lms_csv)}")
    df_cs = spark.read.csv(os.path.join(raw_data_path, clickstream_csv), header=True, inferSchema=True)
    logger.info(f"Loaded Raw Clickstream CSV from : {os.path.join(raw_data_path, clickstream_csv)}")
    df_att = spark.read.csv(os.path.join(raw_data_path, attributes_csv), header=True, inferSchema=True)
    logger.info(f"Loaded Raw Attributes CSV from : {os.path.join(raw_data_path, attributes_csv)}")
    df_fin = spark.read.csv(os.path.join(raw_data_path, financials_csv), header=True, inferSchema=True)
    logger.info(f"Loaded Raw Financials CSV from : {os.path.join(raw_data_path, financials_csv)}")

    bronze_lms_directory = ti.xcom_pull(task_ids='create_bronze_store', key='bronze_lms_directory')
    bronze_cs_directory = ti.xcom_pull(task_ids='create_bronze_store', key='bronze_cs_directory')
    bronze_att_directory = ti.xcom_pull(task_ids='create_bronze_store', key='bronze_att_directory')
    bronze_fin_directory = ti.xcom_pull(task_ids='create_bronze_store', key='bronze_fin_directory')
    current_date = context['ds']

    # LMS
    df_partition = df_lms.filter(F.col('snapshot_date') == current_date)
    filename = bronze_lms_prefix + current_date + '.csv'
    filepath = bronze_lms_directory + filename
    df_partition.toPandas().to_csv(filepath, index=False)
    logger.info(f'Bronze LMS: {current_date}: {df_partition.count()} records saved to: {filepath}')
    ti.xcom_push(key='bronze_lms_filepath', value=filepath)

    # Clickstream
    df_partition = df_cs.filter(F.col('snapshot_date') == current_date)
    filename = bronze_cs_prefix + current_date + '.csv'
    filepath = bronze_cs_directory + filename
    df_partition.toPandas().to_csv(filepath, index=False)
    logger.info(f'Bronze Clickstream: {current_date}: {df_partition.count()} records saved to: {filepath}')
    ti.xcom_push(key='bronze_cs_filepath', value=filepath)

    # Attributes
    df_partition = df_att.filter(F.col('snapshot_date') == current_date)
    filename = bronze_att_prefix + current_date + '.csv'
    filepath = bronze_att_directory + filename
    df_partition.toPandas().to_csv(filepath, index=False)
    logger.info(f'Bronze Attributes: {current_date}: {df_partition.count()} records saved to: {filepath}')
    ti.xcom_push(key='bronze_att_filepath', value=filepath)

    # Financials
    df_partition = df_fin.filter(F.col('snapshot_date') == current_date)
    filename = bronze_fin_prefix + current_date + '.csv'
    filepath = bronze_fin_directory + filename
    df_partition.toPandas().to_csv(filepath, index=False)
    logger.info(f'Bronze Financials: {current_date}: {df_partition.count()} records saved to: {filepath}')
    ti.xcom_push(key='bronze_fin_filepath', value=filepath)

    spark.stop()
