import os
import logging
import pendulum

from airflow.models import XCom

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

# Silver file prefixes
silver_lms_prefix = "silver_lms_"
silver_cs_prefix = "silver_cs_"
silver_att_prefix = "silver_att_"
silver_fin_prefix = "silver_fin_"

# Task
def data_processing_silver(ti, **context):

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

    # Load Silver store paths
    silver_lms_directory = ti.xcom_pull(task_ids='create_silver_store', key='silver_lms_directory')
    silver_cs_directory = ti.xcom_pull(task_ids='create_silver_store', key='silver_cs_directory')
    silver_att_directory = ti.xcom_pull(task_ids='create_silver_store', key='silver_att_directory')
    silver_fin_directory = ti.xcom_pull(task_ids='create_silver_store', key='silver_fin_directory')

    # Load Bronze filepaths
    bronze_lms_filepath = ti.xcom_pull(task_ids='data_processing_bronze', key='bronze_lms_filepath')
    bronze_cs_filepath = ti.xcom_pull(task_ids='data_processing_bronze', key='bronze_cs_filepath')
    bronze_att_filepath = ti.xcom_pull(task_ids='data_processing_bronze', key='bronze_att_filepath')
    bronze_fin_filepath = ti.xcom_pull(task_ids='data_processing_bronze', key='bronze_fin_filepath')
    current_date = context['ds']

    # Get the previous successful TaskInstance for this task
    silver_lms_prev_filepath = XCom.get_one(execution_date=pendulum.parse(current_date).subtract(months=1), task_id="data_processing_silver", key="silver_lms_filepath")
    if silver_lms_prev_filepath:
        df_lms_prev = spark.read.parquet(silver_lms_prev_filepath)
        logger.info(f"Loaded Silver LMS from previous run with {df_lms_prev.count()} rows: {silver_lms_prev_filepath}")
    else:
        logger.warning(f"[{ti.task_id} | {current_date}] No previous Silver LMS task instance found. Is this the first run?")

    silver_cs_prev_filepath = XCom.get_one(execution_date=pendulum.parse(current_date).subtract(months=1), task_id="data_processing_silver", key="silver_cs_filepath")
    if silver_cs_prev_filepath:
        df_cs_prev = spark.read.parquet(silver_cs_prev_filepath)
        logger.info(f"Loaded Silver Clickstream from previous run with {df_cs_prev.count()} rows: {silver_cs_prev_filepath}")
    else:
        logger.warning(f"[{ti.task_id} | {current_date}] No previous Silver Clickstream task instance found. Is this the first run?")
    
    silver_att_prev_filepath = XCom.get_one(execution_date=pendulum.parse(current_date).subtract(months=1), task_id="data_processing_silver", key="silver_att_filepath")
    if silver_att_prev_filepath:
        df_att_prev = spark.read.parquet(silver_att_prev_filepath)
        logger.info(f"Loaded Silver Attributes from previous run with {df_att_prev.count()}: {silver_att_prev_filepath}")
    else:
        logger.warning(f"[{ti.task_id} | {current_date}] No previous task instance found. Is this the first run?")

    silver_fin_prev_filepath = XCom.get_one(execution_date=pendulum.parse(current_date).subtract(months=1), task_id="data_processing_silver", key="silver_fin_filepath")
    if silver_fin_prev_filepath:
        df_fin_prev = spark.read.parquet(silver_fin_prev_filepath)
        logger.info(f"Loaded Silver Financials from previous run with {df_fin_prev.count()}: {silver_fin_prev_filepath}")
    else:
        logger.warning(f"[{ti.task_id} | {current_date}] No previous task instance found. Is this the first run?")

    # Attributes
    att_column_type_map = {
        "customer_id": T.StringType(),
        "name": T.StringType(),
        "age": T.IntegerType(),
        "ssn": T.StringType(),
        "occupation": T.StringType(),
        "snapshot_date": T.DateType()
    }
    df = spark.read.csv(bronze_att_filepath, header=True, inferSchema=False)
    logger.info(f"Loaded Bronze Clickstream with {df.count()} rows from : {bronze_att_filepath}")
 
    # Convert column headers to lower case
    df = df.toDF(*[c.lower() for c in df.columns])
    # Clean underscores
    for column in ["name", "age", "occupation"]:
        df = clean_underscores(df, column)
    # Cast column types
    for column, new_type in att_column_type_map.items():
        df = df.withColumn(column, F.col(column).cast(new_type))
    # Clean age
    df = df.withColumn("age", F.when(F.col("age") < 1, None).otherwise(F.col("age")))
    df = df.withColumn("age", F.when(F.col("age") > 150, None).otherwise(F.col("age")))
    # Clean name and occupation
    df = df.withColumn("name", F.when(F.col("name")=="", None).otherwise(F.col("name")))
    df = df.withColumn("name", F.regexp_replace(F.col("name"), r'"{3,}', ""))
    df = df.withColumn("occupation", F.when(F.col("occupation")=="", None).otherwise(F.col("occupation")))
    # Clean ssn
    pattern = r"^\d{3}-\d{2}-\d{4}$"
    df = df.withColumn("ssn", F.when(F.col("ssn").rlike(pattern), F.col("ssn")).otherwise(None))
    # Populate null values for categorical variables
    for column in ["ssn", "occupation"]:
        df = df.withColumn(column, F.when(F.col(column).isNull(), "Unknown").otherwise(F.col(column)))
    # Check Null values
    for column in df.columns:
        if df.select(column).where(F.col(column).isNull()).count() > 0:
            logger.info(f"Silver Attributes for {current_date}: {df.filter(df[column].isNull()).count()} null values @ {column} column.")

    # Create cumulative dataframe
    if silver_att_prev_filepath: df_all = df_att_prev.union(df)
    else: df_all = df
    # Drop duplicates
    ori_count = df_all.count()
    df_all = df_all.dropDuplicates()
    logger.info(f"Silver Attributes for {current_date}: {df_all.count() - ori_count} duplicates dropped.")
    logger.info(f"Silver Attributes till {current_date}: {df_all.count()} rows.")

    # Write cumulative datafile
    silver_att_filepath = os.path.join(silver_att_directory, silver_att_prefix + current_date + '.parquet')
    df_all.write.mode("overwrite").parquet(silver_att_filepath)
    logger.info(f"Silver Attributes: {current_date}: {df_all.count()} rows saved to: {silver_att_filepath}")
    ti.xcom_push(key='silver_att_filepath', value=silver_att_filepath)


    # Clickstream
    cs_column_type_map = {
        "fe_1": T.IntegerType(),
        "fe_2": T.IntegerType(),
        "fe_3": T.IntegerType(),
        "fe_4": T.IntegerType(),
        "fe_5": T.IntegerType(),
        "fe_6": T.IntegerType(),
        "fe_7": T.IntegerType(),
        "fe_8": T.IntegerType(),
        "fe_9": T.IntegerType(),
        "fe_10": T.IntegerType(),
        "fe_11": T.IntegerType(),
        "fe_12": T.IntegerType(),
        "fe_13": T.IntegerType(),
        "fe_14": T.IntegerType(),
        "fe_15": T.IntegerType(),
        "fe_16": T.IntegerType(),
        "fe_17": T.IntegerType(),
        "fe_18": T.IntegerType(),
        "fe_19": T.IntegerType(),
        "fe_20": T.IntegerType(),
        "customer_id": T.StringType(),
        "snapshot_date": T.DateType()
    }

    df = spark.read.csv(bronze_cs_filepath, header=True, inferSchema=False)
    logger.info(f"Loaded Bronze Clickstream with {df.count()} rows from : {bronze_cs_filepath}")

    # Convert column headers to lower case
    df = df.toDF(*[c.lower() for c in df.columns])
    # Cast column types
    for column, new_type in cs_column_type_map.items():
        df = df.withColumn(column, F.col(column).cast(new_type))

    # Create cumulative dataframe
    if silver_cs_prev_filepath: df_all = df_cs_prev.union(df)
    else: df_all = df
    # Drop duplicates
    ori_count = df_all.count()
    df_all = df_all.dropDuplicates()
    logger.info(f"Silver Clickstream for {current_date}: {df_all.count() - ori_count} duplicates dropped.")
    logger.info(f"Silver Clickstream till {current_date}: {df_all.count()} rows.")

    # Write cumulative datafile
    silver_cs_filepath = os.path.join(silver_cs_directory, silver_cs_prefix + current_date + '.parquet')
    df_all.write.mode("overwrite").parquet(silver_cs_filepath)
    logger.info(f"Silver Clickstream: {current_date}: {df_all.count()} rows saved to: {silver_cs_filepath}")
    ti.xcom_push(key='silver_cs_filepath', value=silver_cs_filepath)


    # Financials
    fin_column_type_map = {
        "customer_id": T.StringType(),
        "annual_income": T.FloatType(),
        "monthly_inhand_salary": T.FloatType(),
        "num_bank_accounts": T.IntegerType(),
        "num_credit_card": T.IntegerType(),
        "interest_rate": T.IntegerType(),
        "num_of_loan": T.IntegerType(),
        "type_of_loan": T.StringType(),
        "delay_from_due_date": T.IntegerType(),
        "num_of_delayed_payment": T.IntegerType(),
        "changed_credit_limit": T.FloatType(),
        "num_credit_inquiries": T.IntegerType(),
        "credit_mix": T.StringType(),
        "outstanding_debt": T.FloatType(),
        "credit_utilization_ratio": T.FloatType(),
        "credit_history_age": T.StringType(),
        "payment_of_min_amount": T.StringType(),
        "total_emi_per_month": T.FloatType(),
        "amount_invested_monthly": T.FloatType(),
        "payment_behaviour": T.StringType(),
        "monthly_balance": T.FloatType(),
        "snapshot_date": T.DateType()
    }

    df = spark.read.csv(bronze_fin_filepath, header=True, inferSchema=False)
    logger.info(f"Loaded Bronze Financials with {df.count()} rows from : {bronze_fin_filepath}")

    # Convert column headers to lower case
    df = df.toDF(*[c.lower() for c in df.columns])
    # Clean underscores
    for column in ["annual_income", "monthly_inhand_salary", "num_bank_accounts", "num_credit_card", "interest_rate", "num_of_loan", "delay_from_due_date", "num_of_delayed_payment", "changed_credit_limit", "num_credit_inquiries", "outstanding_debt", "credit_utilization_ratio", "total_emi_per_month", "amount_invested_monthly", "monthly_balance", "credit_mix"]:
        df = clean_underscores(df, column)
    # Clean payment behaviour
    df = df.withColumn("payment_behaviour", F.when(F.length(F.col("payment_behaviour")) <10, None).otherwise(F.col("payment_behaviour")))
    # Clean negative values
    for column in ['num_of_loan', 'num_credit_card', 'num_bank_accounts', 'delay_from_due_date', 'num_of_delayed_payment']:
        df = df.withColumn(column, F.when(F.col(column) < 0, None).otherwise(F.col(column)))
    # Clean empty
    for column in df.columns:
        df = df.withColumn(column, F.when(F.col(column) == "", None).otherwise(F.col(column)))
        df = df.withColumn(column, F.when(F.col(column).isNull(), None).otherwise(F.col(column)))
    # Cast column types
    for column, new_type in fin_column_type_map.items():
        if new_type == T.IntegerType():
            df = df.withColumn(column, F.col(column).cast(T.FloatType()))
        df = df.withColumn(column, F.col(column).cast(new_type))
    # # Populate null values for categorical variables
    for column in ["credit_mix", "payment_behaviour", "type_of_loan"]:
        df = df.withColumn(column, F.when(F.col(column).isNull(), "Unknown").otherwise(F.col(column)))
    # Check Null values
    for column in df.columns:
        if df.select(column).where(F.col(column).isNull()).count() > 0:
            logger.info(f"Silver Financials for {current_date}: {df.select(column).where(F.col(column).isNull()).count()} null values @ {column} column.")

    # Create cumulative dataframe
    if silver_fin_prev_filepath: df_all = df_fin_prev.union(df)
    else: df_all = df
    # Drop duplicates
    ori_count = df_all.count()
    df_all = df_all.dropDuplicates()
    logger.info(f"Silver Financials for {current_date}: {df_all.count() - ori_count} duplicates dropped.")
    logger.info(f"Silver Financials till {current_date}: {df_all.count()} rows.")

    # Write cumulative datafile
    silver_fin_filepath = os.path.join(silver_fin_directory, silver_fin_prefix + current_date + '.parquet')
    df_all.write.mode("overwrite").parquet(silver_fin_filepath)
    logger.info(f"Silver Financials: {current_date}: {df_all.count()} rows saved to: {silver_fin_filepath}")
    ti.xcom_push(key='silver_fin_filepath', value=silver_fin_filepath)


    # LMS
    lms_column_type_map = {
        "loan_id": T.StringType(),
        "customer_id": T.StringType(),
        "loan_start_date": T.DateType(),
        "tenure": T.IntegerType(),
        "installment_num": T.IntegerType(),
        "loan_amt": T.FloatType(),
        "due_amt": T.FloatType(),
        "paid_amt": T.FloatType(),
        "overdue_amt": T.FloatType(),
        "balance": T.FloatType(),
        "snapshot_date": T.DateType()
    }

    df = spark.read.csv(bronze_lms_filepath, header=True, inferSchema=False)
    logger.info(f"Loaded Bronze LMS with {df.count()} rows from : {bronze_lms_filepath}")

    # Convert column headers to lower case
    df = df.toDF(*[c.lower() for c in df.columns])
    # Drop duplicates
    ori_count = df_all.count()
    df_all = df_all.dropDuplicates()
    logger.info(f"Silver LMS for {current_date}: {df_all.count() - ori_count} duplicates dropped.")
    logger.info(f"Silver LMS till {current_date}: {df_all.count()} rows.")
    # Cast column types
    for column, new_type in lms_column_type_map.items():
        df = df.withColumn(column, F.col(column).cast(new_type))
    # Check Null values
    for column in df.columns:
        if df.select(column).where(F.col(column).isNull()).count() > 0:
            logger.info(f"Silver LMS for {current_date}: {df.select(column).where(F.col(column).isNull()).count()} null values @ {column} column.")

    # Augment data: add month on book
    df = df.withColumn("mob", F.col("installment_num").cast(T.IntegerType()))
    # Augment data: add days past due
    df = df.withColumn("installments_missed", F.when(F.col("due_amt") == 0, 0).otherwise(F.ceil(F.col("overdue_amt") / F.col("due_amt")).cast(T.IntegerType()))).fillna(0)
    df = df.withColumn("first_missed_date", F.when(F.col("installments_missed") > 0, F.add_months(F.col("snapshot_date"), -1 * F.col("installments_missed"))).cast(T.DateType()))
    df = df.withColumn("dpd", F.when(F.col("overdue_amt") > 0.0, F.datediff(F.col("snapshot_date"), F.col("first_missed_date"))).otherwise(0).cast(T.IntegerType()))

    # Create cumulative dataframe
    if silver_lms_prev_filepath: df_all = df_lms_prev.union(df)
    else: df_all = df

    # Write cumulative datafile
    silver_lms_filepath = os.path.join(silver_lms_directory, silver_lms_prefix + current_date + '.parquet')
    df_all.write.mode("overwrite").parquet(silver_lms_filepath)
    logger.info(f"Silver LMS: {current_date}: {df_all.count()} rows saved to: {silver_lms_filepath}")
    ti.xcom_push(key='silver_lms_filepath', value=silver_lms_filepath)


    spark.stop()


# Helper Functions
def clean_underscores(df, column):
    df = df.withColumn(column, F.regexp_replace(column, "_", ""))
    df = df.withColumn(column, F.when(F.col(column) == '', None).otherwise(F.col(column)))
    return df
