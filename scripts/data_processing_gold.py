import os
import logging
import re

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T

from sklearn.model_selection import train_test_split


# Logger
logger = logging.getLogger('ml_pipeline')  # Set the logger name
handler = logging.FileHandler('/app/ml_pipeline.log')
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Gold file prefixes
gold_label_prefix = "gold_label_"
gold_feature_prefix = "gold_feature_"
gold_training_prefix = "gold_train_"
gold_validation_prefix = "gold_valid_"
gold_testing_prefix = "gold_test_"
gold_oot_prefix = "gold_oot_"

# Gold file suffixes
merged_suffix = "_beforeNormalisation"

# Label definition (As given in assignment)
labelAssignment_dpd = 30 # Days past due
labelAssignment_mob = 6 # Month on book


# Tasks
def data_processing_gold_label(ti, **context):

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

    # Load Gold store paths
    gold_label_directory = ti.xcom_pull(task_ids='create_gold_store', key='gold_label_directory')

    # Load Silver filepaths
    silver_lms_filepath = ti.xcom_pull(task_ids='data_processing_silver', key='silver_lms_filepath')
    current_date = context['ds']

    df = spark.read.parquet(silver_lms_filepath)
    logger.info(f"[{ti.task_id}] Loaded Silver LMS with {df.count()} rows from : {silver_lms_filepath}")

    df = df.filter(F.col("mob") == labelAssignment_mob)

    # Label data
    df = df.withColumn("label", F.when(F.col("dpd") >= labelAssignment_dpd, 1).otherwise(0).cast(T.IntegerType()))
    df = df.withColumn("label_def", F.lit(str(labelAssignment_dpd)+'dpd_'+str(labelAssignment_mob)+'mob').cast(T.StringType()))

    # select columns to save and save to gold table, IRL connect to database to write
    df = df.select("loan_id", "customer_id", "loan_start_date", "label", "label_def", "snapshot_date")
    partition_name = gold_label_prefix + current_date + '_' + str(labelAssignment_dpd) + 'dpd_' + str(labelAssignment_mob) + 'mob.parquet'
    filepath = os.path.join(gold_label_directory, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    logger.info(f"[{ti.task_id}] Gold Labels: {current_date}: {df.count()} rows saved to: {filepath}")
    ti.xcom_push(key='gold_label_filepath', value=filepath)


    spark.stop()


def data_processing_gold_feature(ti, **context):

    # Spark Sessions
    spark = SparkSession.builder \
        .master("local[*]") \
        .config("spark.executor.memory", "8g") \
        .config("spark.executor.memoryOverhead", "1g") \
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC") \
        .getOrCreate()
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
    spark.sparkContext.setLogLevel("ERROR")
    spark

    # Load Gold store paths
    gold_feature_directory = ti.xcom_pull(task_ids='create_gold_store', key='gold_feature_directory')

    # Load Silver filepaths
    silver_att_filepath = ti.xcom_pull(task_ids='data_processing_silver', key='silver_att_filepath')
    silver_fin_filepath = ti.xcom_pull(task_ids='data_processing_silver', key='silver_fin_filepath')
    current_date = context['ds']
    
    df_att = spark.read.parquet(silver_att_filepath)
    logger.info(f"[{ti.task_id}] Loaded Silver Attributes with {df_att.count()} rows from : {silver_att_filepath}")
    df_fin = spark.read.parquet(silver_fin_filepath)
    logger.info(f"[{ti.task_id}] Loaded Silver Financials with {df_fin.count()} rows from : {silver_fin_filepath}")

    # Join attributes and financials
    df = df_att.join(df_fin, on=["customer_id", "snapshot_date"], how="outer")
    logger.info(f'[{ti.task_id}] Joined Silver Attributes ({df_att.count()}) and Financials ({df_fin.count()}) to {df.count()} rows with {df.select("customer_id").where(F.col("customer_id").isNull()).count()} null customer_id rows.')
    
    # Further feature engineering
    # Drop identifiers and features which result in overfitting
    df = df.drop("name", "ssn")
    # Convert credit history age
    parse_duration_udf = F.udf(parse_duration, T.FloatType())
    df = df.withColumn("credit_history_age", parse_duration_udf("credit_history_age"))

    # Encoder for type_of_loan which is categorical, but stored in a unique way.
    content = [row['type_of_loan'] for row in df.select('type_of_loan').collect()]
    collapsed_content = ', '.join(content)
    loan_types = re.split(r'[,]\s|and\s', collapsed_content)
    loan_types = set(loan_types)
    [loan_types.remove(x) for x in ['', 'Unknown', 'Not Specified']]

    loan_types = ['Payday Loan', 'Personal Loan', 'Home Equity Loan', 'Credit-Builder Loan', 'Mortgage Loan', 'Student Loan', 'Debt Consolidation Loan', 'Auto Loan']
    for loan_type in loan_types:
        df = df.withColumn('type_of_loan'+'_'+loan_type.lower().replace(' ', '_'), F.when(F.col('type_of_loan').contains(loan_type), 1).otherwise(0).cast(T.IntegerType()))
    df = df.drop('type_of_loan')

    # Create new features
    df = df.withColumn("investment_income_ratio", F.when(F.col("monthly_inhand_salary") == 0, 1).otherwise(F.col("amount_invested_monthly") / F.col("monthly_inhand_salary")))
    df = df.withColumn("income_per_card", F.when(F.col("num_credit_card") == 0, F.col("monthly_inhand_salary")).otherwise(F.col("monthly_inhand_salary") / F.col("num_credit_card")))

    # Save
    partition_name = gold_feature_prefix + current_date + '.parquet'
    filepath = os.path.join(gold_feature_directory, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Gold Features: {df.count()} rows saved to: {filepath}")
    ti.xcom_push(key='gold_feature_filepath', value=filepath)


    spark.stop()


def data_post_processing(ti, **context):

    # Spark Sessions
    spark = SparkSession.builder \
        .master("local[*]") \
        .config("spark.executor.memory", "8g") \
        .config("spark.executor.memoryOverhead", "1g") \
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC") \
        .getOrCreate()
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
    spark.sparkContext.setLogLevel("ERROR")
    spark

    # Load Gold store paths
    gold_training_view = ti.xcom_pull(task_ids='create_gold_store', key='gold_training_view')
    gold_validation_view = ti.xcom_pull(task_ids='create_gold_store', key='gold_validation_view')
    gold_testing_view = ti.xcom_pull(task_ids='create_gold_store', key='gold_testing_view')
    gold_oot_view = ti.xcom_pull(task_ids='create_gold_store', key='gold_oot_view')

    # Load Gold filepaths
    gold_label_filepath = ti.xcom_pull(task_ids='data_processing_gold_label', key='gold_label_filepath')
    gold_feature_filepath = ti.xcom_pull(task_ids='data_processing_gold_feature', key='gold_feature_filepath')
    current_date = context['ds']

    df_feature = spark.read.parquet(gold_feature_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded Gold Features with {df_feature.count()} rows from: {gold_feature_filepath}")
    df_label = spark.read.parquet(gold_label_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded Gold Labels with {df_label.count()} rows from: {gold_label_filepath}")

    # Merge feature and label data. Be careful of time leakage. Feature snapshot date should be equal to loan start date, not label snapshot date.
    df_feature = df_feature.withColumn("feature_snapshot_date", F.col("snapshot_date"))
    df_feature = df_feature.withColumnRenamed("snapshot_date", "loan_start_date")
    df_merged = df_feature.join(df_label, on=["customer_id", "loan_start_date"], how="inner")
    df_merged = df_merged.withColumnRenamed("snapshot_date", "label_snapshot_date")
    logger.info(f"[{ti.task_id} | {current_date}] Merged Gold features and labels with {df_merged.count()} rows.")

    # Data split - Train, Validation, Test, OOT
    df_oot = df_merged.filter(F.col("label_snapshot_date") == current_date)
    df_train = df_merged.filter(F.col("label_snapshot_date") != current_date).toPandas()
    df_train, df_test = train_test_split(df_train, test_size=0.1, shuffle=True, random_state=42, stratify=df_train["label"])
    df_train, df_val = train_test_split(df_train, test_size=0.2, shuffle=True, random_state=42, stratify=df_train["label"])
    # Convert back to spark dataframe
    df_train = spark.createDataFrame(df_train)
    df_val = spark.createDataFrame(df_val)
    df_test = spark.createDataFrame(df_test)
    # Restore Null values. Conversion to Pandas turns Nulls to NANs.
    for column in ['age', 'annual_income', 'monthly_inhand_salary', 'num_bank_accounts', 'num_credit_card', 'interest_rate', 'num_of_loan', 'delay_from_due_date', 'num_of_delayed_payment', \
                'changed_credit_limit', 'num_credit_inquiries', 'outstanding_debt', 'credit_utilization_ratio', 'total_emi_per_month', 'amount_invested_monthly', 'monthly_balance']:
        df_test = df_test.withColumn(column, F.when(F.isnan(F.col(column)), None).otherwise(F.col(column)))
        df_val = df_val.withColumn(column, F.when(F.isnan(F.col(column)), None).otherwise(F.col(column)))
        df_train = df_train.withColumn(column, F.when(F.isnan(F.col(column)), None).otherwise(F.col(column)))

    # Save post-processing merged datasets for next step of data normalisation
    partition_name = gold_training_prefix + current_date + merged_suffix + '.parquet'
    filepath = os.path.join(gold_training_view, partition_name)
    df_train.to_parquet(filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Gold Training View (Before Normalisation): {df_train.count()} rows saved to: {filepath}")
    ti.xcom_push(key='gold_training_view_before_normalisation', value=filepath)
    # Validation
    partition_name = gold_validation_prefix + current_date + merged_suffix + '.parquet'
    filepath = os.path.join(gold_validation_view, partition_name)
    df_val.to_parquet(filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Gold Validation View (Before Normalisation): {df_val.count()} rows saved to: {filepath}")
    ti.xcom_push(key='gold_validation_view_before_normalisation', value=filepath)
    # Testing
    partition_name = gold_testing_prefix + current_date + merged_suffix + '.parquet'
    filepath = os.path.join(gold_testing_view, partition_name)
    df_test.to_parquet(filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Gold Testing View (Before Normalisation): {df_test.count()} rows saved to: {filepath}")
    ti.xcom_push(key='gold_testing_view_before_normalisation', value=filepath)
    # OOT
    partition_name = gold_oot_prefix + current_date + merged_suffix + '.parquet'
    filepath = os.path.join(gold_oot_view, partition_name)
    df_val.to_parquet(filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Gold OOT View (Before Normalisation): {df_oot.count()} rows saved to: {filepath}")
    ti.xcom_push(key='gold_oot_view_before_normalisation', value=filepath)


    spark.stop()


# Helper Functions
def parse_duration(text):
    match = re.match(r"(\d+)\s+Years?\s+and\s+(\d+)\s+Months?", text)
    if match:
        years = int(match.group(1))
        months = int(match.group(2))
        return years + months / 12.0
    return 0
