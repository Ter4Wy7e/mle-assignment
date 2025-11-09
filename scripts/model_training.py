import os
import logging
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from scripts.encoder_functions import cap_outliers_enc, fill_nulls_enc

from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import xgboost as xgb

# Create Logging Directory
if not os.path.exists("/app/logs"):
    os.makedirs("/app/logs")
    
# Logger
logger = logging.getLogger('training_pipeline')  # Set the logger name
handler = logging.FileHandler('/app/logs/training.log')
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Encoders
encoder_prefix = "encoder_"

labelAssignment_dpd = 30 # Days past due
labelAssignment_mob = 6 # Month on book
label_version_suffix = str(labelAssignment_dpd) + 'dpd_' + str(labelAssignment_mob) + 'mob'

model_suffix = 'generic'

# Gold file prefixes
gold_label_prefix = "gold_label_"
gold_feature_prefix = "gold_feature_"
gold_training_prefix = "gold_train_"
gold_validation_prefix = "gold_valid_"
gold_testing_prefix = "gold_test_"
gold_oot_prefix = "gold_oot_"

# Gold file suffixes
ready_suffix = "ready"

# Model Register
model_register_joblib = 'model_register.joblib'
model_register_csv = 'model_register.csv'

def train_encoders(ti, **context):

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

    # Load model store paths
    model_bank_directory = ti.xcom_pull(task_ids='create_model_bank', key='model_bank_directory')
    # Load gold filepaths
    gold_training_view_before_normalisation_filepath = ti.xcom_pull(task_ids='data_post_processing', key='gold_training_view_before_normalisation')
    current_date = context['ds']

    df_training = spark.read.parquet(gold_training_view_before_normalisation_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded Training View with {df_training.count()} rows from: {gold_training_view_before_normalisation_filepath}")

    # Cap outliers
    caps = {
        'annual_income': df_training.agg(F.percentile('annual_income', 0.97)).collect()[0][0],
        'monthly_inhand_salary': df_training.agg(F.percentile('monthly_inhand_salary', 0.97)).collect()[0][0],
        'repayment_ability': df_training.agg(F.percentile('repayment_ability', 0.97)).collect()[0][0],
        'num_bank_accounts': df_training.agg(F.percentile('num_bank_accounts', 0.97)).collect()[0][0],
        'num_credit_card': df_training.agg(F.percentile('num_credit_card', 0.97)).collect()[0][0],
        'interest_rate': df_training.agg(F.percentile('interest_rate', 0.97)).collect()[0][0],
        'num_of_loan': df_training.agg(F.percentile('num_of_loan', 0.97)).collect()[0][0],
        'delay_from_due_date': df_training.agg(F.percentile('delay_from_due_date', 0.97)).collect()[0][0],
        'num_of_delayed_payment': df_training.agg(F.percentile('num_of_delayed_payment', 0.97)).collect()[0][0],
        'changed_credit_limit': df_training.agg(F.percentile('changed_credit_limit', 0.97)).collect()[0][0],
        'num_credit_inquiries': df_training.agg(F.percentile('num_credit_inquiries', 0.97)).collect()[0][0],
        'outstanding_debt': df_training.agg(F.percentile('outstanding_debt', 0.97)).collect()[0][0],
        'credit_utilization_ratio': df_training.agg(F.percentile('credit_utilization_ratio', 0.97)).collect()[0][0],
        'total_emi_per_month': df_training.agg(F.percentile('total_emi_per_month', 0.97)).collect()[0][0],
        'amount_invested_monthly': df_training.agg(F.percentile('amount_invested_monthly', 0.97)).collect()[0][0],
        'monthly_balance': df_training.agg(F.percentile('monthly_balance', 0.97)).collect()[0][0]
    }
    # def cap_outliers_enc(df, caps = caps):
    #     for key in caps.keys():
    #         df = df.withColumn(key, F.when(F.col(key) > caps[key], caps[key]).otherwise(F.col(key)))
    #     return df
    filename = encoder_prefix + 'cap_outliers_values' + '_' + current_date + '_' + label_version_suffix + '_' + model_suffix + '.joblib'
    filepath = os.path.join(model_bank_directory, filename)
    joblib.dump(caps, filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Cap values for cap_outliers_enc saved to: {filepath}")
    ti.xcom_push(key='cap_outliers_values_filepath', value=filepath)

    # Fill nulls
    fill_nulls_values = {
        'age': df_training.select(F.median('age')).first()[0],
        'annual_income': df_training.select(F.median('annual_income')).first()[0],
        'monthly_inhand_salary': df_training.select(F.median('monthly_inhand_salary')).first()[0],
        'repayment_ability': df_training.select(F.median('repayment_ability')).first()[0],
        'num_bank_accounts': df_training.select(F.median('num_bank_accounts')).first()[0],
        'num_credit_card': df_training.select(F.median('num_credit_card')).first()[0],
        'interest_rate': df_training.select(F.median('interest_rate')).first()[0],
        'num_of_loan': df_training.select(F.median('num_of_loan')).first()[0],
        'delay_from_due_date': df_training.select(F.median('delay_from_due_date')).first()[0],
        'num_of_delayed_payment': df_training.select(F.median('num_of_delayed_payment')).first()[0],
        'changed_credit_limit': df_training.select(F.median('changed_credit_limit')).first()[0],
        'num_credit_inquiries': df_training.select(F.median('num_credit_inquiries')).first()[0],
        'outstanding_debt': df_training.select(F.median('outstanding_debt')).first()[0],
        'credit_utilization_ratio': df_training.select(F.median('credit_utilization_ratio')).first()[0],
        'total_emi_per_month': df_training.select(F.median('total_emi_per_month')).first()[0],
        'amount_invested_monthly': df_training.select(F.median('amount_invested_monthly')).first()[0],
        'monthly_balance': df_training.select(F.median('monthly_balance')).first()[0]
    }
    # def fill_nulls_enc(df, fill_nulls_values = fill_nulls_values):
    #     for key in fill_nulls_values.keys():
    #         df = df.withColumn(key, F.when(F.col(key).isNull(), fill_nulls_values[key]).otherwise(F.col(key)))
    #     return df
    filename = encoder_prefix + 'fill_nulls_values' + '_' + current_date + '_' + label_version_suffix + '_' + model_suffix + '.joblib'
    filepath = os.path.join(model_bank_directory, filename)
    joblib.dump(fill_nulls_values, filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Fill values for fill_nulls_enc saved to: {filepath}")
    ti.xcom_push(key='fill_nulls_values_filepath', value=filepath)

    occupation_enc = OneHotEncoder(handle_unknown='ignore')
    occupation_enc.fit(df_training.select('occupation').toPandas())
    filename = encoder_prefix + 'occupation_enc' + '_' + current_date + '_' + label_version_suffix + '_' + model_suffix + '.joblib'
    filepath = os.path.join(model_bank_directory, filename)
    joblib.dump(occupation_enc, filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Encoder occupation_enc saved to: {filepath}")
    ti.xcom_push(key='occupation_enc_filepath', value=filepath)

    payment_of_min_amount_enc = OneHotEncoder(handle_unknown='ignore')
    df_training = df_training.withColumn("payment_of_min_amount", F.when(F.col("payment_of_min_amount") == "NM", None).otherwise(F.col("payment_of_min_amount")))
    payment_of_min_amount_enc.fit(df_training.select('payment_of_min_amount').toPandas())
    filename = encoder_prefix + 'payment_of_min_amount_enc' + '_' + current_date + '_' + label_version_suffix + '_' + model_suffix + '.joblib'
    filepath = os.path.join(model_bank_directory, filename)
    joblib.dump(payment_of_min_amount_enc, filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Encoder payment_of_min_amount_enc saved to: {filepath}")
    ti.xcom_push(key='payment_of_min_amount_enc_filepath', value=filepath)

    payment_behaviour_enc = OneHotEncoder(handle_unknown='ignore')
    payment_behaviour_enc.fit(df_training.select('payment_behaviour').toPandas())
    filename = encoder_prefix + 'payment_behaviour_enc' + '_' + current_date + '_' + label_version_suffix + '_' + model_suffix + '.joblib'
    filepath = os.path.join(model_bank_directory, filename)
    joblib.dump(payment_behaviour_enc, filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Encoder payment_behaviour_enc saved to: {filepath}")
    ti.xcom_push(key='payment_behaviour_enc_filepath', value=filepath)

    credit_mix_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df_training = df_training.withColumn("credit_mix", F.when(F.col("credit_mix") == "Unknown", None).otherwise(F.col("credit_mix")))
    credit_mix_enc.fit(df_training.select('credit_mix').toPandas())
    filename = encoder_prefix + 'credit_mix_enc' + '_' + current_date + '_' + label_version_suffix + '_' + model_suffix + '.joblib'
    filepath = os.path.join(model_bank_directory, filename)
    joblib.dump(credit_mix_enc, filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Encoder credit_mix_enc saved to: {filepath}")
    ti.xcom_push(key='credit_mix_enc_filepath', value=filepath)

    annual_income_scaler = StandardScaler()
    annual_income_scaler.fit(df_training.select('annual_income').toPandas())
    filename = encoder_prefix + 'annual_income_scaler' + '_' + current_date + '_' + label_version_suffix + '_' + model_suffix + '.joblib'
    filepath = os.path.join(model_bank_directory, filename)
    joblib.dump(annual_income_scaler, filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Encoder annual_income_scaler saved to: {filepath}")
    ti.xcom_push(key='annual_income_scaler_filepath', value=filepath)

    repayment_ability_scaler = StandardScaler()
    repayment_ability_scaler.fit(df_training.select('repayment_ability').toPandas())
    filename = encoder_prefix + 'repayment_ability_scaler' + '_' + current_date + '_' + label_version_suffix + '_' + model_suffix + '.joblib'
    filepath = os.path.join(model_bank_directory, filename)
    joblib.dump(repayment_ability_scaler, filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Encoder repayment_ability_scaler saved to: {filepath}")
    ti.xcom_push(key='repayment_ability_scaler_filepath', value=filepath)

    monthly_inhand_salary_scaler = StandardScaler()
    monthly_inhand_salary_scaler.fit(df_training.select('monthly_inhand_salary').toPandas())
    filename = encoder_prefix + 'monthly_inhand_salary_scaler' + '_' + current_date + '_' + label_version_suffix + '_' + model_suffix + '.joblib'
    filepath = os.path.join(model_bank_directory, filename)
    joblib.dump(monthly_inhand_salary_scaler, filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Encoder monthly_inhand_salary_scaler saved to: {filepath}")
    ti.xcom_push(key='monthly_inhand_salary_scaler_filepath', value=filepath)

    outstanding_debt_scaler = StandardScaler()
    outstanding_debt_scaler.fit(df_training.select('outstanding_debt').toPandas())
    filename = encoder_prefix + 'outstanding_debt_scaler' + '_' + current_date + '_' + label_version_suffix + '_' + model_suffix + '.joblib'
    filepath = os.path.join(model_bank_directory, filename)
    joblib.dump(outstanding_debt_scaler, filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Encoder outstanding_debt_scaler saved to: {filepath}")
    ti.xcom_push(key='outstanding_debt_scaler_filepath', value=filepath)

    amount_invested_monthly_scaler = StandardScaler()
    amount_invested_monthly_scaler.fit(df_training.select('amount_invested_monthly').toPandas())
    filename = encoder_prefix + 'amount_invested_monthly_scaler' + '_' + current_date + '_' + label_version_suffix + '_' + model_suffix + '.joblib'
    filepath = os.path.join(model_bank_directory, filename)
    joblib.dump(amount_invested_monthly_scaler, filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Encoder amount_invested_monthly_scaler saved to: {filepath}")
    ti.xcom_push(key='amount_invested_monthly_scaler_filepath', value=filepath)

    monthly_balance_scaler = StandardScaler()
    monthly_balance_scaler.fit(df_training.select('monthly_balance').toPandas())
    filename = encoder_prefix + 'monthly_balance_scaler' + '_' + current_date + '_' + label_version_suffix + '_' + model_suffix + '.joblib'
    filepath = os.path.join(model_bank_directory, filename)
    joblib.dump(monthly_balance_scaler, filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Encoder monthly_balance_scaler saved to: {filepath}")
    ti.xcom_push(key='monthly_balance_scaler_filepath', value=filepath)

    spark.stop()


def training_preprocessing(ti, **context):

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

    # Load gold filepaths
    gold_training_view = ti.xcom_pull(task_ids='create_gold_store', key='gold_training_view')
    gold_validation_view = ti.xcom_pull(task_ids='create_gold_store', key='gold_validation_view')
    gold_testing_view = ti.xcom_pull(task_ids='create_gold_store', key='gold_testing_view')
    gold_oot_view = ti.xcom_pull(task_ids='create_gold_store', key='gold_oot_view')

    gold_training_view_before_normalisation_filepath = ti.xcom_pull(task_ids='data_post_processing', key='gold_training_view_before_normalisation')
    
    current_date = context['ds']

    # Load encoders filepaths
    cap_outliers_values_filepath = ti.xcom_pull(task_ids='train_encoders', key='cap_outliers_values_filepath')
    fill_nulls_values_filepath = ti.xcom_pull(task_ids='train_encoders', key='fill_nulls_values_filepath')
    occupation_enc_filepath = ti.xcom_pull(task_ids='train_encoders', key='occupation_enc_filepath')
    payment_of_min_amount_enc_filepath = ti.xcom_pull(task_ids='train_encoders', key='payment_of_min_amount_enc_filepath')
    payment_behaviour_enc_filepath = ti.xcom_pull(task_ids='train_encoders', key='payment_behaviour_enc_filepath')
    credit_mix_enc_filepath = ti.xcom_pull(task_ids='train_encoders', key='credit_mix_enc_filepath')
    annual_income_scaler_filepath = ti.xcom_pull(task_ids='train_encoders', key='annual_income_scaler_filepath')
    monthly_inhand_salary_scaler_filepath = ti.xcom_pull(task_ids='train_encoders', key='monthly_inhand_salary_scaler_filepath')
    repayment_ability_scaler_filepath = ti.xcom_pull(task_ids='train_encoders', key='repayment_ability_scaler_filepath')
    outstanding_debt_scaler_filepath = ti.xcom_pull(task_ids='train_encoders', key='outstanding_debt_scaler_filepath')
    amount_invested_monthly_scaler_filepath = ti.xcom_pull(task_ids='train_encoders', key='amount_invested_monthly_scaler_filepath')
    monthly_balance_scaler_filepath = ti.xcom_pull(task_ids='train_encoders', key='monthly_balance_scaler_filepath')

    # Load values and encoders
    cap_outliers_values = joblib.load(cap_outliers_values_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded cap_outliers_values from: {cap_outliers_values_filepath}")
    fill_nulls_values = joblib.load(fill_nulls_values_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded fill_nulls_values from: {fill_nulls_values_filepath}")
    occupation_enc = joblib.load(occupation_enc_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded occupation_enc from: {occupation_enc_filepath}")
    payment_of_min_amount_enc = joblib.load(payment_of_min_amount_enc_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded payment_of_min_amount_enc from: {payment_of_min_amount_enc_filepath}")
    payment_behaviour_enc = joblib.load(payment_behaviour_enc_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded payment_behaviour_enc from: {payment_behaviour_enc_filepath}")
    credit_mix_enc = joblib.load(credit_mix_enc_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded credit_mix_enc from: {credit_mix_enc_filepath}")
    annual_income_scaler = joblib.load(annual_income_scaler_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded annual_income_scaler from: {annual_income_scaler_filepath}")
    monthly_inhand_salary_scaler = joblib.load(monthly_inhand_salary_scaler_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded monthly_inhand_salary_scaler from: {monthly_inhand_salary_scaler_filepath}")
    repayment_ability_scaler = joblib.load(repayment_ability_scaler_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded repayment_ability_scaler from: {repayment_ability_scaler_filepath}")
    outstanding_debt_scaler = joblib.load(outstanding_debt_scaler_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded outstanding_debt_scaler from: {outstanding_debt_scaler_filepath}")
    amount_invested_monthly_scaler = joblib.load(amount_invested_monthly_scaler_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded amount_invested_monthly_scaler from: {amount_invested_monthly_scaler_filepath}")
    monthly_balance_scaler = joblib.load(monthly_balance_scaler_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded monthly_balance_scaler from: {monthly_balance_scaler_filepath}")

    # Load data
    gold_training_view_before_normalisation_filepath = ti.xcom_pull(task_ids='data_post_processing', key='gold_training_view_before_normalisation')
    gold_validation_view_before_normalisation_filepath = ti.xcom_pull(task_ids='data_post_processing', key='gold_validation_view_before_normalisation')
    gold_testing_view_before_normalisation_filepath = ti.xcom_pull(task_ids='data_post_processing', key='gold_testing_view_before_normalisation')
    gold_oot_view_before_normalisation_filepath = ti.xcom_pull(task_ids='data_post_processing', key='gold_oot_view_before_normalisation')

    df_train = spark.read.parquet(gold_training_view_before_normalisation_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded gold training view from: {gold_training_view_before_normalisation_filepath}")
    df_val = spark.read.parquet(gold_validation_view_before_normalisation_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded gold validation view from: {gold_validation_view_before_normalisation_filepath}")
    df_test = spark.read.parquet(gold_testing_view_before_normalisation_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded gold testing view from: {gold_testing_view_before_normalisation_filepath}")
    df_oot = spark.read.parquet(gold_oot_view_before_normalisation_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded gold OOT view from: {gold_oot_view_before_normalisation_filepath}")

    # Functions
    df_train = cap_outliers_enc(df_train, cap_outliers_values)
    df_val = cap_outliers_enc(df_val, cap_outliers_values)
    df_test = cap_outliers_enc(df_test, cap_outliers_values)
    df_oot = cap_outliers_enc(df_oot, cap_outliers_values)
    logger.info(f"[{ti.task_id} | {current_date}] Encoded cap_outliers for training, validation, test and OOT datasets.")
    df_train = fill_nulls_enc(df_train, fill_nulls_values)
    df_val = fill_nulls_enc(df_val, fill_nulls_values)
    df_test = fill_nulls_enc(df_test, fill_nulls_values)
    df_oot = fill_nulls_enc(df_oot, fill_nulls_values)
    logger.info(f"[{ti.task_id} | {current_date}] Encoded fill_null_values for training, validation, test and OOT datasets.")

    # Categorical one-hot encoding
    def encode_categorical(df):
        df = df.withColumn("payment_of_min_amount", F.when(F.col("payment_of_min_amount") == "NM", None).otherwise(F.col("payment_of_min_amount")))

        df_pd = df.toPandas()

        occupation_array = occupation_enc.transform(df_pd[['occupation']])
        occupation_array = occupation_array.toarray() 
        df_encoding = pd.DataFrame(occupation_array, columns=occupation_enc.get_feature_names_out(['occupation']))

        df_encoding = df_encoding[occupation_enc.get_feature_names_out(['occupation'])]
        df_pd = pd.concat([df_pd.drop(columns=['occupation']), df_encoding], axis=1)

        payment_array = payment_of_min_amount_enc.transform(df_pd[['payment_of_min_amount']])
        payment_array = payment_array.toarray() 
        df_encoding = pd.DataFrame(payment_array, columns=payment_of_min_amount_enc.get_feature_names_out(['payment_of_min_amount']))
        df_encoding = df_encoding[payment_of_min_amount_enc.get_feature_names_out(['payment_of_min_amount'])]
        df_pd = pd.concat([df_pd.drop(columns=['payment_of_min_amount']), df_encoding], axis=1)

        behaviour_array = payment_behaviour_enc.transform(df_pd[['payment_behaviour']])
        behaviour_array = behaviour_array.toarray() 
        df_encoding = pd.DataFrame(behaviour_array, columns=payment_behaviour_enc.get_feature_names_out(['payment_behaviour']))
        df_encoding = df_encoding[payment_behaviour_enc.get_feature_names_out(['payment_behaviour'])]
        df_pd = pd.concat([df_pd.drop(columns=['payment_behaviour']), df_encoding], axis=1)

        df = spark.createDataFrame(df_pd)
        return df
    
    df_oot = encode_categorical(df_oot)
    df_test = encode_categorical(df_test)
    df_val = encode_categorical(df_val)
    df_train = encode_categorical(df_train)
    logger.info(f"[{ti.task_id} | {current_date}] Completed categorical one-hot encoding for training, validation, test and OOT datasets.")

    # Ordinal encoding
    def encode_ordinal(df):
        df = df.withColumn("credit_mix", F.when(F.col("credit_mix") == "Unknown", None).otherwise(F.col("credit_mix")))

        df_pd = df.toPandas()
        df_pd['credit_mix'] = credit_mix_enc.transform(df_pd[['credit_mix']])
        df = spark.createDataFrame(df_pd)
        return df

    df_oot = encode_ordinal(df_oot)
    df_test = encode_ordinal(df_test)
    df_val = encode_ordinal(df_val)
    df_train = encode_ordinal(df_train)
    logger.info(f"[{ti.task_id} | {current_date}] Completed ordinal encoding for training, validation, test and OOT datasets.")

    # Scaling
    def scaling(df):
        df_pd = df.toPandas()

        df_pd['annual_income'] = annual_income_scaler.transform(df_pd[['annual_income']])
        df_pd['monthly_inhand_salary'] = monthly_inhand_salary_scaler.transform(df_pd[['monthly_inhand_salary']])
        df_pd['repayment_ability'] = repayment_ability_scaler.transform(df_pd[['repayment_ability']])
        df_pd['outstanding_debt'] = outstanding_debt_scaler.transform(df_pd[['outstanding_debt']])
        df_pd['amount_invested_monthly'] = amount_invested_monthly_scaler.transform(df_pd[['amount_invested_monthly']])
        df_pd['monthly_balance'] = monthly_balance_scaler.transform(df_pd[['monthly_balance']])

        df = spark.createDataFrame(df_pd)
        return df

    df_oot = scaling(df_oot)
    df_test = scaling(df_test)
    df_val = scaling(df_val)
    df_train = scaling(df_train)
    logger.info(f"[{ti.task_id} | {current_date}] Completed data scaling for training, validation, test and OOT datasets.")

    # Drop final identifiers, split X Y, and save
    df_train = df_train.drop('customer_id', 'loan_start_date', 'feature_snapshot_date', 'loan_id', 'label_def', 'label_snapshot_date')
    df_oot = df_oot.drop('customer_id', 'loan_start_date', 'feature_snapshot_date', 'loan_id', 'label_def', 'label_snapshot_date')
    df_test = df_test.drop('customer_id', 'loan_start_date', 'feature_snapshot_date', 'loan_id', 'label_def', 'label_snapshot_date')
    df_val = df_val.drop('customer_id', 'loan_start_date', 'feature_snapshot_date', 'loan_id', 'label_def', 'label_snapshot_date')
    logger.info(f"[{ti.task_id} | {current_date}] Completed dropping of identifiers for training, validation, test and OOT datasets.")
    # X-Y Split
    df_train_Y = df_train.select('label').toPandas()
    df_train_X = df_train.drop('label').toPandas()
    df_oot_Y = df_oot.select('label').toPandas()
    df_oot_X = df_oot.drop('label').toPandas()
    df_test_Y = df_test.select('label').toPandas()
    df_test_X = df_test.drop('label').toPandas()
    df_val_Y = df_val.select('label').toPandas()
    df_val_X = df_val.drop('label').toPandas()
    logger.info(f"[{ti.task_id} | {current_date}] Completed X-Y split for training, validation, test and OOT datasets.")

    # Save
    partition_name_X = gold_training_prefix + '_' + current_date + '_' + label_version_suffix +  '_' + ready_suffix + '_X.parquet'
    partition_name_Y = gold_training_prefix + '_' + current_date + '_' + label_version_suffix +  '_' + ready_suffix + '_Y.parquet'
    filepath_X = os.path.join(gold_training_view, partition_name_X)
    filepath_Y = os.path.join(gold_training_view, partition_name_Y)
    df_train_X.to_parquet(filepath_X)
    logger.info(f"[{ti.task_id} | {current_date}] Gold Training View (Ready) features saved to: {filepath_X}")
    df_train_Y.to_parquet(filepath_Y)
    logger.info(f"[{ti.task_id} | {current_date}] Gold Training View (Ready) labels saved to: {filepath_Y}")
    ti.xcom_push(key='gold_training_view_X_filepath', value=filepath_X)
    ti.xcom_push(key='gold_training_view_Y_filepath', value=filepath_Y)

    partition_name_X = gold_validation_prefix + '_' + current_date + '_' + label_version_suffix +  '_' + ready_suffix + '_X.parquet'
    partition_name_Y = gold_validation_prefix + '_' + current_date + '_' + label_version_suffix +  '_' + ready_suffix + '_Y.parquet'
    filepath_X = os.path.join(gold_validation_view, partition_name_X)
    filepath_Y = os.path.join(gold_validation_view, partition_name_Y)
    df_val_X.to_parquet(filepath_X)
    logger.info(f"[{ti.task_id} | {current_date}] Gold Validation View (Ready) features saved to: {filepath_X}")
    df_val_Y.to_parquet(filepath_Y)
    logger.info(f"[{ti.task_id} | {current_date}] Gold Validation View (Ready) labels saved to: {filepath_Y}")
    ti.xcom_push(key='gold_validation_view_X_filepath', value=filepath_X)
    ti.xcom_push(key='gold_validation_view_Y_filepath', value=filepath_Y)

    partition_name_X = gold_testing_prefix + '_' + current_date + '_' + label_version_suffix +  '_' + ready_suffix + '_X.parquet'
    partition_name_Y = gold_testing_prefix + '_' + current_date + '_' + label_version_suffix +  '_' + ready_suffix + '_Y.parquet'
    filepath_X = os.path.join(gold_testing_view, partition_name_X)
    filepath_Y = os.path.join(gold_testing_view, partition_name_Y)
    df_test_X.to_parquet(filepath_X)
    logger.info(f"[{ti.task_id} | {current_date}] Gold Testing View (Ready) features saved to: {filepath_X}")
    df_test_Y.to_parquet(filepath_Y)
    logger.info(f"[{ti.task_id} | {current_date}] Gold Testing View (Ready) labels saved to: {filepath_Y}")
    ti.xcom_push(key='gold_testing_view_X_filepath', value=filepath_X)
    ti.xcom_push(key='gold_testing_view_Y_filepath', value=filepath_Y)

    partition_name_X = gold_oot_prefix + '_' + current_date + '_' + label_version_suffix +  '_' + ready_suffix + '_X.parquet'
    partition_name_Y = gold_oot_prefix + '_' + current_date + '_' + label_version_suffix +  '_' + ready_suffix + '_Y.parquet'
    filepath_X = os.path.join(gold_oot_view, partition_name_X)
    filepath_Y = os.path.join(gold_oot_view, partition_name_Y)
    df_oot_X.to_parquet(filepath_X)
    logger.info(f"[{ti.task_id} | {current_date}] Gold OOT View (Ready) features saved to: {filepath_X}")
    df_oot_Y.to_parquet(filepath_Y)
    logger.info(f"[{ti.task_id} | {current_date}] Gold OOT View (Ready) labels saved to: {filepath_Y}")
    ti.xcom_push(key='gold_oot_view_X_filepath', value=filepath_X)
    ti.xcom_push(key='gold_oot_view_Y_filepath', value=filepath_Y)


    spark.stop()


def train_logistic_regression(ti, **context):
    
    model_prefix = 'lr_clf'
    current_date = context['ds']

    model_bank_directory = ti.xcom_pull(task_ids='create_model_bank', key='model_bank_directory')

    # Load encoders filepaths
    training_filepath_X = ti.xcom_pull(task_ids='training_preprocessing', key='gold_training_view_X_filepath')
    training_filepath_Y = ti.xcom_pull(task_ids='training_preprocessing', key='gold_training_view_Y_filepath')
    validation_filepath_X = ti.xcom_pull(task_ids='training_preprocessing', key='gold_validation_view_X_filepath')
    validation_filepath_Y = ti.xcom_pull(task_ids='training_preprocessing', key='gold_validation_view_Y_filepath')
    testing_filepath_X = ti.xcom_pull(task_ids='training_preprocessing', key='gold_testing_view_X_filepath')
    testing_filepath_Y = ti.xcom_pull(task_ids='training_preprocessing', key='gold_testing_view_Y_filepath')
    oot_filepath_X = ti.xcom_pull(task_ids='training_preprocessing', key='gold_oot_view_X_filepath')
    oot_filepath_Y = ti.xcom_pull(task_ids='training_preprocessing', key='gold_oot_view_Y_filepath')

    df_train_X = pd.read_parquet(training_filepath_X)
    df_train_Y = pd.read_parquet(training_filepath_Y)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded Gold Training datasets with X: {df_train_X.shape[0]} rows and Y: {df_train_Y.shape[0]} rows.")
    df_val_X = pd.read_parquet(validation_filepath_X)
    df_val_Y = pd.read_parquet(validation_filepath_Y)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded Gold Validation datasets with X: {df_train_X.shape[0]} rows and Y: {df_train_Y.shape[0]} rows.")
    df_test_X = pd.read_parquet(testing_filepath_X)
    df_test_Y = pd.read_parquet(testing_filepath_Y)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded Gold Testing datasets with X: {df_train_X.shape[0]} rows and Y: {df_train_Y.shape[0]} rows.")
    df_oot_X = pd.read_parquet(oot_filepath_X)
    df_oot_Y = pd.read_parquet(oot_filepath_Y)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded Gold OOT datasets with X: {df_train_X.shape[0]} rows and Y: {df_train_Y.shape[0]} rows.")
    # Meaningless to maintain differentiation between training and validation because we will be using Search CV.
    df_train_val_X = pd.concat([df_train_X, df_val_X], axis=0, ignore_index=True)
    df_train_val_Y = pd.concat([df_train_Y, df_val_Y], axis=0, ignore_index=True)


    lr_clf = LogisticRegression(solver='liblinear', tol=1e-4, fit_intercept=True, class_weight='balanced', random_state=42)

    # Hyperparameter space
    param_distributions = {
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 1, 100, 1000],  # Regularization strength
        'penalty': ['l1', 'l2'],
        'max_iter': [100, 200, 500, 1000, 10000]
    }
    # Set up the random search with cross-validation
    random_search = RandomizedSearchCV(
        estimator=lr_clf,
        param_distributions=param_distributions,
        scoring=metrics.make_scorer(metrics.recall_score),
        n_iter=100,  # Number of iterations for random search
        cv=5,       # Number of folds in cross-validation
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(df_train_val_X, df_train_val_Y.values.ravel())

    # Output the best parameters and best score
    logger.info(f"[{ti.task_id} | {current_date}] LR CLF: Best parameters: {random_search.best_params_}.")
    logger.info(f"[{ti.task_id} | {current_date}] LR CLF: Best score: {random_search.best_score_}.")

    # Evaluate the model on the train set
    best_model = random_search.best_estimator_

    y_pred = best_model.predict(df_train_val_X)
    train_recall_score = metrics.recall_score(df_train_val_Y.values.ravel(), y_pred)
    train_f1_score = metrics.f1_score(df_train_val_Y.values.ravel(), y_pred)
    train_auc_score = metrics.roc_auc_score(df_train_val_Y.values.ravel(), y_pred)
    logger.info(f"[{ti.task_id} | {current_date}] LR CLF: Train Recall score: {train_recall_score}.")
    logger.info(f"[{ti.task_id} | {current_date}] LR CLF: Train F1 score: {train_f1_score}.")
    logger.info(f"[{ti.task_id} | {current_date}] LR CLF: Train GINI score: {round(2*train_auc_score-1,3)}.")

    y_pred = best_model.predict(df_test_X)
    test_recall_score = metrics.recall_score(df_test_Y.values.ravel(), y_pred)
    test_f1_score = metrics.f1_score(df_test_Y.values.ravel(), y_pred)
    test_auc_score = metrics.roc_auc_score(df_test_Y.values.ravel(), y_pred)
    logger.info(f"[{ti.task_id} | {current_date}] LR CLF: Test Recall score: {train_recall_score}.")
    logger.info(f"[{ti.task_id} | {current_date}] LR CLF: Test F1 score: {train_f1_score}.")
    logger.info(f"[{ti.task_id} | {current_date}] LR CLF: Test GINI score: {round(2*train_auc_score-1,3)}.")

    y_pred = best_model.predict(df_oot_X)
    oot_recall_score = metrics.recall_score(df_oot_Y.values.ravel(), y_pred)
    oot_f1_score = metrics.f1_score(df_oot_Y.values.ravel(), y_pred)
    oot_auc_score = metrics.roc_auc_score(df_oot_Y.values.ravel(), y_pred)
    logger.info(f"[{ti.task_id} | {current_date}] LR CLF: OOT Recall score: {train_recall_score}.")
    logger.info(f"[{ti.task_id} | {current_date}] LR CLF: OOT F1 score: {train_f1_score}.")
    logger.info(f"[{ti.task_id} | {current_date}] LR CLF: OOT GINI score: {round(2*train_auc_score-1,3)}.")


    # Save to model bank
    filename = model_prefix + '_' + current_date + '_' + label_version_suffix + '.joblib'
    filepath = os.path.join(model_bank_directory, filename)
    joblib.dump(best_model, filepath)
    logger.info(f"[{ti.task_id} | {current_date}] LR CLF: Saved model to: {filepath}.")

    # Store results and key parameters to table
    results = {
        'run_date': [current_date],
        'model_type': ['Logistic Regression'],
        'label_version' : [label_version_suffix],
        'model_params': [random_search.best_params_],
        'train_X_path': [training_filepath_X],
        'train_Y_path': [training_filepath_Y],
        'val_X_path': [validation_filepath_X],
        'val_Y_path': [validation_filepath_Y],
        'test_X_path': [testing_filepath_X],
        'test_Y_path': [testing_filepath_Y],
        'oot_X_path': [oot_filepath_X],
        'oot_Y_path': [oot_filepath_Y],
        'train_recall' : [train_recall_score],
        'train_f1' : [train_f1_score],
        'train_gini' : [round(2*train_auc_score-1,3)],
        'test_recall' : [test_recall_score],
        'test_f1' : [test_f1_score],
        'test_gini' : [round(2*test_auc_score-1,3)],
        'oot_recall' : [oot_recall_score],
        'oot_f1' : [oot_f1_score],
        'oot_gini' : [round(2*oot_auc_score-1,3)]
    }

    df_results = pd.DataFrame(results)

    model_register_csv_filepath = os.path.join(model_bank_directory, model_register_csv)
    model_register_joblib_filepath = os.path.join(model_bank_directory, model_register_joblib)
    if os.path.exists(model_register_joblib_filepath):
        model_register = joblib.load(model_register_joblib_filepath)
        df_results = pd.concat([model_register, df_results])
    joblib.dump(df_results, model_register_joblib_filepath)
    df_results.to_csv(model_register_csv_filepath, index=False, sep='|')
    logger.info(f"[{ti.task_id} | {current_date}] LR CLF: Model register updated at: {model_register_csv_filepath}.")

    df_results_lr = df_results[df_results['model_type'] == 'Logistic Regression']

    plt.figure(figsize=(24, 16))

    plt.plot(df_results_lr['run_date'], df_results_lr['train_recall'], label="Train Recall")
    plt.plot(df_results_lr['run_date'], df_results_lr['train_f1'], label="Train F1")
    plt.plot(df_results_lr['run_date'], df_results_lr['test_recall'], label="Test Recall")
    plt.plot(df_results_lr['run_date'], df_results_lr['test_f1'], label="Test F1")
    plt.plot(df_results_lr['run_date'], df_results_lr['oot_recall'], label="OOT Recall")
    plt.plot(df_results_lr['run_date'], df_results_lr['oot_f1'], label="OOT F1")
    
    plt.xlabel('Date')
    plt.ylabel("Performance Score")
    plt.title('Training Results for Logistic Regression')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_bank_directory, 'training_results_lr_' + current_date + '.png'))  # Save the image
    plt.close()


def train_xgb(ti, **context):

    model_prefix = 'xgb_clf'
    current_date = context['ds']

    model_bank_directory = ti.xcom_pull(task_ids='create_model_bank', key='model_bank_directory')

    # Load encoders filepaths
    training_filepath_X = ti.xcom_pull(task_ids='training_preprocessing', key='gold_training_view_X_filepath')
    training_filepath_Y = ti.xcom_pull(task_ids='training_preprocessing', key='gold_training_view_Y_filepath')
    validation_filepath_X = ti.xcom_pull(task_ids='training_preprocessing', key='gold_validation_view_X_filepath')
    validation_filepath_Y = ti.xcom_pull(task_ids='training_preprocessing', key='gold_validation_view_Y_filepath')
    testing_filepath_X = ti.xcom_pull(task_ids='training_preprocessing', key='gold_testing_view_X_filepath')
    testing_filepath_Y = ti.xcom_pull(task_ids='training_preprocessing', key='gold_testing_view_Y_filepath')
    oot_filepath_X = ti.xcom_pull(task_ids='training_preprocessing', key='gold_oot_view_X_filepath')
    oot_filepath_Y = ti.xcom_pull(task_ids='training_preprocessing', key='gold_oot_view_Y_filepath')

    df_train_X = pd.read_parquet(training_filepath_X)
    df_train_Y = pd.read_parquet(training_filepath_Y)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded Gold Training datasets with X: {df_train_X.shape[0]} rows and Y: {df_train_Y.shape[0]} rows.")
    df_val_X = pd.read_parquet(validation_filepath_X)
    df_val_Y = pd.read_parquet(validation_filepath_Y)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded Gold Validation datasets with X: {df_train_X.shape[0]} rows and Y: {df_train_Y.shape[0]} rows.")
    df_test_X = pd.read_parquet(testing_filepath_X)
    df_test_Y = pd.read_parquet(testing_filepath_Y)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded Gold Testing datasets with X: {df_train_X.shape[0]} rows and Y: {df_train_Y.shape[0]} rows.")
    df_oot_X = pd.read_parquet(oot_filepath_X)
    df_oot_Y = pd.read_parquet(oot_filepath_Y)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded Gold OOT datasets with X: {df_train_X.shape[0]} rows and Y: {df_train_Y.shape[0]} rows.")
    # Meaningless to maintain differentiation between training and validation because we will be using Search CV.
    df_train_val_X = pd.concat([df_train_X, df_val_X], axis=0, ignore_index=True)
    df_train_val_Y = pd.concat([df_train_Y, df_val_Y], axis=0, ignore_index=True)


    xgb_clf = xgb.XGBClassifier(n_jobs=-1, random_state=42)

    # Hyperparameter space
    y = df_train_val_Y.values.ravel()
    scale_pos_weight = np.sum(y == 0) / np.sum(y == 1)
    param_distributions = {
        'n_estimators': [10, 25, 50, 75, 100, 150, 200],
        'scale_pos_weight': [scale_pos_weight, scale_pos_weight*2, scale_pos_weight*3, scale_pos_weight*0.7, scale_pos_weight*0.5],
        'max_depth': [2, 3, 5, 7, 9, 15, 20],  # lower max_depth to simplify the model
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.5, 1, 3, 5, 7, 9, 10],
        'min_child_weight': [1, 3, 5, 7, 9],
        'reg_alpha': [0, 0.1, 0.5, 0.7, 1],
        'reg_lambda': [0.1, 0.5, 1, 1.5, 2, 5, 7, 9, 10]
    }
    # Set up the random search with cross-validation
    xgb_clf_random_search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_distributions,
        scoring=metrics.make_scorer(metrics.f1_score),
        n_iter=100,  # Number of iterations for random search
        cv=5,       # Number of folds in cross-validation
        random_state=42,
        n_jobs=-1
    )

    xgb_clf_random_search.fit(df_train_val_X, df_train_val_Y.values.ravel())

    # Output the best parameters and best score
    logger.info(f"[{ti.task_id} | {current_date}] XGB CLF: Best parameters: {xgb_clf_random_search.best_params_}.")
    logger.info(f"[{ti.task_id} | {current_date}] XGB CLF: Best score: {xgb_clf_random_search.best_score_}.")

    # Evaluate the model on the train set
    best_model = xgb_clf_random_search.best_estimator_

    y_pred = best_model.predict(df_train_val_X)
    train_recall_score = metrics.f1_score(df_train_val_Y.values.ravel(), y_pred)
    train_f1_score = metrics.f1_score(df_train_val_Y.values.ravel(), y_pred)
    train_auc_score = metrics.roc_auc_score(df_train_val_Y.values.ravel(), y_pred)
    logger.info(f"[{ti.task_id} | {current_date}] XGB CLF: Train Recall score: {train_recall_score}.")
    logger.info(f"[{ti.task_id} | {current_date}] XGB CLF: Train F1 score: {train_f1_score}.")
    logger.info(f"[{ti.task_id} | {current_date}] XGB CLF: Train GINI score: {round(2*train_auc_score-1,3)}.")

    y_pred = best_model.predict(df_test_X)
    test_recall_score = metrics.recall_score(df_test_Y.values.ravel(), y_pred)
    test_f1_score = metrics.f1_score(df_test_Y.values.ravel(), y_pred)
    test_auc_score = metrics.roc_auc_score(df_test_Y.values.ravel(), y_pred)
    logger.info(f"[{ti.task_id} | {current_date}] XGB CLF: Test Recall score: {train_recall_score}.")
    logger.info(f"[{ti.task_id} | {current_date}] XGB CLF: Test F1 score: {train_f1_score}.")
    logger.info(f"[{ti.task_id} | {current_date}] XGB CLF: Test GINI score: {round(2*train_auc_score-1,3)}.")

    y_pred = best_model.predict(df_oot_X)
    oot_recall_score = metrics.recall_score(df_oot_Y.values.ravel(), y_pred)
    oot_f1_score = metrics.f1_score(df_oot_Y.values.ravel(), y_pred)
    oot_auc_score = metrics.roc_auc_score(df_oot_Y.values.ravel(), y_pred)
    logger.info(f"[{ti.task_id} | {current_date}] XGB CLF: OOT Recall score: {train_recall_score}.")
    logger.info(f"[{ti.task_id} | {current_date}] XGB CLF: OOT F1 score: {train_f1_score}.")
    logger.info(f"[{ti.task_id} | {current_date}] XGB CLF: OOT GINI score: {round(2*train_auc_score-1,3)}.")


    # Save to model bank
    filename = model_prefix + '_' + current_date + '_' + label_version_suffix + '.joblib'
    filepath = os.path.join(model_bank_directory, filename)
    joblib.dump(best_model, filepath)
    logger.info(f"[{ti.task_id} | {current_date}] XGB CLF: Saved model to: {filepath}.")

    # Store results and key parameters to table
    results = {
        'run_date': [current_date],
        'model_type': ['XGBoost'],
        'label_version' : [label_version_suffix],
        'model_params': [xgb_clf_random_search.best_params_],
        'train_X_path': [training_filepath_X],
        'train_Y_path': [training_filepath_Y],
        'val_X_path': [validation_filepath_X],
        'val_Y_path': [validation_filepath_Y],
        'test_X_path': [testing_filepath_X],
        'test_Y_path': [testing_filepath_Y],
        'oot_X_path': [oot_filepath_X],
        'oot_Y_path': [oot_filepath_Y],
        'train_recall' : [train_recall_score],
        'train_f1' : [train_f1_score],
        'train_gini' : [round(2*train_auc_score-1,3)],
        'test_recall' : [test_recall_score],
        'test_f1' : [test_f1_score],
        'test_gini' : [round(2*test_auc_score-1,3)],
        'oot_recall' : [oot_recall_score],
        'oot_f1' : [oot_f1_score],
        'oot_gini' : [round(2*oot_auc_score-1,3)]
    }

    df_results = pd.DataFrame(results)

    model_register_csv_filepath = os.path.join(model_bank_directory, model_register_csv)
    model_register_joblib_filepath = os.path.join(model_bank_directory, model_register_joblib)
    if os.path.exists(model_register_joblib_filepath):
        model_register = joblib.load(model_register_joblib_filepath)
        df_results = pd.concat([model_register, df_results])
    joblib.dump(df_results, model_register_joblib_filepath)
    df_results.to_csv(model_register_csv_filepath, index=False, sep='|')
    logger.info(f"[{ti.task_id} | {current_date}] LR CLF: Model register updated at: {model_register_csv_filepath}.")
    ti.xcom_push(key='model_register_joblib_filepath', value=model_register_joblib_filepath)

    df_results_xgb = df_results[df_results['model_type'] == 'XGBoost']

    plt.figure(figsize=(24, 16))

    plt.plot(df_results_xgb['run_date'], df_results_xgb['train_recall'], label="Train Recall")
    plt.plot(df_results_xgb['run_date'], df_results_xgb['train_f1'], label="Train F1")
    plt.plot(df_results_xgb['run_date'], df_results_xgb['test_recall'], label="Test Recall")
    plt.plot(df_results_xgb['run_date'], df_results_xgb['test_f1'], label="Test F1")
    plt.plot(df_results_xgb['run_date'], df_results_xgb['oot_recall'], label="OOT Recall")
    plt.plot(df_results_xgb['run_date'], df_results_xgb['oot_f1'], label="OOT F1")
    
    plt.xlabel('Date')
    plt.ylabel("Performance Score")
    plt.title('Training Results for XGBoost')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_bank_directory, 'training_results_xbg_' + current_date + '.png')) 

    df_results_lr = df_results[df_results['model_type'] == 'Logistic Regression']

    plt.figure(figsize=(24, 16))

    plt.plot(df_results_lr['run_date'], df_results_lr['oot_recall'], label="LR Recall")
    plt.plot(df_results_xgb['run_date'], df_results_xgb['oot_recall'], label="XGB Recall")
    plt.plot(df_results_lr['run_date'], df_results_lr['oot_f1'], label="LR F1")
    plt.plot(df_results_xgb['run_date'], df_results_xgb['oot_f1'], label="XGB F1")

    
    plt.xlabel('Date')
    plt.ylabel("Performance Score")
    plt.title('Comparison of Training Results')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_bank_directory, 'training_results_comparison_' + current_date + '.png'))  # Save the image
    plt.close()

