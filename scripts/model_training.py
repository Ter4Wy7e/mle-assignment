import os
import logging
import joblib
import pandas as pd

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


# Logger
logger = logging.getLogger('ml_pipeline')  # Set the logger name
handler = logging.FileHandler('/app/ml_pipeline.log')
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Encoders
encoder_prefix = "encoder_"

labelAssignment_dpd = 30 # Days past due
labelAssignment_mob = 6 # Month on book
label_version_suffix = '_' + str(labelAssignment_dpd) + 'dpd_' + str(labelAssignment_mob)

model_suffix = '_generic'

# Gold file prefixes
gold_label_prefix = "gold_label_"
gold_feature_prefix = "gold_feature_"
gold_training_prefix = "gold_train_"
gold_validation_prefix = "gold_valid_"
gold_testing_prefix = "gold_test_"
gold_oot_prefix = "gold_oot_"

# Gold file suffixes
ready_suffix = "_ready"



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
    gold_training_view_before_normalisation_filepath = ti.xcom_push(task_ids='data_post_processing', key='gold_training_view_before_normalisation')
    current_date = context['ds']

    df_training = spark.read.parquet(gold_training_view_before_normalisation_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded Training View with {df_training.count()} rows from: {gold_training_view_before_normalisation_filepath}")

    # Cap outliers
    caps = {
        'annual_income': df_training.agg(F.percentile('annual_income', 0.97)).collect()[0][0],
        'monthly_inhand_salary': df_training.agg(F.percentile('monthly_inhand_salary', 0.97)).collect()[0][0],
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
    def cap_outliers_enc(df, caps = caps):
        for key in caps.keys():
            df = df.withColumn(key, F.when(F.col(key) > caps[key], caps[key]).otherwise(F.col(key)))
        return df
    filename = encoder_prefix + 'cap_outliers_enc' + current_date + label_version_suffix + model_suffix + '.joblib'
    filepath = os.path.join(model_bank_directory, filename)
    joblib.dump(cap_outliers_enc, filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Encoder cap_outliers_enc saved to: {filepath}")
    ti.xcom_push(key='cap_outliers_enc_filepath', value=filepath)

    # Fill nulls
    fill_null_values = {
        'age': df_training.select(F.median('age')).first()[0],
        'annual_income': df_training.select(F.median('annual_income')).first()[0],
        'monthly_inhand_salary': df_training.select(F.median('monthly_inhand_salary')).first()[0],
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
    def fill_nulls_enc(df, fill_null_values = fill_null_values):
        for key in fill_null_values.keys():
            df = df.withColumn(key, F.when(F.col(key).isNull(), fill_null_values[key]).otherwise(F.col(key)))
        return df
    filename = encoder_prefix + 'fill_nulls' + current_date + label_version_suffix + model_suffix + '.joblib'
    filepath = os.path.join(model_bank_directory, filename)
    joblib.dump(fill_nulls_enc, filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Encoder fill_nulls_enc saved to: {filepath}")
    ti.xcom_push(key='fill_nulls_enc_filepath', value=filepath)

    occupation_enc = OneHotEncoder(handle_unknown='ignore')
    occupation_enc.fit(df_training.select('occupation').toPandas())
    filename = encoder_prefix + 'occupation_enc' + current_date + label_version_suffix + model_suffix + '.joblib'
    filepath = os.path.join(model_bank_directory, filename)
    joblib.dump(occupation_enc, filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Encoder occupation_enc saved to: {filepath}")
    ti.xcom_push(key='occupation_enc_filepath', value=filepath)

    payment_of_min_amount_enc = OneHotEncoder(handle_unknown='ignore')
    df_training = df_training.withColumn("payment_of_min_amount", F.when(F.col("payment_of_min_amount") == "NM", None).otherwise(F.col("payment_of_min_amount")))
    payment_of_min_amount_enc.fit(df_training.select('payment_of_min_amount').toPandas())
    filename = encoder_prefix + 'payment_of_min_amount_enc' + current_date + label_version_suffix + model_suffix + '.joblib'
    filepath = os.path.join(model_bank_directory, filename)
    joblib.dump(payment_of_min_amount_enc, filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Encoder payment_of_min_amount_enc saved to: {filepath}")
    ti.xcom_push(key='payment_of_min_amount_enc_filepath', value=filepath)

    payment_behaviour_enc = OneHotEncoder(handle_unknown='ignore')
    payment_behaviour_enc.fit(df_training.select('payment_behaviour').toPandas())
    filename = encoder_prefix + 'payment_behaviour_enc' + current_date + label_version_suffix + model_suffix + '.joblib'
    filepath = os.path.join(model_bank_directory, filename)
    joblib.dump(payment_behaviour_enc, filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Encoder payment_behaviour_enc saved to: {filepath}")
    ti.xcom_push(key='payment_behaviour_enc_filepath', value=filepath)

    credit_mix_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df_training = df_training.withColumn("credit_mix", F.when(F.col("credit_mix") == "Unknown", None).otherwise(F.col("credit_mix")))
    credit_mix_enc.fit(df_training.select('credit_mix').toPandas())
    filename = encoder_prefix + 'credit_mix_enc' + current_date + label_version_suffix + model_suffix + '.joblib'
    filepath = os.path.join(model_bank_directory, filename)
    joblib.dump(credit_mix_enc, filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Encoder credit_mix_enc saved to: {filepath}")
    ti.xcom_push(key='credit_mix_enc_filepath', value=filepath)

    annual_income_scaler = StandardScaler()
    annual_income_scaler.fit(df_training.select('annual_income').toPandas())
    filename = encoder_prefix + 'annual_income_scaler' + current_date + label_version_suffix + model_suffix + '.joblib'
    filepath = os.path.join(model_bank_directory, filename)
    joblib.dump(annual_income_scaler, filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Encoder annual_income_scaler saved to: {filepath}")
    ti.xcom_push(key='annual_income_scaler_filepath', value=filepath)

    monthly_inhand_salary_scaler = StandardScaler()
    monthly_inhand_salary_scaler.fit(df_training.select('monthly_inhand_salary').toPandas())
    filename = encoder_prefix + 'monthly_inhand_salary_scaler' + current_date + label_version_suffix + model_suffix + '.joblib'
    filepath = os.path.join(model_bank_directory, filename)
    joblib.dump(monthly_inhand_salary_scaler, filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Encoder monthly_inhand_salary_scaler saved to: {filepath}")
    ti.xcom_push(key='monthly_inhand_salary_scaler_filepath', value=filepath)

    outstanding_debt_scaler = StandardScaler()
    outstanding_debt_scaler.fit(df_training.select('outstanding_debt').toPandas())
    filename = encoder_prefix + 'outstanding_debt_scaler' + current_date + label_version_suffix + model_suffix + '.joblib'
    filepath = os.path.join(model_bank_directory, filename)
    joblib.dump(outstanding_debt_scaler, filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Encoder outstanding_debt_scaler saved to: {filepath}")
    ti.xcom_push(key='outstanding_debt_scaler_filepath', value=filepath)

    amount_invested_monthly_scaler = StandardScaler()
    amount_invested_monthly_scaler.fit(df_training.select('amount_invested_monthly').toPandas())
    filename = encoder_prefix + 'amount_invested_monthly_scaler' + current_date + label_version_suffix + model_suffix + '.joblib'
    filepath = os.path.join(model_bank_directory, filename)
    joblib.dump(amount_invested_monthly_scaler, filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Encoder amount_invested_monthly_scaler saved to: {filepath}")
    ti.xcom_push(key='amount_invested_monthly_scaler_filepath', value=filepath)

    monthly_balance_scaler = StandardScaler()
    monthly_balance_scaler.fit(df_training.select('monthly_balance').toPandas())
    filename = encoder_prefix + 'monthly_balance_scaler' + current_date + label_version_suffix + model_suffix + '.joblib'
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
    gold_training_view = ti.xcom_pull(task_ids='create_gold_store', key='gold_training_view'),
    gold_validation_view = ti.xcom_pull(task_ids='create_gold_store', key='gold_validation_view'),
    gold_testing_view = ti.xcom_pull(task_ids='create_gold_store', key='gold_testing_view'),
    gold_oot_view = ti.xcom_pull(task_ids='create_gold_store', key='gold_oot_view')

    gold_training_view_before_normalisation_filepath = ti.xcom_push(task_ids='data_post_processing', key='gold_training_view_before_normalisation')
    
    current_date = context['ds']

    # Load encoders filepaths
    cap_outliers_enc_filepath = ti.xcom_pull(task_ids='create_encoders', key='cap_outliers_enc_filepath')
    fill_null_values_enc_filepath = ti.xcom_pull(task_ids='create_encoders', key='fill_null_values_enc_filepath')
    occupation_enc_filepath = ti.xcom_pull(task_ids='create_encoders', key='occupation_enc_filepath')
    payment_of_min_amount_enc_filepath = ti.xcom_pull(task_ids='create_encoders', key='payment_of_min_amount_enc_filepath')
    payment_behaviour_enc_filepath = ti.xcom_pull(task_ids='create_encoders', key='payment_behaviour_enc_filepath')
    credit_mix_enc_filepath = ti.xcom_pull(task_ids='create_encoders', key='credit_mix_enc_filepath')
    annual_income_scaler_filepath = ti.xcom_pull(task_ids='create_encoders', key='annual_income_scaler_filepath')
    monthly_inhand_salary_scaler_filepath = ti.xcom_pull(task_ids='create_encoders', key='monthly_inhand_salary_scaler_filepath')
    outstanding_debt_scaler_filepath = ti.xcom_pull(task_ids='create_encoders', key='outstanding_debt_scaler_filepath')
    amount_invested_monthly_scaler_filepath = ti.xcom_pull(task_ids='create_encoders', key='amount_invested_monthly_scaler_filepath')
    monthly_balance_scaler_filepath = ti.xcom_pull(task_ids='create_encoders', key='monthly_balance_scaler_filepath')

    # Load encoders
    cap_outliers_enc = joblib.load(cap_outliers_enc_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded cap_outliers_enc from: {cap_outliers_enc_filepath}")
    fill_null_values_enc = joblib.load(fill_null_values_enc_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded fill_null_values_enc from: {fill_null_values_enc_filepath}")
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
    df_train = cap_outliers_enc(df_train)
    df_val = cap_outliers_enc(df_val)
    df_test = cap_outliers_enc(df_test)
    df_oot = cap_outliers_enc(df_oot)
    logger.info(f"[{ti.task_id} | {current_date}] Encoded cap_outliers for training, validation, test and OOT datasets.")
    df_train = fill_null_values_enc(df_train)
    df_val = fill_null_values_enc(df_val)
    df_test = fill_null_values_enc(df_test)
    df_oot = fill_null_values_enc(df_oot)
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
    partition_name_X = gold_training_prefix + current_date + label_version_suffix + ready_suffix + '_X.parquet'
    partition_name_Y = gold_training_prefix + current_date + label_version_suffix + ready_suffix + '_Y.parquet'
    filepath_X = os.path.join(gold_training_view, partition_name_X)
    filepath_Y = os.path.join(gold_training_view, partition_name_Y)
    df_train_X.to_parquet(filepath_X)
    logger.info(f"[{ti.task_id} | {current_date}] Gold Training View (Ready) features saved to: {filepath_X}")
    df_train_Y.to_parquet(filepath_Y)
    logger.info(f"[{ti.task_id} | {current_date}] Gold Training View (Ready) labels saved to: {filepath_Y}")

    partition_name_X = gold_validation_prefix + current_date + label_version_suffix + ready_suffix + '_X.parquet'
    partition_name_Y = gold_validation_prefix + current_date + label_version_suffix + ready_suffix + '_Y.parquet'
    filepath_X = os.path.join(gold_validation_view, partition_name_X)
    filepath_Y = os.path.join(gold_validation_view, partition_name_Y)
    df_val_X.to_parquet(filepath_X)
    logger.info(f"[{ti.task_id} | {current_date}] Gold Validation View (Ready) features saved to: {filepath_X}")
    df_val_Y.to_parquet(filepath_Y)
    logger.info(f"[{ti.task_id} | {current_date}] Gold Validation View (Ready) labels saved to: {filepath_Y}")

    partition_name_X = gold_testing_prefix + current_date + label_version_suffix + ready_suffix + '_X.parquet'
    partition_name_Y = gold_testing_prefix + current_date + label_version_suffix + ready_suffix + '_Y.parquet'
    filepath_X = os.path.join(gold_testing_view, partition_name_X)
    filepath_Y = os.path.join(gold_testing_view, partition_name_Y)
    df_test_X.to_parquet(filepath_X)
    logger.info(f"[{ti.task_id} | {current_date}] Gold Testing View (Ready) features saved to: {filepath_X}")
    df_test_Y.to_parquet(filepath_Y)
    logger.info(f"[{ti.task_id} | {current_date}] Gold Testing View (Ready) labels saved to: {filepath_Y}")

    partition_name_X = gold_oot_prefix + current_date + label_version_suffix + ready_suffix + '_X.parquet'
    partition_name_Y = gold_oot_prefix + current_date + label_version_suffix + ready_suffix + '_Y.parquet'
    filepath_X = os.path.join(gold_oot_view, partition_name_X)
    filepath_Y = os.path.join(gold_oot_view, partition_name_Y)
    df_oot_X.to_parquet(filepath_X)
    logger.info(f"[{ti.task_id} | {current_date}] Gold OOT View (Ready) features saved to: {filepath_X}")
    df_oot_Y.to_parquet(filepath_Y)
    logger.info(f"[{ti.task_id} | {current_date}] Gold OOT View (Ready) labels saved to: {filepath_Y}")


    spark.stop()
