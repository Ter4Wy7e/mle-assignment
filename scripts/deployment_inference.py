import os
import logging
import pendulum
import joblib

import pandas as pd

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T

from sklearn.model_selection import train_test_split

from scripts.encoder_functions import cap_outliers_enc, fill_nulls_enc

# Create Logging Directory
if not os.path.exists("/app/logs"):
    os.makedirs("/app/logs")

# Logger
logger = logging.getLogger('inference_pipeline')  # Set the logger name
handler = logging.FileHandler('/app/logs/inference_pipeline.log')
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Inference directories
outbox_directory = "/app/outbox"

gold_inference_prefix = "gold_inference_"

def create_outbox(ti, **context):
    current_date = context['ds']

    if not os.path.exists(outbox_directory):
        os.makedirs(outbox_directory)
        logger.info(f"[{ti.task_id} | {current_date}] Created outbox: {outbox_directory}")
    else:
        logger.info(f"[{ti.task_id} | {current_date}] Outbox already exists: {outbox_directory}")
    ti.xcom_push(key='outbox_directory', value=outbox_directory)


def deploy_model(ti, **context):

    model_bank_directory = ti.xcom_pull(task_ids='create_model_bank', key='model_bank_directory')

    current_date = context['ds']
    pendulum.parse(current_date)

    # Active model selection
    if pendulum.parse(current_date) >= pendulum.datetime(2024, 8, 1):
        active_model = 'xgb_clf'
        active_model_date = '2024-08-01'
        active_model_version = '30dpd_6mob'
        active_metric_p0 = 'recall'
        active_threshold_p0 = 0.70
        active_metric_p1 = 'f1'
        active_threshold_p1 = 0.6
        active_metric_p2 = 'gini'
        active_threshold_p2 = 0.3
        
        active_model_filename = active_model + '_' + active_model_date + '_' + active_model_version + '.joblib'
        encoder_payment_of_min_amount_enc_filename = 'encoder_payment_of_min_amount_enc_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_payment_behaviour_enc_filename = 'encoder_payment_behaviour_enc_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_outstanding_debt_scaler_filename = 'encoder_outstanding_debt_scaler_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_occupation_enc = 'encoder_occupation_enc_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_monthly_inhand_salary_scaler = 'encoder_monthly_inhand_salary_scaler_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_repayment_ability_scaler = 'encoder_repayment_ability_scaler_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_monthly_balance_scaler = 'encoder_monthly_balance_scaler_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_fill_nulls_values = 'encoder_fill_nulls_values_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_credit_mix_enc = 'encoder_credit_mix_enc_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_cap_outliers_values = 'encoder_cap_outliers_values_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_annual_income_scaler = 'encoder_annual_income_scaler_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_amount_invested_monthly_scaler = 'encoder_amount_invested_monthly_scaler_' + active_model_date + '_' + active_model_version + '_generic.joblib'

        active_model_filepath = os.path.join(model_bank_directory, active_model_filename)
        encoder_payment_of_min_amount_enc_filepath = os.path.join(model_bank_directory, encoder_payment_of_min_amount_enc_filename)
        encoder_payment_behaviour_enc_filepath = os.path.join(model_bank_directory, encoder_payment_behaviour_enc_filename)
        encoder_outstanding_debt_scaler_filepath = os.path.join(model_bank_directory, encoder_outstanding_debt_scaler_filename)
        encoder_occupation_enc_filepath = os.path.join(model_bank_directory, encoder_occupation_enc)
        encoder_monthly_inhand_salary_scaler_filepath = os.path.join(model_bank_directory, encoder_monthly_inhand_salary_scaler)
        encoder_repayment_ability_scaler_filepath = os.path.join(model_bank_directory, encoder_repayment_ability_scaler)
        encoder_monthly_balance_scaler_filepath = os.path.join(model_bank_directory, encoder_monthly_balance_scaler)
        encoder_fill_nulls_values_filepath = os.path.join(model_bank_directory, encoder_fill_nulls_values)
        encoder_credit_mix_enc_filepath = os.path.join(model_bank_directory, encoder_credit_mix_enc)
        encoder_cap_outliers_values_filepath = os.path.join(model_bank_directory, encoder_cap_outliers_values)
        encoder_annual_income_scaler_filepath = os.path.join(model_bank_directory, encoder_annual_income_scaler)
        encoder_amount_invested_monthly_scaler_filepath = os.path.join(model_bank_directory, encoder_amount_invested_monthly_scaler)

        ti.xcom_push(key='active_model', value=active_model)
        ti.xcom_push(key='active_model_date', value=active_model_date)
        ti.xcom_push(key='active_model_version', value=active_model_version)
        ti.xcom_push(key='active_metric_p0', value=active_metric_p0)
        ti.xcom_push(key='active_threshold_p0', value=active_threshold_p0)
        ti.xcom_push(key='active_metric_p1', value=active_metric_p1)
        ti.xcom_push(key='active_threshold_p1', value=active_threshold_p1)
        ti.xcom_push(key='active_metric_p2', value=active_metric_p2)
        ti.xcom_push(key='active_threshold_p2', value=active_threshold_p2)

        ti.xcom_push(key='active_model_filepath', value=active_model_filepath)
        ti.xcom_push(key='encoder_payment_of_min_amount_enc_filepath', value=encoder_payment_of_min_amount_enc_filepath)
        ti.xcom_push(key='encoder_payment_behaviour_enc_filepath', value=encoder_payment_behaviour_enc_filepath)
        ti.xcom_push(key='encoder_outstanding_debt_scaler_filepath', value=encoder_outstanding_debt_scaler_filepath)
        ti.xcom_push(key='encoder_occupation_enc_filepath', value=encoder_occupation_enc_filepath)
        ti.xcom_push(key='encoder_monthly_inhand_salary_scaler_filepath', value=encoder_monthly_inhand_salary_scaler_filepath)
        ti.xcom_push(key='encoder_repayment_ability_scaler_filepath', value=encoder_repayment_ability_scaler_filepath)
        ti.xcom_push(key='encoder_monthly_balance_scaler_filepath', value=encoder_monthly_balance_scaler_filepath)
        ti.xcom_push(key='encoder_fill_nulls_values_filepath', value=encoder_fill_nulls_values_filepath)
        ti.xcom_push(key='encoder_credit_mix_enc_filepath', value=encoder_credit_mix_enc_filepath)
        ti.xcom_push(key='encoder_cap_outliers_values_filepath', value=encoder_cap_outliers_values_filepath)
        ti.xcom_push(key='encoder_annual_income_scaler_filepath', value=encoder_annual_income_scaler_filepath)
        ti.xcom_push(key='encoder_amount_invested_monthly_scaler_filepath', value=encoder_amount_invested_monthly_scaler_filepath)

    elif pendulum.parse(current_date) >= pendulum.datetime(2024, 2, 1):
        active_model = 'xgb_clf'
        active_model_date = '2024-02-01'
        active_model_version = '30dpd_6mob'
        active_metric_p0 = 'recall'
        active_threshold_p0 = 0.70
        active_metric_p1 = 'f1'
        active_threshold_p1 = 0.6
        active_metric_p2 = 'gini'
        active_threshold_p2 = 0.3
        
        active_model_filename = active_model + '_' + active_model_date + '_' + active_model_version + '.joblib'
        encoder_payment_of_min_amount_enc_filename = 'encoder_payment_of_min_amount_enc_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_payment_behaviour_enc_filename = 'encoder_payment_behaviour_enc_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_outstanding_debt_scaler_filename = 'encoder_outstanding_debt_scaler_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_occupation_enc = 'encoder_occupation_enc_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_monthly_inhand_salary_scaler = 'encoder_monthly_inhand_salary_scaler_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_repayment_ability_scaler = 'encoder_repayment_ability_scaler_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_monthly_balance_scaler = 'encoder_monthly_balance_scaler_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_fill_nulls_values = 'encoder_fill_nulls_values_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_credit_mix_enc = 'encoder_credit_mix_enc_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_cap_outliers_values = 'encoder_cap_outliers_values_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_annual_income_scaler = 'encoder_annual_income_scaler_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_amount_invested_monthly_scaler = 'encoder_amount_invested_monthly_scaler_' + active_model_date + '_' + active_model_version + '_generic.joblib'

        active_model_filepath = os.path.join(model_bank_directory, active_model_filename)
        encoder_payment_of_min_amount_enc_filepath = os.path.join(model_bank_directory, encoder_payment_of_min_amount_enc_filename)
        encoder_payment_behaviour_enc_filepath = os.path.join(model_bank_directory, encoder_payment_behaviour_enc_filename)
        encoder_outstanding_debt_scaler_filepath = os.path.join(model_bank_directory, encoder_outstanding_debt_scaler_filename)
        encoder_occupation_enc_filepath = os.path.join(model_bank_directory, encoder_occupation_enc)
        encoder_monthly_inhand_salary_scaler_filepath = os.path.join(model_bank_directory, encoder_monthly_inhand_salary_scaler)
        encoder_repayment_ability_scaler_filepath = os.path.join(model_bank_directory, encoder_repayment_ability_scaler)
        encoder_monthly_balance_scaler_filepath = os.path.join(model_bank_directory, encoder_monthly_balance_scaler)
        encoder_fill_nulls_values_filepath = os.path.join(model_bank_directory, encoder_fill_nulls_values)
        encoder_credit_mix_enc_filepath = os.path.join(model_bank_directory, encoder_credit_mix_enc)
        encoder_cap_outliers_values_filepath = os.path.join(model_bank_directory, encoder_cap_outliers_values)
        encoder_annual_income_scaler_filepath = os.path.join(model_bank_directory, encoder_annual_income_scaler)
        encoder_amount_invested_monthly_scaler_filepath = os.path.join(model_bank_directory, encoder_amount_invested_monthly_scaler)

        ti.xcom_push(key='active_model', value=active_model)
        ti.xcom_push(key='active_model_date', value=active_model_date)
        ti.xcom_push(key='active_model_version', value=active_model_version)
        ti.xcom_push(key='active_metric_p0', value=active_metric_p0)
        ti.xcom_push(key='active_threshold_p0', value=active_threshold_p0)
        ti.xcom_push(key='active_metric_p1', value=active_metric_p1)
        ti.xcom_push(key='active_threshold_p1', value=active_threshold_p1)
        ti.xcom_push(key='active_metric_p2', value=active_metric_p2)
        ti.xcom_push(key='active_threshold_p2', value=active_threshold_p2)

        ti.xcom_push(key='active_model_filepath', value=active_model_filepath)
        ti.xcom_push(key='encoder_payment_of_min_amount_enc_filepath', value=encoder_payment_of_min_amount_enc_filepath)
        ti.xcom_push(key='encoder_payment_behaviour_enc_filepath', value=encoder_payment_behaviour_enc_filepath)
        ti.xcom_push(key='encoder_outstanding_debt_scaler_filepath', value=encoder_outstanding_debt_scaler_filepath)
        ti.xcom_push(key='encoder_occupation_enc_filepath', value=encoder_occupation_enc_filepath)
        ti.xcom_push(key='encoder_monthly_inhand_salary_scaler_filepath', value=encoder_monthly_inhand_salary_scaler_filepath)
        ti.xcom_push(key='encoder_repayment_ability_scaler_filepath', value=encoder_repayment_ability_scaler_filepath)
        ti.xcom_push(key='encoder_monthly_balance_scaler_filepath', value=encoder_monthly_balance_scaler_filepath)
        ti.xcom_push(key='encoder_fill_nulls_values_filepath', value=encoder_fill_nulls_values_filepath)
        ti.xcom_push(key='encoder_credit_mix_enc_filepath', value=encoder_credit_mix_enc_filepath)
        ti.xcom_push(key='encoder_cap_outliers_values_filepath', value=encoder_cap_outliers_values_filepath)
        ti.xcom_push(key='encoder_annual_income_scaler_filepath', value=encoder_annual_income_scaler_filepath)
        ti.xcom_push(key='encoder_amount_invested_monthly_scaler_filepath', value=encoder_amount_invested_monthly_scaler_filepath)

    elif pendulum.parse(current_date) >= pendulum.datetime(2023, 12, 1):
        active_model = 'xgb_clf'
        active_model_date = '2023-12-01'
        active_model_version = '30dpd_6mob'
        active_metric_p0 = 'recall'
        active_threshold_p0 = 0.70
        active_metric_p1 = 'f1'
        active_threshold_p1 = 0.6
        active_metric_p2 = 'gini'
        active_threshold_p2 = 0.3
        
        active_model_filename = active_model + '_' + active_model_date + '_' + active_model_version + '.joblib'
        encoder_payment_of_min_amount_enc_filename = 'encoder_payment_of_min_amount_enc_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_payment_behaviour_enc_filename = 'encoder_payment_behaviour_enc_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_outstanding_debt_scaler_filename = 'encoder_outstanding_debt_scaler_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_occupation_enc = 'encoder_occupation_enc_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_monthly_inhand_salary_scaler = 'encoder_monthly_inhand_salary_scaler_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_repayment_ability_scaler = 'encoder_repayment_ability_scaler_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_monthly_balance_scaler = 'encoder_monthly_balance_scaler_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_fill_nulls_values = 'encoder_fill_nulls_values_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_credit_mix_enc = 'encoder_credit_mix_enc_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_cap_outliers_values = 'encoder_cap_outliers_values_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_annual_income_scaler = 'encoder_annual_income_scaler_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_amount_invested_monthly_scaler = 'encoder_amount_invested_monthly_scaler_' + active_model_date + '_' + active_model_version + '_generic.joblib'

        active_model_filepath = os.path.join(model_bank_directory, active_model_filename)
        encoder_payment_of_min_amount_enc_filepath = os.path.join(model_bank_directory, encoder_payment_of_min_amount_enc_filename)
        encoder_payment_behaviour_enc_filepath = os.path.join(model_bank_directory, encoder_payment_behaviour_enc_filename)
        encoder_outstanding_debt_scaler_filepath = os.path.join(model_bank_directory, encoder_outstanding_debt_scaler_filename)
        encoder_occupation_enc_filepath = os.path.join(model_bank_directory, encoder_occupation_enc)
        encoder_monthly_inhand_salary_scaler_filepath = os.path.join(model_bank_directory, encoder_monthly_inhand_salary_scaler)
        encoder_repayment_ability_scaler_filepath = os.path.join(model_bank_directory, encoder_repayment_ability_scaler)
        encoder_monthly_balance_scaler_filepath = os.path.join(model_bank_directory, encoder_monthly_balance_scaler)
        encoder_fill_nulls_values_filepath = os.path.join(model_bank_directory, encoder_fill_nulls_values)
        encoder_credit_mix_enc_filepath = os.path.join(model_bank_directory, encoder_credit_mix_enc)
        encoder_cap_outliers_values_filepath = os.path.join(model_bank_directory, encoder_cap_outliers_values)
        encoder_annual_income_scaler_filepath = os.path.join(model_bank_directory, encoder_annual_income_scaler)
        encoder_amount_invested_monthly_scaler_filepath = os.path.join(model_bank_directory, encoder_amount_invested_monthly_scaler)

        ti.xcom_push(key='active_model', value=active_model)
        ti.xcom_push(key='active_model_date', value=active_model_date)
        ti.xcom_push(key='active_model_version', value=active_model_version)
        ti.xcom_push(key='active_metric_p0', value=active_metric_p0)
        ti.xcom_push(key='active_threshold_p0', value=active_threshold_p0)
        ti.xcom_push(key='active_metric_p1', value=active_metric_p1)
        ti.xcom_push(key='active_threshold_p1', value=active_threshold_p1)
        ti.xcom_push(key='active_metric_p2', value=active_metric_p2)
        ti.xcom_push(key='active_threshold_p2', value=active_threshold_p2)

        ti.xcom_push(key='active_model_filepath', value=active_model_filepath)
        ti.xcom_push(key='encoder_payment_of_min_amount_enc_filepath', value=encoder_payment_of_min_amount_enc_filepath)
        ti.xcom_push(key='encoder_payment_behaviour_enc_filepath', value=encoder_payment_behaviour_enc_filepath)
        ti.xcom_push(key='encoder_outstanding_debt_scaler_filepath', value=encoder_outstanding_debt_scaler_filepath)
        ti.xcom_push(key='encoder_occupation_enc_filepath', value=encoder_occupation_enc_filepath)
        ti.xcom_push(key='encoder_monthly_inhand_salary_scaler_filepath', value=encoder_monthly_inhand_salary_scaler_filepath)
        ti.xcom_push(key='encoder_repayment_ability_scaler_filepath', value=encoder_repayment_ability_scaler_filepath)
        ti.xcom_push(key='encoder_monthly_balance_scaler_filepath', value=encoder_monthly_balance_scaler_filepath)
        ti.xcom_push(key='encoder_fill_nulls_values_filepath', value=encoder_fill_nulls_values_filepath)
        ti.xcom_push(key='encoder_credit_mix_enc_filepath', value=encoder_credit_mix_enc_filepath)
        ti.xcom_push(key='encoder_cap_outliers_values_filepath', value=encoder_cap_outliers_values_filepath)
        ti.xcom_push(key='encoder_annual_income_scaler_filepath', value=encoder_annual_income_scaler_filepath)
        ti.xcom_push(key='encoder_amount_invested_monthly_scaler_filepath', value=encoder_amount_invested_monthly_scaler_filepath)

    elif pendulum.parse(current_date) >= pendulum.datetime(2023, 9, 1):
        active_model = 'lr_clf'
        active_model_date = '2023-09-01'
        active_model_version = '30dpd_6mob'
        active_metric_p0 = 'recall'
        active_threshold_p0 = 0.70
        active_metric_p1 = 'f1'
        active_threshold_p1 = 0.6
        active_metric_p2 = 'gini'
        active_threshold_p2 = 0.3
        
        active_model_filename = active_model + '_' + active_model_date + '_' + active_model_version + '.joblib'
        encoder_payment_of_min_amount_enc_filename = 'encoder_payment_of_min_amount_enc_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_payment_behaviour_enc_filename = 'encoder_payment_behaviour_enc_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_outstanding_debt_scaler_filename = 'encoder_outstanding_debt_scaler_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_occupation_enc = 'encoder_occupation_enc_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_monthly_inhand_salary_scaler = 'encoder_monthly_inhand_salary_scaler_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_repayment_ability_scaler = 'encoder_repayment_ability_scaler_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_monthly_balance_scaler = 'encoder_monthly_balance_scaler_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_fill_nulls_values = 'encoder_fill_nulls_values_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_credit_mix_enc = 'encoder_credit_mix_enc_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_cap_outliers_values = 'encoder_cap_outliers_values_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_annual_income_scaler = 'encoder_annual_income_scaler_' + active_model_date + '_' + active_model_version + '_generic.joblib'
        encoder_amount_invested_monthly_scaler = 'encoder_amount_invested_monthly_scaler_' + active_model_date + '_' + active_model_version + '_generic.joblib'

        active_model_filepath = os.path.join(model_bank_directory, active_model_filename)
        encoder_payment_of_min_amount_enc_filepath = os.path.join(model_bank_directory, encoder_payment_of_min_amount_enc_filename)
        encoder_payment_behaviour_enc_filepath = os.path.join(model_bank_directory, encoder_payment_behaviour_enc_filename)
        encoder_outstanding_debt_scaler_filepath = os.path.join(model_bank_directory, encoder_outstanding_debt_scaler_filename)
        encoder_occupation_enc_filepath = os.path.join(model_bank_directory, encoder_occupation_enc)
        encoder_monthly_inhand_salary_scaler_filepath = os.path.join(model_bank_directory, encoder_monthly_inhand_salary_scaler)
        encoder_repayment_ability_scaler_filepath = os.path.join(model_bank_directory, encoder_repayment_ability_scaler)
        encoder_monthly_balance_scaler_filepath = os.path.join(model_bank_directory, encoder_monthly_balance_scaler)
        encoder_fill_nulls_values_filepath = os.path.join(model_bank_directory, encoder_fill_nulls_values)
        encoder_credit_mix_enc_filepath = os.path.join(model_bank_directory, encoder_credit_mix_enc)
        encoder_cap_outliers_values_filepath = os.path.join(model_bank_directory, encoder_cap_outliers_values)
        encoder_annual_income_scaler_filepath = os.path.join(model_bank_directory, encoder_annual_income_scaler)
        encoder_amount_invested_monthly_scaler_filepath = os.path.join(model_bank_directory, encoder_amount_invested_monthly_scaler)

        ti.xcom_push(key='active_model', value=active_model)
        ti.xcom_push(key='active_model_date', value=active_model_date)
        ti.xcom_push(key='active_model_version', value=active_model_version)
        ti.xcom_push(key='active_metric_p0', value=active_metric_p0)
        ti.xcom_push(key='active_threshold_p0', value=active_threshold_p0)
        ti.xcom_push(key='active_metric_p1', value=active_metric_p1)
        ti.xcom_push(key='active_threshold_p1', value=active_threshold_p1)
        ti.xcom_push(key='active_metric_p2', value=active_metric_p2)
        ti.xcom_push(key='active_threshold_p2', value=active_threshold_p2)

        ti.xcom_push(key='active_model_filepath', value=active_model_filepath)
        ti.xcom_push(key='encoder_payment_of_min_amount_enc_filepath', value=encoder_payment_of_min_amount_enc_filepath)
        ti.xcom_push(key='encoder_payment_behaviour_enc_filepath', value=encoder_payment_behaviour_enc_filepath)
        ti.xcom_push(key='encoder_outstanding_debt_scaler_filepath', value=encoder_outstanding_debt_scaler_filepath)
        ti.xcom_push(key='encoder_occupation_enc_filepath', value=encoder_occupation_enc_filepath)
        ti.xcom_push(key='encoder_monthly_inhand_salary_scaler_filepath', value=encoder_monthly_inhand_salary_scaler_filepath)
        ti.xcom_push(key='encoder_repayment_ability_scaler_filepath', value=encoder_repayment_ability_scaler_filepath)
        ti.xcom_push(key='encoder_monthly_balance_scaler_filepath', value=encoder_monthly_balance_scaler_filepath)
        ti.xcom_push(key='encoder_fill_nulls_values_filepath', value=encoder_fill_nulls_values_filepath)
        ti.xcom_push(key='encoder_credit_mix_enc_filepath', value=encoder_credit_mix_enc_filepath)
        ti.xcom_push(key='encoder_cap_outliers_values_filepath', value=encoder_cap_outliers_values_filepath)
        ti.xcom_push(key='encoder_annual_income_scaler_filepath', value=encoder_annual_income_scaler_filepath)
        ti.xcom_push(key='encoder_amount_invested_monthly_scaler_filepath', value=encoder_amount_invested_monthly_scaler_filepath)

    else:
        ti.xcom_push(key='active_model_filepath', value='')
        ti.xcom_push(key='encoder_payment_of_min_amount_enc_filepath', value='')
        ti.xcom_push(key='encoder_payment_behaviour_enc_filepath', value='')
        ti.xcom_push(key='encoder_outstanding_debt_scaler_filepath', value='')
        ti.xcom_push(key='encoder_occupation_enc_filepath', value='')
        ti.xcom_push(key='encoder_monthly_inhand_salary_scaler_filepath', value='')
        ti.xcom_push(key='encoder_repayment_ability_scaler_filepath', value='')
        ti.xcom_push(key='encoder_monthly_balance_scaler_filepath', value='')
        ti.xcom_push(key='encoder_fill_nulls_values_filepath', value='')
        ti.xcom_push(key='encoder_credit_mix_enc_filepath', value='')
        ti.xcom_push(key='encoder_cap_outliers_values_filepath', value='')
        ti.xcom_push(key='encoder_annual_income_scaler_filepath', value='')
        ti.xcom_push(key='encoder_amount_invested_monthly_scaler_filepath', value='')

    logger.info(f"[{ti.task_id} | {current_date}] Active model filepath: {active_model_filepath}")
    logger.info(f"[{ti.task_id} | {current_date}] Encoder payment of min amount filepath: {encoder_payment_of_min_amount_enc_filepath}")
    logger.info(f"[{ti.task_id} | {current_date}] Encoder payment behaviour filepath: {encoder_payment_behaviour_enc_filepath}")
    logger.info(f"[{ti.task_id} | {current_date}] Encoder outstanding debt scaler filepath: {encoder_outstanding_debt_scaler_filepath}")
    logger.info(f"[{ti.task_id} | {current_date}] Encoder occupation filepath: {encoder_occupation_enc_filepath}")
    logger.info(f"[{ti.task_id} | {current_date}] Encoder monthly inhand salary scaler filepath: {encoder_monthly_inhand_salary_scaler_filepath}")
    logger.info(f"[{ti.task_id} | {current_date}] Encoder repayment ability scaler filepath: {encoder_repayment_ability_scaler_filepath}")
    logger.info(f"[{ti.task_id} | {current_date}] Encoder monthly balance scaler filepath: {encoder_monthly_balance_scaler_filepath}")
    logger.info(f"[{ti.task_id} | {current_date}] Encoder fill nulls values filepath: {encoder_fill_nulls_values_filepath}")
    logger.info(f"[{ti.task_id} | {current_date}] Encoder credit mix filepath: {encoder_credit_mix_enc_filepath}")
    logger.info(f"[{ti.task_id} | {current_date}] Encoder cap outliers values filepath: {encoder_cap_outliers_values_filepath}")
    logger.info(f"[{ti.task_id} | {current_date}] Encoder annual income scaler filepath: {encoder_annual_income_scaler_filepath}")
    logger.info(f"[{ti.task_id} | {current_date}] Encoder amount invested monthly scaler filepath: {encoder_amount_invested_monthly_scaler_filepath}")


def inference_preprocessing(ti, **context):

    current_date = context['ds']

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

    # Load gold inference filepath
    gold_inference_view = ti.xcom_pull(task_ids='create_gold_store', key='gold_inference_view')
    gold_inference_view_before_normalisation_filepath = ti.xcom_pull(task_ids='data_post_processing', key='gold_inference_view_before_normalisation')
    df_inference = spark.read.parquet(gold_inference_view_before_normalisation_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded gold inference view from: {gold_inference_view_before_normalisation_filepath}")

    current_date = context['ds']

    # Drop irrelevant columns but retain unique re-identifiers
    df_inference = df_inference.drop('feature_snapshot_date', 'loan_id', 'label_def', 'label_snapshot_date', 'label')
    logger.info(f"[{ti.task_id} | {current_date}] Completed dropping of unused ccolumns for inference dataset.")
    
    # Load encoders filepaths
    encoder_payment_of_min_amount_enc_filepath = ti.xcom_pull(task_ids='train_encoders', key='payment_of_min_amount_enc_filepath')
    encoder_payment_behaviour_enc_filepath = ti.xcom_pull(task_ids='train_encoders', key='payment_behaviour_enc_filepath')
    encoder_outstanding_debt_scaler_filepath = ti.xcom_pull(task_ids='train_encoders', key='outstanding_debt_scaler_filepath')
    encoder_occupation_enc_filepath = ti.xcom_pull(task_ids='train_encoders', key='occupation_enc_filepath')
    encoder_monthly_inhand_salary_scaler_filepath = ti.xcom_pull(task_ids='train_encoders', key='monthly_inhand_salary_scaler_filepath')
    encoder_repayment_ability_scaler_filepath = ti.xcom_pull(task_ids='train_encoders', key='repayment_ability_scaler_filepath')
    encoder_monthly_balance_scaler_filepath = ti.xcom_pull(task_ids='train_encoders', key='monthly_balance_scaler_filepath')
    encoder_fill_nulls_values_filepath = ti.xcom_pull(task_ids='train_encoders', key='fill_nulls_values_filepath')
    encoder_credit_mix_enc_filepath = ti.xcom_pull(task_ids='train_encoders', key='credit_mix_enc_filepath')
    encoder_cap_outliers_values_filepath = ti.xcom_pull(task_ids='train_encoders', key='cap_outliers_values_filepath')
    encoder_annual_income_scaler_filepath = ti.xcom_pull(task_ids='train_encoders', key='annual_income_scaler_filepath')
    encoder_amount_invested_monthly_scaler_filepath = ti.xcom_pull(task_ids='train_encoders', key='amount_invested_monthly_scaler_filepath')

    # Load values and encoders
    cap_outliers_values = joblib.load(encoder_cap_outliers_values_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded cap_outliers_values from: {encoder_cap_outliers_values_filepath}")
    fill_nulls_values = joblib.load(encoder_fill_nulls_values_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded fill_nulls_values from: {encoder_fill_nulls_values_filepath}")
    occupation_enc = joblib.load(encoder_occupation_enc_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded occupation_enc from: {encoder_occupation_enc_filepath}")
    payment_of_min_amount_enc = joblib.load(encoder_payment_of_min_amount_enc_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded payment_of_min_amount_enc from: {encoder_payment_of_min_amount_enc_filepath}")
    payment_behaviour_enc = joblib.load(encoder_payment_behaviour_enc_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded payment_behaviour_enc from: {encoder_payment_behaviour_enc_filepath}")
    credit_mix_enc = joblib.load(encoder_credit_mix_enc_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded credit_mix_enc from: {encoder_credit_mix_enc_filepath}")
    annual_income_scaler = joblib.load(encoder_annual_income_scaler_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded annual_income_scaler from: {encoder_annual_income_scaler_filepath}")
    monthly_inhand_salary_scaler = joblib.load(encoder_monthly_inhand_salary_scaler_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded monthly_inhand_salary_scaler from: {encoder_monthly_inhand_salary_scaler_filepath}")
    repayment_ability_scaler = joblib.load(encoder_repayment_ability_scaler_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded repayment_ability_scaler from: {encoder_repayment_ability_scaler_filepath}")
    outstanding_debt_scaler = joblib.load(encoder_outstanding_debt_scaler_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded outstanding_debt_scaler from: {encoder_outstanding_debt_scaler_filepath}")
    amount_invested_monthly_scaler = joblib.load(encoder_amount_invested_monthly_scaler_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded amount_invested_monthly_scaler from: {encoder_amount_invested_monthly_scaler_filepath}")
    monthly_balance_scaler = joblib.load(encoder_monthly_balance_scaler_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded monthly_balance_scaler from: {encoder_monthly_balance_scaler_filepath}")

    # Functions
    df_inference = cap_outliers_enc(df_inference, cap_outliers_values)
    logger.info(f"[{ti.task_id} | {current_date}] Encoded cap_outliers for inference dataset.")
    df_inference = fill_nulls_enc(df_inference, fill_nulls_values)
    logger.info(f"[{ti.task_id} | {current_date}] Encoded fill_null_values for inference dataset.")

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
    
    df_inference = encode_categorical(df_inference)
    logger.info(f"[{ti.task_id} | {current_date}] Completed categorical one-hot encoding for inference dataset.")

    # Ordinal encoding
    def encode_ordinal(df):
        df = df.withColumn("credit_mix", F.when(F.col("credit_mix") == "Unknown", None).otherwise(F.col("credit_mix")))

        df_pd = df.toPandas()
        df_pd['credit_mix'] = credit_mix_enc.transform(df_pd[['credit_mix']])
        df = spark.createDataFrame(df_pd)
        return df

    df_inference = encode_ordinal(df_inference)
    logger.info(f"[{ti.task_id} | {current_date}] Completed ordinal encoding for inference dataset.")

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

    df_inference = scaling(df_inference)
    logger.info(f"[{ti.task_id} | {current_date}] Completed data scaling for inference dataset.")

    # Save
    partition_name_X = gold_inference_prefix + '_' + current_date + '_ready_X.parquet'
    filepath_X = os.path.join(gold_inference_view, partition_name_X)
    df_inference.write.mode("overwrite").parquet(filepath_X)
    logger.info(f"[{ti.task_id} | {current_date}] Gold Inference View (Ready) saved to: {filepath_X}")
    ti.xcom_push(key='gold_inference_view_X_filepath', value=filepath_X)


    spark.stop()


def infer(ti, **context):

    outbox_directory = ti.xcom_pull(task_ids='create_outbox', key='outbox_directory')
    current_date = context['ds']

    infer_X_filepath = ti.xcom_pull(task_ids='inference_preprocessing', key='gold_inference_view_X_filepath')
    infer_X = pd.read_parquet(infer_X_filepath)
    logger.info(f"[{ti.task_id} | {current_date}] Loaded inference dataset with X: {infer_X.shape[0]} rows.")

    active_model_filepath = ti.xcom_pull(task_ids='deploy_model', key='active_model_filepath')
    active_model = joblib.load(active_model_filepath)

    y_pred = active_model.predict(infer_X.drop(['customer_id', 'loan_start_date'], axis=1))
    logger.info(f"[{ti.task_id} | {current_date}] Predicted 1:{y_pred[y_pred == 1].shape[0]}. Predicted 0:{y_pred[y_pred == 0].shape[0]}.")
    
    results_pd = pd.DataFrame(y_pred, columns=['prediction'])
    results = pd.concat([infer_X[['customer_id', 'loan_start_date']], results_pd], axis=1)
    results_filename_parquet = 'inference_results_' + current_date + '.parquet'
    results_filepath_parquet = os.path.join(outbox_directory, results_filename_parquet)
    results.to_parquet(results_filepath_parquet)
    results_filename_csv = 'inference_results_' + current_date + '.csv'
    results_filepath_csv = os.path.join(outbox_directory, results_filename_csv)
    results.to_csv(results_filepath_csv)
    logger.info(f"[{ti.task_id} | {current_date}] {results.shape[0]} Predictions (Parquet) saved to outbox: {results_filepath_parquet}.")
    logger.info(f"[{ti.task_id} | {current_date}] {results.shape[0]} Predictions (CSV) saved to outbox: {results_filepath_csv}.")
    ti.xcom_push(key='inference_results_filepath', value=results_filepath_parquet)

