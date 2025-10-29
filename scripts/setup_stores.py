import os
import logging


# Logger
logger = logging.getLogger('ml_pipeline')  # Set the logger name
handler = logging.FileHandler('/app/ml_pipeline.log')
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# Data Store Configurations
datamart_directory = "/app/datamart"

bronze_lms_directory = "bronze/lms/"
bronze_cs_directory = "bronze/clickstream/"
bronze_att_directory = "bronze/attributes/"
bronze_fin_directory = "bronze/financials/"

silver_lms_directory = "silver/lms/"
silver_cs_directory = "silver/clickstream/"
silver_att_directory = "silver/attributes/"
silver_fin_directory = "silver/financials/"

gold_label_directory = "gold/label/"
gold_feature_directory = "gold/feature/"

gold_training_view = "gold/train_view/"
gold_validation_view = "gold/validation_view/"
gold_testing_view = "gold/test_view/"
gold_oot_view = "gold/oot_view/"

# Model Bank Configurations
model_bank_directory = "/app/model_bank"


# Tasks
def create_datamart(ti):
    if not os.path.exists(datamart_directory):
        os.makedirs(datamart_directory)
        logger.info(f"Created datamart directory: {datamart_directory}")
    else:
        logger.info(f"Datamart directory already exists: {datamart_directory}")
    ti.xcom_push(key='datamart_directory', value=datamart_directory)

def create_bronze_store(ti):
    datamart_directory = ti.xcom_pull(task_ids='create_datamart', key='datamart_directory')
    silver_store = {
        "bronze_lms_directory": os.path.join(datamart_directory, bronze_lms_directory),
        "bronze_cs_directory": os.path.join(datamart_directory, bronze_cs_directory),
        "bronze_att_directory": os.path.join(datamart_directory, bronze_att_directory),
        "bronze_fin_directory": os.path.join(datamart_directory, bronze_fin_directory)
    }
    for label, path in silver_store.items():
        if not os.path.exists(path):
            os.makedirs(path)
            logger.info(f"Created directory for {label}: {path}")
        else:
            logger.info(f"Directory for {label} already exists: {path}")
        ti.xcom_push(key=label, value=os.path.join(datamart_directory, path))

def create_silver_store(ti):
    datamart_directory = ti.xcom_pull(task_ids='create_datamart', key='datamart_directory')
    silver_store = {
        "silver_lms_directory": os.path.join(datamart_directory, silver_lms_directory),
        "silver_cs_directory": os.path.join(datamart_directory, silver_cs_directory),
        "silver_att_directory": os.path.join(datamart_directory, silver_att_directory),
        "silver_fin_directory": os.path.join(datamart_directory, silver_fin_directory)
    }
    for label, path in silver_store.items():
        if not os.path.exists(path):
            os.makedirs(path)
            logger.info(f"Created directory for {label}: {path}")
        else:
            logger.info(f"Directory for {label} already exists: {path}")
        ti.xcom_push(key=label, value=os.path.join(datamart_directory, path))

def create_gold_store(ti):
    datamart_directory = ti.xcom_pull(task_ids='create_datamart', key='datamart_directory')
    gold_store = {
        "gold_label_directory": os.path.join(datamart_directory, gold_label_directory),
        "gold_feature_directory": os.path.join(datamart_directory, gold_feature_directory),
        "gold_training_view": os.path.join(datamart_directory, gold_training_view),
        "gold_validation_view": os.path.join(datamart_directory, gold_validation_view),
        "gold_testing_view": os.path.join(datamart_directory, gold_testing_view),
        "gold_oot_view": os.path.join(datamart_directory, gold_oot_view)
    }
    for label, path in gold_store.items():
        if not os.path.exists(path):
            os.makedirs(path)
            logger.info(f"Created directory for {label}: {path}")
        else:
            logger.info(f"Directory for {label} already exists: {path}")
        ti.xcom_push(key=label, value=os.path.join(datamart_directory, path))

def create_model_bank(ti, **context):
    current_date = context['ds']
    if not os.path.exists(model_bank_directory):
        os.makedirs(model_bank_directory)
        logger.info(f"[{ti.task_id} | {current_date}] Created model bank directory: {model_bank_directory}")
    else:
        logger.info(f"[{ti.task_id} | {current_date}] Model bank directory already exists: {model_bank_directory}")
    ti.xcom_push(key='model_bank_directory', value=model_bank_directory)
