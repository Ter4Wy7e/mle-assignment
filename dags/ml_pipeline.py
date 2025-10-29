from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.models import Variable

import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import logging

from scripts.setup_stores import create_datamart, create_bronze_store, create_silver_store, create_gold_store, create_model_bank
from scripts.data_processing_bronze import data_processing_bronze
from scripts.data_processing_silver import data_processing_silver
from scripts.data_processing_gold import data_processing_gold_label
from scripts.data_processing_gold import data_processing_gold_feature
from scripts.data_processing_gold import data_post_processing
from scripts.model_training import train_encoders, training_preprocessing

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=1)
}

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
end_data_pipeline = EmptyOperator(task_id='end_data_pipeline', dag=ml_pipeline)

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
create_model_bank = PythonOperator(
    task_id='create_model_bank',
    python_callable=create_model_bank,
    start_date=datetime(2023, 6, 1),
    dag=ml_pipeline
)
data_post_processing = PythonOperator(
    task_id='data_post_processing',
    python_callable=data_post_processing,
    start_date=datetime(2023, 6, 1),
    dag=ml_pipeline
)
train_encoders = PythonOperator(
    task_id='train_encoders',
    python_callable=train_encoders,
    start_date=datetime(2023, 6, 1),
    dag=ml_pipeline
)
training_preprocessing = PythonOperator(
    task_id='training_preprocessing',
    python_callable=training_preprocessing,
    start_date=datetime(2023, 6, 1),
    dag=ml_pipeline
)





# Task dependencies
start_data_pipeline >> create_datamart >> create_bronze_store >> data_processing_bronze >> data_processing_silver
create_datamart >> create_silver_store >> data_processing_silver >> data_processing_gold_label >> create_model_bank
create_datamart >> create_silver_store >> data_processing_silver >> data_processing_gold_feature >> create_model_bank
create_datamart >> create_gold_store >> data_processing_gold_label
create_datamart >> create_gold_store >> data_processing_gold_feature

create_model_bank >> data_post_processing >> train_encoders >> training_preprocessing




'''

# Configuration - you can move these to Airflow Variables
DATA_PATH = Variable.get("data_path", default_var="/tmp/ml_data/")
MODEL_PATH = Variable.get("model_path", default_var="/tmp/models/")


def create_directories():
    """Create necessary directories for the pipeline"""
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)
    logging.info(f"Created directories: {DATA_PATH}, {MODEL_PATH}")

def data_cleaning(**kwargs):
    """
    Data cleaning and preprocessing step
    In a real scenario, this would load from your actual data source
    """
    try:
        # Simulate loading data - replace with your actual data source
        # For demonstration, creating sample data
        np.random.seed(42)
        n_samples = 1000
        
        # Create sample features
        feature_1 = np.random.normal(0, 1, n_samples)
        feature_2 = np.random.normal(5, 2, n_samples)
        feature_3 = np.random.randint(0, 10, n_samples)
        
        # Create target variable
        y = ((feature_1 > 0) & (feature_2 > 5) & (feature_3 > 3)).astype(int)
        
        # Create DataFrame
        df = pd.DataFrame({
            'feature_1': feature_1,
            'feature_2': feature_2,
            'feature_3': feature_3,
            'target': y
        })
        
        # Data cleaning steps
        # 1. Handle missing values (if any)
        df = df.fillna(method='ffill')
        
        # 2. Remove duplicates
        df = df.drop_duplicates()
        
        # 3. Feature engineering
        df['feature_1_squared'] = df['feature_1'] ** 2
        df['feature_2_log'] = np.log(df['feature_2'] + 1)  # +1 to avoid log(0)
        
        # 4. Remove outliers (simple IQR method)
        Q1 = df[['feature_1', 'feature_2']].quantile(0.25)
        Q3 = df[['feature_1', 'feature_2']].quantile(0.75)
        IQR = Q3 - Q1
        
        outlier_mask = ~((df[['feature_1', 'feature_2']] < (Q1 - 1.5 * IQR)) | 
                        (df[['feature_1', 'feature_2']] > (Q3 + 1.5 * IQR))).any(axis=1)
        df = df[outlier_mask]
        
        # Save cleaned data
        cleaned_data_path = os.path.join(DATA_PATH, 'cleaned_data.csv')
        df.to_csv(cleaned_data_path, index=False)
        
        # Push file path to XCom for downstream tasks
        kwargs['ti'].xcom_push(key='cleaned_data_path', value=cleaned_data_path)
        kwargs['ti'].xcom_push(key='data_shape', value=df.shape)
        
        logging.info(f"Data cleaning completed. Shape: {df.shape}")
        logging.info(f"Data saved to: {cleaned_data_path}")
        
    except Exception as e:
        logging.error(f"Error in data cleaning: {str(e)}")
        raise

def feature_engineering(**kwargs):
    """
    Feature engineering and preparation for model training
    """
    try:
        # Pull data path from XCom
        ti = kwargs['ti']
        cleaned_data_path = ti.xcom_pull(task_ids='data_cleaning', key='cleaned_data_path')
        
        # Load cleaned data
        df = pd.read_csv(cleaned_data_path)
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Additional feature engineering can go here
        # For example: scaling, encoding, etc.
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Save processed data
        processed_data = {
            'X_train': X_train,
            'X_test': X_test, 
            'y_train': y_train,
            'y_test': y_test
        }
        
        processed_data_path = os.path.join(DATA_PATH, 'processed_data.pkl')
        with open(processed_data_path, 'wb') as f:
            pickle.dump(processed_data, f)
        
        # Push paths to XCom
        ti.xcom_push(key='processed_data_path', value=processed_data_path)
        ti.xcom_push(key='train_shape', value=X_train.shape)
        ti.xcom_push(key='test_shape', value=X_test.shape)
        
        logging.info(f"Feature engineering completed. Train shape: {X_train.shape}")
        
    except Exception as e:
        logging.error(f"Error in feature engineering: {str(e)}")
        raise

def model_training(**kwargs):
    """
    Model training and evaluation
    """
    try:
        ti = kwargs['ti']
        processed_data_path = ti.xcom_pull(task_ids='feature_engineering', key='processed_data_path')
        
        # Load processed data
        with open(processed_data_path, 'rb') as f:
            processed_data = pickle.load(f)
        
        X_train = processed_data['X_train']
        X_test = processed_data['X_test']
        y_train = processed_data['y_train'] 
        y_test = processed_data['y_test']
        
        # Model training - using RandomForest as example
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Model evaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Generate classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Save model
        model_path = os.path.join(MODEL_PATH, f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metrics
        metrics = {
            'accuracy': accuracy,
            'classification_report': class_report,
            'training_date': datetime.now().isoformat(),
            'model_path': model_path
        }
        
        metrics_path = os.path.join(MODEL_PATH, 'latest_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Push to XCom
        ti.xcom_push(key='model_path', value=model_path)
        ti.xcom_push(key='accuracy', value=accuracy)
        ti.xcom_push(key='metrics_path', value=metrics_path)
        
        logging.info(f"Model training completed. Accuracy: {accuracy:.4f}")
        logging.info(f"Model saved to: {model_path}")
        
    except Exception as e:
        logging.error(f"Error in model training: {str(e)}")
        raise

def model_validation(**kwargs):
    """
    Validate model performance against thresholds
    """
    try:
        ti = kwargs['ti']
        accuracy = ti.xcom_pull(task_ids='model_training', key='accuracy')
        metrics_path = ti.xcom_pull(task_ids='model_training', key='metrics_path')
        
        # Load metrics
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Define validation thresholds
        ACCURACY_THRESHOLD = 0.7  # Minimum acceptable accuracy
        
        validation_result = {
            'accuracy': accuracy,
            'accuracy_threshold': ACCURACY_THRESHOLD,
            'accuracy_met': accuracy >= ACCURACY_THRESHOLD,
            'validation_passed': accuracy >= ACCURACY_THRESHOLD,
            'validation_date': datetime.now().isoformat()
        }
        
        # Save validation results
        validation_path = os.path.join(MODEL_PATH, 'validation_results.json')
        with open(validation_path, 'w') as f:
            json.dump(validation_result, f, indent=2)
        
        ti.xcom_push(key='validation_passed', value=validation_result['validation_passed'])
        ti.xcom_push(key='validation_results', value=validation_result)
        
        logging.info(f"Model validation completed. Passed: {validation_result['validation_passed']}")
        
        if not validation_result['validation_passed']:
            logging.warning(f"Model accuracy {accuracy:.4f} below threshold {ACCURACY_THRESHOLD}")
            
    except Exception as e:
        logging.error(f"Error in model validation: {str(e)}")
        raise

def batch_inference(**kwargs):
    """
    Perform batch inference on new data
    This would typically load new, unseen data for predictions
    """
    try:
        ti = kwargs['ti']
        validation_passed = ti.xcom_pull(task_ids='model_validation', key='validation_passed')
        model_path = ti.xcom_pull(task_ids='model_training', key='model_path')
        
        if not validation_passed:
            logging.warning("Skipping inference due to failed validation")
            return
        
        # Load the trained model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Simulate loading new data for inference
        # In practice, this would be your actual inference data source
        np.random.seed(123)
        n_inference_samples = 100
        
        inference_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n_inference_samples),
            'feature_2': np.random.normal(5, 2, n_inference_samples),
            'feature_3': np.random.randint(0, 10, n_inference_samples),
            'feature_1_squared': np.random.normal(0, 1, n_inference_samples) ** 2,
            'feature_2_log': np.log(np.random.normal(5, 2, n_inference_samples) + 1)
        })
        
        # Make predictions
        predictions = model.predict(inference_data)
        prediction_proba = model.predict_proba(inference_data)
        
        # Create results DataFrame
        results = inference_data.copy()
        results['prediction'] = predictions
        results['prediction_probability'] = prediction_proba.max(axis=1)
        
        # Save inference results
        inference_output_path = os.path.join(DATA_PATH, f'inference_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        results.to_csv(inference_output_path, index=False)
        
        ti.xcom_push(key='inference_results_path', value=inference_output_path)
        ti.xcom_push(key='inference_samples', value=len(results))
        
        logging.info(f"Batch inference completed. Processed {len(results)} samples")
        logging.info(f"Results saved to: {inference_output_path}")
        
    except Exception as e:
        logging.error(f"Error in batch inference: {str(e)}")
        raise

def send_success_notification(**kwargs):
    """
    Send success notification with pipeline results
    """
    ti = kwargs['ti']
    
    # Gather metrics from XCom
    accuracy = ti.xcom_pull(task_ids='model_training', key='accuracy')
    data_shape = ti.xcom_pull(task_ids='data_cleaning', key='data_shape')
    validation_passed = ti.xcom_pull(task_ids='model_validation', key='validation_passed')
    inference_samples = ti.xcom_pull(task_ids='batch_inference', key='inference_samples')
    
    subject = f"ML Pipeline Success - {datetime.now().strftime('%Y-%m-%d')}"
    
    body = f"""
    Machine Learning Pipeline completed successfully!
    
    Pipeline Results:
    - Data Shape: {data_shape}
    - Model Accuracy: {accuracy:.4f}
    - Validation Passed: {validation_passed}
    - Inference Samples Processed: {inference_samples}
    - Execution Time: {datetime.now()}
    
    The model has been trained and deployed for inference.
    """
    
    # In a real scenario, you would send this via email or other notification service
    logging.info(subject)
    logging.info(body)

# Define tasks
start_task = DummyOperator(task_id='start', dag=dag)
end_task = DummyOperator(task_id='end', dag=dag)

create_dirs_task = PythonOperator(
    task_id='create_directories',
    python_callable=create_directories,
    dag=dag,
)

data_cleaning_task = PythonOperator(
    task_id='data_cleaning',
    python_callable=data_cleaning,
    dag=dag,
)

feature_engineering_task = PythonOperator(
    task_id='feature_engineering',
    python_callable=feature_engineering,
    dag=dag,
)

model_training_task = PythonOperator(
    task_id='model_training',
    python_callable=model_training,
    dag=dag,
)

model_validation_task = PythonOperator(
    task_id='model_validation',
    python_callable=model_validation,
    dag=dag,
)

batch_inference_task = PythonOperator(
    task_id='batch_inference',
    python_callable=batch_inference,
    dag=dag,
)

success_notification_task = PythonOperator(
    task_id='send_success_notification',
    python_callable=send_success_notification,
    dag=dag,
)

# Define task dependencies
start_task >> create_dirs_task >> data_cleaning_task >> feature_engineering_task
feature_engineering_task >> model_training_task >> model_validation_task
model_validation_task >> batch_inference_task >> success_notification_task >> end_task

# Optional: Add a failure notification task
def on_failure_callback(context):
    dag_run = context.get('dag_run')
    task_instance = context.get('task_instance')
    
    error_message = f"""
    ML Pipeline Failed!
    
    DAG: {dag_run.dag_id}
    Task: {task_instance.task_id}
    Execution Date: {context.get('execution_date')}
    Exception: {context.get('exception')}
    """
    
    logging.error(error_message)

# Set failure callback for the DAG
dag.default_args['on_failure_callback'] = on_failure_callback

'''