from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os

default_args = {
    'owner': 'kunal',
    'start_date': datetime(2025, 3, 1),
    'retries': 1,
}

dag = DAG(
    'ml_pipeline',
    default_args=default_args,
    schedule_interval='@daily',  # Run daily
    catchup=False
)

def ingest():
    os.system("python ../scripts/data_ingestion.py")

def visualize():
    os.system("python ../scripts/data_visualisation.py")

def preprocess():
    os.system("python ../scripts/data_preprocessing.py")

def train():
    os.system("python ../scripts/model_training.py")

# def deploy():
#     os.system("python ../scripts/model_deployment.py")

ingest_task = PythonOperator(task_id='ingest_data', python_callable=ingest, dag=dag)
visualize_task = PythonOperator(task_id='visualize_data', python_callable=visualize, dag=dag)
preprocess_task = PythonOperator(task_id='preprocess_data', python_callable=preprocess, dag=dag)
train_task = PythonOperator(task_id='train_model', python_callable=train, dag=dag)
# deploy_task = PythonOperator(task_id='deploy_model', python_callable=deploy, dag=dag)

# ingest_task >> preprocess_task >> train_task >> deploy_task
ingest_task >> preprocess_task >> train_task 
