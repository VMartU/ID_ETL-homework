from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
import os

def extract():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    os.makedirs('/tmp/breast_cancer', exist_ok=True)
    df.to_csv('/tmp/breast_cancer/dataset.csv', index=False)

def train_model():
    df = pd.read_csv('/tmp/breast_cancer/dataset.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    joblib.dump(model, '/tmp/breast_cancer/model.pkl')
    with open('/tmp/breast_cancer/metrics.txt', 'w') as f:
        f.write(f'Accuracy: {acc:.4f}')

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 6, 1),
    'retries': 1,
}

with DAG(
    dag_id='breast_cancer_pipeline',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['ml', 'demo'],
) as dag:

    t1 = PythonOperator(
        task_id='extract_data',
        python_callable=extract,
    )

    t2 = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )

    t1 >> t2
