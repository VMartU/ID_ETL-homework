from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging
import os
import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Настройка логгирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Функция 1 — Извлечение данных
def extract():
    try:
        logging.info("Загружаем датасет breast_cancer из sklearn...")
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target

        os.makedirs('/tmp/breast_cancer', exist_ok=True)
        df.to_csv('/tmp/breast_cancer/dataset.csv', index=False)
        logging.info("Данные сохранены в /tmp/breast_cancer/dataset.csv")
    except Exception as e:
        logging.error(f"Ошибка при извлечении данных: {e}")
        raise

# Функция 2 — Обучение модели
def train_model():
    try:
        logging.info("Читаем датасет и обучаем модель...")
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
        logging.info(f"Обучение завершено. Accuracy: {acc:.4f}")
    except Exception as e:
        logging.error(f"Ошибка при обучении модели: {e}")
        raise

# Функция 3 — Сохранение результатов
def save_to_local(execution_date=None, **kwargs):
    try:
        date_str = execution_date.strftime('%Y-%m-%d') if execution_date else datetime.today().strftime('%Y-%m-%d')
        source_dir = '/tmp/breast_cancer'
        target_dir = f'/home/user/airflow_project/results/{date_str}'
        os.makedirs(target_dir, exist_ok=True)

        for fname in ['model.pkl', 'metrics.txt']:
            src = os.path.join(source_dir, fname)
            dst = os.path.join(target_dir, fname)
            if os.path.exists(src):
                with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
                    fdst.write(fsrc.read())
                logging.info(f"Скопирован файл {fname} в {target_dir}")
            else:
                logging.warning(f"Файл не найден: {src}")
    except Exception as e:
        logging.error(f"Ошибка при сохранении результатов: {e}")
        raise

# Аргументы DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 6, 1),
    'retries': 2,
    'retry_delay': timedelta(minutes=1),
    'execution_timeout': timedelta(minutes=5),
}

# Определение DAG
with DAG(
    dag_id='breast_cancer_pipeline',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['ml', 'robust', 'pipeline'],
    description='ETL + ML pipeline for breast cancer classification with robustness handling'
) as dag:

    t1 = PythonOperator(
        task_id='extract_data',
        python_callable=extract,
    )

    t2 = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )

    t3 = PythonOperator(
        task_id='save_results',
        python_callable=save_to_local,
        provide_context=True,
    )

    # Последовательность выполнения
    t1 >> t2 >> t3
