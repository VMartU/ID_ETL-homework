import pandas as pd
import os
import argparse
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Логгирование
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def preprocess_data(input_path, output_dir):
    logging.info(f"Загружаем данные из {input_path}")
    df = pd.read_csv(input_path)

    # Преобразуем целевую переменную
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    # Делим X и y
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    # Масштабируем признаки
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Разделяем на train и test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Создаем папку, если нужно
    os.makedirs(output_dir, exist_ok=True)

    # Сохраняем с помощью joblib
    joblib.dump(X_train, os.path.join(output_dir, "X_train.pkl"))
    joblib.dump(X_test, os.path.join(output_dir, "X_test.pkl"))
    joblib.dump(y_train, os.path.join(output_dir, "y_train.pkl"))
    joblib.dump(y_test, os.path.join(output_dir, "y_test.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

    logging.info(f"Данные успешно сохранены в папку: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess breast cancer data")
    parser.add_argument("--input_path", type=str, default="results/data_loaded.csv")
    parser.add_argument("--output_dir", type=str, default="results/")
    args = parser.parse_args()

    preprocess_data(args.input_path, args.output_dir)