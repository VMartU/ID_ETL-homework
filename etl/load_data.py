import pandas as pd
import os
import argparse
import logging

# Настройка логгирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_data(input_path, output_path):
    # Имена колонок
    columns = ["id", "diagnosis"] + [f"feature_{i}" for i in range(1, 31)]
    
    # Загрузка данных
    logging.info(f"Загружаем данные из: {input_path}")
    df = pd.read_csv(input_path, header=None, names=columns)
    
    # Удаляем ID
    df = df.drop(columns=["id"])

    # Сохраняем в промежуточный файл
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"Данные сохранены в: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load breast cancer dataset.")
    parser.add_argument("--input_path", type=str, default="wdbc.data.csv", help="Путь к исходному CSV")
    parser.add_argument("--output_path", type=str, default="results/data_loaded.csv", help="Путь для сохранения")

    args = parser.parse_args()
    load_data(args.input_path, args.output_path)