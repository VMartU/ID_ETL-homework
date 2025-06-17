import argparse
import os
import shutil
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def save_results(model_path, metrics_path, target_dir):
    # Проверяем, что файлы существуют
    if not os.path.exists(model_path) or not os.path.exists(metrics_path):
        logging.error("Модель или метрики не найдены!")
        return

    # Создаём папку для финальных результатов
    os.makedirs(target_dir, exist_ok=True)

    # Копируем файлы
    shutil.copy(model_path, os.path.join(target_dir, "model.pkl"))
    shutil.copy(metrics_path, os.path.join(target_dir, "metrics.json"))

    logging.info(f"Финальные результаты сохранены в: {target_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save final results locally")
    parser.add_argument("--model_path", type=str, default="results/model.pkl")
    parser.add_argument("--metrics_path", type=str, default="results/metrics.json")
    parser.add_argument("--target_dir", type=str, default="results/final/")

    args = parser.parse_args()
    save_results(args.model_path, args.metrics_path, args.target_dir)