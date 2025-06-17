import argparse
import os
import shutil
import logging

# Логгирование
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def save_results(model_path, metrics_path, target_dir):
    try:
        logging.info("Проверка наличия файлов модели и метрик...")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(f"Файл метрик не найден: {metrics_path}")

        os.makedirs(target_dir, exist_ok=True)

        shutil.copy(model_path, os.path.join(target_dir, "model.pkl"))
        shutil.copy(metrics_path, os.path.join(target_dir, "metrics.json"))

        logging.info(f"Финальные результаты успешно сохранены в: {target_dir}")
    except Exception as e:
        logging.error(f"Ошибка при сохранении результатов: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save final results locally")
    parser.add_argument("--model_path", type=str, default="results/model.pkl", help="Путь к файлу модели")
    parser.add_argument("--metrics_path", type=str, default="results/metrics.json", help="Путь к файлу метрик")
    parser.add_argument("--target_dir", type=str, default="results/final/", help="Папка для сохранения результатов")

    args = parser.parse_args()
    save_results(args.model_path, args.metrics_path, args.target_dir)
