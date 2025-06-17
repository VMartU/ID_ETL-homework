import argparse
import joblib
import os
import json
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Логгирование
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def evaluate(model_path, x_test_path, y_test_path, output_path):
    logging.info("Загрузка модели и тестовых данных...")
    model = joblib.load(model_path)
    X_test = joblib.load(x_test_path)
    y_test = joblib.load(y_test_path)

    logging.info("Получение предсказаний...")
    y_pred = model.predict(X_test)

    # Вычисляем метрики
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    # Сохраняем в JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    logging.info(f"Метрики сохранены в: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model_path", type=str, default="results/model.pkl")
    parser.add_argument("--x_test_path", type=str, default="results/X_test.pkl")
    parser.add_argument("--y_test_path", type=str, default="results/y_test.pkl")
    parser.add_argument("--output_path", type=str, default="results/metrics.json")
    args = parser.parse_args()

    evaluate(args.model_path, args.x_test_path, args.y_test_path, args.output_path)