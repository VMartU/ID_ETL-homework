import os
import argparse
import logging
import joblib
from sklearn.linear_model import LogisticRegression

# Настройка логгирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def train_model(x_path, y_path, output_model_path):
    # Загрузка данных
    logging.info("Загружаем X_train и y_train")
    X_train = joblib.load(x_path)
    y_train = joblib.load(y_path)

    # Инициализация и обучение модели
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    logging.info("Модель успешно обучена")

    # Сохраняем модель
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    joblib.dump(model, output_model_path)
    logging.info(f"Модель сохранена в: {output_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train logistic regression model")
    parser.add_argument("--x_path", type=str, default="results/X_train.pkl")
    parser.add_argument("--y_path", type=str, default="results/y_train.pkl")
    parser.add_argument("--output_model_path", type=str, default="results/model.pkl")
    args = parser.parse_args()

    train_model(args.x_path, args.y_path, args.output_model_path)