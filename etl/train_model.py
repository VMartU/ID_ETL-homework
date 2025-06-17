import os
import argparse
import logging
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError

# Настройка логгирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def train_model(x_path, y_path, output_model_path):
    try:
        logging.info("Загрузка обучающих данных...")
        if not os.path.exists(x_path):
            raise FileNotFoundError(f"Файл X_train не найден: {x_path}")
        if not os.path.exists(y_path):
            raise FileNotFoundError(f"Файл y_train не найден: {y_path}")

        X_train = joblib.load(x_path)
        y_train = joblib.load(y_path)

        logging.info("Инициализация и обучение модели LogisticRegression...")
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        logging.info("Модель успешно обучена.")

        # Сохраняем модель
        os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
        joblib.dump(model, output_model_path)
        logging.info(f"Модель сохранена по пути: {output_model_path}")

    except ValueError as ve:
        logging.error(f"Ошибка валидации данных: {ve}")
        raise
    except NotFittedError as nfe:
        logging.error(f"Модель не обучена: {nfe}")
        raise
    except Exception as e:
        logging.error(f"Ошибка при обучении модели: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train logistic regression model")
    parser.add_argument("--x_path", type=str, default="results/X_train.pkl", help="Путь к признакам")
    parser.add_argument("--y_path", type=str, default="results/y_train.pkl", help="Путь к целевым меткам")
    parser.add_argument("--output_model_path", type=str, default="results/model.pkl", help="Путь сохранения модели")

    args = parser.parse_args()
    train_model(args.x_path, args.y_path, args.output_model_path)
