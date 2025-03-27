import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import argparse
import os

def predict_gender(image_path, model_path = 'model/1/gender_model.keras'):
    """
    Предсказывает пол по изображению подчерка.
    
    Параметры:
    - model_path (str): Путь к обученной модели (.h5).
    - image_path (str): Путь к изображению для классификации.
    """
    
    # --- Проверка файлов ---
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель {model_path} не найдена!")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Изображение {image_path} не найдено!")

    # --- Загрузка модели ---
    model = tf.keras.models.load_model(model_path)

    # --- Загрузка и предобработка изображения ---
    img = image.load_img(image_path, target_size=(224, 224))  # MobileNetV2 ожидает 224x224
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Добавляем размерность батча
    img_array = img_array / 255.0  # Нормализация [0, 1]

    # --- Предсказание ---
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)  # Индекс класса с максимальной вероятностью
    confidence = prediction[0][class_idx] * 100  # Уверенность в %

    # --- Вывод результата ---
    gender = "мужской" if class_idx == 0 else "женский"
    print(f"Результат: {gender} (вероятность: {confidence:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Предсказание пола по подчерку.")
    parser.add_argument('--model', type=str, default='model/1/gender_model.keras', help='Путь к обученной модели (.h5)')
    parser.add_argument('--image', type=str, required=True, help='Путь к изображению (PNG/JPG)')
    args = parser.parse_args()
    
    predict_gender(args.image, args.model)