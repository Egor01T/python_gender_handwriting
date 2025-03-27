import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2  # Более легкая модель, чем ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import argparse
import os

def train_model(data_dir='dataset', model_save_path='model/1/gender_model.keras', epochs=10, fine_tune=False):
    """
    Обучает модель классификации пола по изображениям подчерков.
    
    Параметры:
    - data_dir (str): Путь к папке с данными (должна содержать подпапки male/ и female/).
    - model_save_path (str): Куда сохранить модель (по умолчанию 'gender_model.h5').
    - epochs (int): Количество эпох обучения (по умолчанию 10).
    - fine_tune (bool): Если True, размораживает часть слоев для тонкой настройки.
    """

    # --- Параметры ---
    img_size = (224, 224)  # MobileNetV2 ожидает 224x224
    batch_size = 32
    num_classes = 2  # мужской / женский

    # --- Проверка структуры данных ---
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Директория {data_dir} не найдена!")
    if not all([os.path.exists(os.path.join(data_dir, cls)) for cls in ['male', 'female']]):
        raise ValueError("Должны быть подпапки 'male' и 'female' внутри data_dir!")

    # --- Загрузка предобученной модели ---
    # Используем MobileNetV2 вместо ResNet50 (быстрее и легче)
    base_model = MobileNetV2(
        weights='imagenet',  # Предобученные веса
        include_top=False,    # Без верхнего классификатора
        input_shape=(img_size[0], img_size[1], 3)
    )

    # Замораживаем все слои (чтобы не переобучать их сразу)
    base_model.trainable = False

    # --- Добавление своих слоев ---
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Усреднение карт признаков
    x = Dense(512, activation='relu')(x)  # Полносвязный слой (меньше нейронов, чем в ResNet)
    x = Dropout(0.5)(x)  # Регуляризация для борьбы с переобучением
    predictions = Dense(num_classes, activation='softmax')(x)  # Выходной слой

    # Собираем модель
    model = Model(inputs=base_model.input, outputs=predictions)

    # --- Компиляция модели ---
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # Меньше learning_rate для стабильности
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # --- Генераторы данных с аугментацией ---
    # Для тренировочных данных — аугментация
    train_datagen = ImageDataGenerator(
        rescale=1./255,          # Нормализация [0, 255] -> [0, 1]
        rotation_range=20,       # Поворот ±20°
        width_shift_range=0.2,   # Сдвиг по ширине ±20%
        height_shift_range=0.2,  # Сдвиг по высоте ±20%
        shear_range=0.2,         # Искажение
        zoom_range=0.2,          # Увеличение/уменьшение
        horizontal_flip=True,    # Горизонтальное отражение
        validation_split=0.2     # 20% данных для валидации
    )

    # Для валидации — только нормализация (без аугментации)
    val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    # Тренировочный генератор
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    # Валидационный генератор
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # --- Callbacks (контроль обучения) ---
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3),  # Остановка, если нет улучшений
        ModelCheckpoint(  # Сохраняет лучшую модель
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]

    # --- Первый этап обучения (только верхние слои) ---
    print("\n[1] Обучение верхних слоев...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size,
        epochs=epochs,
        callbacks=callbacks
    )

    # --- Fine-Tuning (если включен) ---
    if fine_tune:
        print("\n[2] Fine-Tuning (разморозка части слоев)...")
        base_model.trainable = True  # Размораживаем базовую модель

        # Размораживаем только последние 50 слоев (чтобы не переобучать все)
        for layer in base_model.layers[:-50]:
            layer.trainable = False

        # Перекомпилируем с меньшим learning_rate
        model.compile(
            optimizer=Adam(learning_rate=0.00001),  # Очень маленький LR для тонкой настройки
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Продолжаем обучение
        model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            validation_data=val_generator,
            validation_steps=val_generator.samples // batch_size,
            epochs=epochs // 2,  # Fine-tuning требует меньше эпох
            callbacks=callbacks
        )

    print(f"\nМодель сохранена в {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение модели классификации пола по подчерку.")
    parser.add_argument('--data_dir', type=str, default='dataset', help='Путь к папке с данными (должна содержать male/ и female/)')
    parser.add_argument('--model_path', type=str, default='model/1/gender_model.keras', help='Куда сохранить модель')
    parser.add_argument('--epochs', type=int, default=10, help='Количество эпох')
    parser.add_argument('--fine_tune', action='store_true', help='Включить Fine-Tuning')
    args = parser.parse_args()

    train_model(args.data_dir, args.model_path, args.epochs, args.fine_tune)