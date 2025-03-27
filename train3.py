import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import argparse
import os
import numpy as np

def create_data_generators(data_dir, img_size=(224, 224), batch_size=8):
    """Создает генераторы данных с правильной структурой"""
    # Усиленная аугментация для тренировочных данных
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    # Только нормализация для валидационных данных
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    # Генератор тренировочных данных
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',  # Используем binary для бинарной классификации
        subset='training',
        shuffle=True,
        seed=42
    )

    # Генератор валидационных данных
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    return train_generator, val_generator

def build_model(input_shape=(224, 224, 3)):
    """Строит модель с правильной архитектурой"""
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # 1 нейрон с сигмоидой для бинарной классификации
    ])

    return model

def train_model(data_dir='dataset', model_path='model/3/gender_model.keras',epochs = 50,batch_size = 8):
    # Параметры
    img_size = (224, 224)  # Стандартный размер для EfficientNet

    # Создаем генераторы данных
    train_gen, val_gen = create_data_generators(data_dir, img_size, batch_size)

    # Строим модель
    model = build_model(input_shape=(img_size[0], img_size[1], 3))
    
    # Компилируем модель
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Коллбэки
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True)
    ]

    # Обучение
    history = model.fit(
        train_gen,
        steps_per_epoch=max(1, train_gen.samples // batch_size),
        validation_data=val_gen,
        validation_steps=max(1, val_gen.samples // batch_size),
        epochs=epochs,
        callbacks=callbacks,
        verbose=2
    )

    # Оценка
    val_acc = history.history['val_accuracy'][-1]
    print(f"\nЛучшая точность на валидации: {val_acc:.2%}")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение модели классификации пола по почерку")
    parser.add_argument('--data_dir', default='dataset', help='Путь к директории с данными')
    parser.add_argument('--batch_size', type=int, default=8, help='')
    parser.add_argument('--epochs', type=int, default=50, help='Количество эпох обучения')
    args = parser.parse_args()
    train_model(args.data_dir,"model/3/gender_model.keras",args.epochs,args.batch_size)