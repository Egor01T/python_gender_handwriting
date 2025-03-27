import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import argparse
import os

def train_model(data_dir='dataset', model_save_path='model/2/gender_model.keras', epochs=30, fine_tune=False):
    # --- Проверка и создание директорий ---
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Директория {data_dir} не найдена!")
    
    # --- Параметры ---
    img_size = (224, 224)
    batch_size = 32
    
    # --- Генераторы данных с улучшенной аугментацией ---
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        width_shift_range=0.25,
        height_shift_range=0.25,
        shear_range=0.25,
        zoom_range=0.25,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        validation_split=0.2,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    # --- Создание генераторов ---
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # --- Построение модели ---
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size[0], img_size[1], 3)
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(img_size[0], img_size[1], 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(2, activation='softmax')(x)

    model = Model(inputs, outputs)

    # --- Компиляция модели ---
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # --- Callbacks ---
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # --- Обучение модели ---
    print("\nНачало обучения...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # --- Тонкая настройка ---
    if fine_tune:
        print("\nНачало тонкой настройки...")
        base_model.trainable = True
        for layer in base_model.layers[:100]:
            layer.trainable = False
            
        model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            validation_data=val_generator,
            validation_steps=val_generator.samples // batch_size,
            epochs=int(epochs * 0.5),
            callbacks=callbacks,
            verbose=1
        )

    # --- Сохранение и вывод результатов ---
    val_loss, val_acc = model.evaluate(val_generator)
    print(f"\nФинальная точность на валидации: {val_acc:.2%}")
    print(f"Модель сохранена в: {model_save_path}")

    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение модели классификации пола по почерку")
    parser.add_argument('--data_dir', default='dataset', help='Путь к директории с данными')
    parser.add_argument('--model_path', default='model/2/gender_model.keras', help='Путь для сохранения модели')
    parser.add_argument('--epochs', type=int, default=30, help='Количество эпох обучения')
    parser.add_argument('--fine_tune', action='store_true', help='Использовать тонкую настройку модели')
    args = parser.parse_args()
    
    train_model(args.data_dir, args.model_path, args.epochs, args.fine_tune)